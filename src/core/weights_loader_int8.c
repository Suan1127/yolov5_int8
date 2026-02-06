#include "weights_loader_int8.h"
#include "jsmn.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_INT8_KEYS 128
#define MAX_NAME_LEN  96

struct weights_loader_int8_t {
    int8_t* data;
    size_t data_size;
    int num_keys;
    char names[MAX_INT8_KEYS][MAX_NAME_LEN];
    size_t offset[MAX_INT8_KEYS];
    size_t numel[MAX_INT8_KEYS];
    float scale_w[MAX_INT8_KEYS];
};

static int parse_json_int(const char* json, const jsmntok_t* t) {
    char buf[32];
    int len = t->end - t->start;
    if (len <= 0 || len >= (int)sizeof(buf)) return 0;
    memcpy(buf, json + t->start, (size_t)len);
    buf[len] = '\0';
    return (int)strtol(buf, NULL, 10);
}

static float parse_json_float(const char* json, const jsmntok_t* t) {
    char buf[32];
    int len = t->end - t->start;
    if (len <= 0 || len >= (int)sizeof(buf)) return 0.f;
    memcpy(buf, json + t->start, (size_t)len);
    buf[len] = '\0';
    return (float)strtod(buf, NULL);
}

static int find_key(const char* json, jsmntok_t* tokens, int n, int obj_idx, const char* key, jsmntok_t* out_val) {
    if (obj_idx >= n || tokens[obj_idx].type != JSMN_OBJECT) return -1;
    int size = tokens[obj_idx].size;
    int i = obj_idx + 1;
    for (int p = 0; p < size && i < n; p++) {
        if (tokens[i].type != JSMN_STRING) { i++; if (i < n) i++; continue; }
        int klen = tokens[i].end - tokens[i].start;
        if (klen > 0 && (size_t)klen == strlen(key) && strncmp(json + tokens[i].start, key, (size_t)klen) == 0) {
            i++;
            if (i < n) { *out_val = tokens[i]; return i; }
        }
        i++;
        if (i < n) i++;
    }
    return -1;
}

weights_loader_int8_t* weights_loader_int8_create(const char* weights_dir) {
    if (!weights_dir) return NULL;
    char path_bin[512], path_json[512];
    snprintf(path_bin, sizeof(path_bin), "%s/weights_int8.bin", weights_dir);
    snprintf(path_json, sizeof(path_json), "%s/scales_int8.json", weights_dir);

    FILE* fp = fopen(path_bin, "rb");
    if (!fp) return NULL;
    fseek(fp, 0, SEEK_END);
    long fsize = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    if (fsize <= 0) { fclose(fp); return NULL; }
    int8_t* data = (int8_t*)malloc((size_t)fsize);
    if (!data) { fclose(fp); return NULL; }
    if (fread(data, 1, (size_t)fsize, fp) != (size_t)fsize) {
        free(data);
        fclose(fp);
        return NULL;
    }
    fclose(fp);

    size_t json_size;
    char* json = NULL;
    fp = fopen(path_json, "rb");
    if (!fp) { free(data); return NULL; }
    fseek(fp, 0, SEEK_END);
    json_size = (size_t)ftell(fp);
    fseek(fp, 0, SEEK_SET);
    json = (char*)malloc(json_size + 1);
    if (!json) { fclose(fp); free(data); return NULL; }
    if (fread(json, 1, json_size, fp) != json_size) { free(json); fclose(fp); free(data); return NULL; }
    json[json_size] = '\0';
    fclose(fp);

    jsmn_parser parser;
    jsmn_init(&parser);
    int nt = jsmn_parse(&parser, json, json_size, NULL, 0);
    if (nt < 0) { free(json); free(data); return NULL; }
    jsmntok_t* tokens = (jsmntok_t*)malloc((size_t)nt * sizeof(jsmntok_t));
    if (!tokens) { free(json); free(data); return NULL; }
    jsmn_init(&parser);
    if (jsmn_parse(&parser, json, json_size, tokens, nt) < 0) {
        free(tokens);
        free(json);
        free(data);
        return NULL;
    }

    weights_loader_int8_t* loader = (weights_loader_int8_t*)calloc(1, sizeof(weights_loader_int8_t));
    if (!loader) { free(tokens); free(json); free(data); return NULL; }
    loader->data = data;
    loader->data_size = (size_t)fsize;

    jsmntok_t root = tokens[0];
    if (root.type != JSMN_OBJECT) goto fail;
    jsmntok_t order_tok, scales_tok, shapes_tok;
    int order_idx = find_key(json, tokens, nt, 0, "order", &order_tok);
    int scales_idx = find_key(json, tokens, nt, 0, "scales", &scales_tok);
    int shapes_idx = find_key(json, tokens, nt, 0, "shapes", &shapes_tok);
    if (order_idx < 0 || scales_idx < 0 || shapes_idx < 0 || order_tok.type != JSMN_ARRAY ||
        scales_tok.type != JSMN_OBJECT || shapes_tok.type != JSMN_OBJECT)
        goto fail;

    int order_size = order_tok.size;
    int order_i = order_idx + 1;
    size_t run_offset = 0;
    for (int k = 0; k < order_size && k < MAX_INT8_KEYS && order_i < nt; k++) {
        if (tokens[order_i].type != JSMN_STRING) { order_i++; continue; }
        int namelen = tokens[order_i].end - tokens[order_i].start;
        if (namelen >= MAX_NAME_LEN) { order_i++; continue; }
        memcpy(loader->names[loader->num_keys], json + tokens[order_i].start, (size_t)namelen);
        loader->names[loader->num_keys][namelen] = '\0';
        order_i++;

        loader->offset[loader->num_keys] = run_offset;
        size_t numel = 1;
        jsmntok_t scale_val, shape_val;
        int si = find_key(json, tokens, nt, shapes_idx, loader->names[loader->num_keys], &shape_val);
        if (si >= 0 && shape_val.type == JSMN_ARRAY) {
            int ss = shape_val.size;
            int sj = si + 1;
            for (int d = 0; d < ss && d < 4 && sj < nt; d++) {
                numel *= (size_t)parse_json_int(json, &tokens[sj]);
                sj++;
            }
        }
        loader->numel[loader->num_keys] = numel;
        run_offset += numel;

        int scale_i = find_key(json, tokens, nt, scales_idx, loader->names[loader->num_keys], &scale_val);
        loader->scale_w[loader->num_keys] = (scale_i >= 0 && scale_val.type == JSMN_PRIMITIVE)
            ? parse_json_float(json, &scale_val) : 0.01f;
        loader->num_keys++;
    }

    free(tokens);
    free(json);
    return loader;
fail:
    free(tokens);
    free(json);
    if (loader) free(loader);
    free(data);
    return NULL;
}

void weights_loader_int8_free(weights_loader_int8_t* loader) {
    if (loader) {
        if (loader->data) free(loader->data);
        free(loader);
    }
}

int weights_loader_int8_get(weights_loader_int8_t* loader, const char* name,
                             const int8_t** out_ptr, float* out_scale_w, size_t* out_numel) {
    if (!loader || !name || !out_ptr || !out_scale_w || !out_numel) return -1;
    for (int i = 0; i < loader->num_keys; i++) {
        if (strcmp(loader->names[i], name) == 0) {
            *out_ptr = loader->data + loader->offset[i];
            *out_scale_w = loader->scale_w[i];
            *out_numel = loader->numel[i];
            return 0;
        }
    }
    return -1;
}
