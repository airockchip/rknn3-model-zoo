#ifndef LAYA_TEXT_H
#define LAYA_TEXT_H

#include "laya.h"
#include "nlohmann/json.hpp"

/* Ordered objects preserve the upstream option and request order. */
typedef nlohmann::ordered_json json;

const char *laya_json_string(const json &value);
int laya_load_json(const char *path, json *result);
int laya_save_json(const char *path, const json *result);
int laya_load_resources(struct laya_context *ctx, const char *root, json *config);
int laya_prepare(struct laya_context *ctx, const json &state, const json &question,
    json *keys, int *tokens, int *qtype);
int laya_postprocess(struct laya_context *ctx, const json &question, const json &keys,
    const json &config, int qtype, json *answer);

#endif
