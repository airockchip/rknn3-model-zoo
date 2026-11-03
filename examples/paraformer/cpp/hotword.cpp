// Copyright (c) 2026 Rockchip Electronics Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include "hotword.h"

#include "paraformer.h"
#include "paraformer_utils.h"
#include "rknn3_api.h"
#include "time_utils.h"

#include <ctype.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define HOTWORD_VOCAB_SIZE PARAFORMER_VOCAB_SIZE
#define HOTWORD_NO_BIAS_ID 8377
#define HOTWORD_SOS_ID 1
#define HOTWORD_UNK_ID 8403
#define HOTWORD_HIDDEN_SIZE PARAFORMER_ENCODER_DIM
#define HOTWORD_GATE_SIZE (HOTWORD_HIDDEN_SIZE * 4)
#define HOTWORD_LSTM_LAYERS 2
#define HOTWORD_MAX_NPU_INPUTS 5
#define HOTWORD_MAX_NPU_SHAPES 16
#define HOTWORD_LINE_SIZE 4096

typedef struct {
    float* storage;
    float* embedding;
    float* weight_ih[HOTWORD_LSTM_LAYERS];
    float* weight_hh[HOTWORD_LSTM_LAYERS];
    float* bias_ih[HOTWORD_LSTM_LAYERS];
    float* bias_hh[HOTWORD_LSTM_LAYERS];
} bias_encoder_weights_t;

typedef struct {
    char* text;
    int32_t id;
} token_entry_t;

typedef struct {
    char* storage;
    token_entry_t* entries;
    int count;
} token_map_t;

typedef struct {
    char* key;
    char* pieces;
} seg_entry_t;

typedef struct {
    char* storage;
    seg_entry_t* entries;
    int count;
} seg_table_t;

typedef struct {
    rknn3_context context;
    rknn3_tensor_mem* input_mem[HOTWORD_MAX_NPU_INPUTS];
    rknn3_tensor_mem* output_mem;
    rknn3_tensor_attr input_attrs[HOTWORD_MAX_NPU_INPUTS];
    rknn3_tensor_attr output_attr;
    rknn3_tensor_attr shape_input_attrs[HOTWORD_MAX_NPU_SHAPES][HOTWORD_MAX_NPU_INPUTS];
    rknn3_tensor_attr shape_output_attrs[HOTWORD_MAX_NPU_SHAPES];
    int32_t shape_ids[HOTWORD_MAX_NPU_SHAPES];
    uint32_t input_count;
    uint32_t shape_count;
    int is_dynamic;
    const char* name;
} hotword_npu_model_t;

struct paraformer_hotword_context {
    hotword_npu_model_t seaco;
    hotword_npu_model_t asf;
    bias_encoder_weights_t bias_weights;
    int32_t* hotword_ids;
    int32_t* hotword_lengths;
    float* contextual_info;
    int hotword_count;
    int max_hotword_len;
    float seaco_weight;
    int asf_nfilter;
};

/** Returns the smaller of two signed dimensions. */
static int min_int(int a, int b) { return a < b ? a : b; }

/** Returns the element count represented by an RKNN3 tensor shape. */
static size_t tensor_numel(const rknn3_tensor_attr* attr) {
    size_t count = 1;
    for (uint32_t index = 0; index < attr->n_dims; ++index) {
        count *= attr->shape[index];
    }
    return count;
}

/** Returns the time dimension used by the hotword model interfaces. */
static int tensor_time(const rknn3_tensor_attr* attr) { return attr && attr->n_dims >= 2 ? (int)attr->shape[1] : 1; }

/** Returns the final feature dimension of one tensor. */
static int tensor_last_dim(const rknn3_tensor_attr* attr) {
    return attr && attr->n_dims > 0 ? (int)attr->shape[attr->n_dims - 1] : 1;
}

/** Reads an entire binary resource into caller-owned memory. */
static void* load_binary_file(const char* path, size_t* size) {
    FILE* file = NULL;
    long file_size = 0;
    void* data = NULL;

    if (!path || !size)
        return NULL;
    file = fopen(path, "rb");
    if (!file) {
        printf("Error: cannot open %s\n", path);
        return NULL;
    }
    if (fseek(file, 0, SEEK_END) != 0 || (file_size = ftell(file)) < 0 || fseek(file, 0, SEEK_SET) != 0) {
        fclose(file);
        return NULL;
    }
    data = malloc((size_t)file_size);
    if (!data || fread(data, 1, (size_t)file_size, file) != (size_t)file_size) {
        free(data);
        fclose(file);
        return NULL;
    }
    fclose(file);
    *size = (size_t)file_size;
    return data;
}

/** Compares two vocabulary entries by UTF-8 token text. */
static int compare_token_entries(const void* left, const void* right) {
    const token_entry_t* lhs = (const token_entry_t*)left;
    const token_entry_t* rhs = (const token_entry_t*)right;
    return strcmp(lhs->text, rhs->text);
}

/** Compares two segmentation entries by UTF-8 key. */
static int compare_seg_entries(const void* left, const void* right) {
    const seg_entry_t* lhs = (const seg_entry_t*)left;
    const seg_entry_t* rhs = (const seg_entry_t*)right;
    return strcmp(lhs->key, rhs->key);
}

/** Releases a parsed token map. */
static void release_token_map(token_map_t* map) {
    if (!map)
        return;
    free(map->entries);
    free(map->storage);
    memset(map, 0, sizeof(*map));
}

/** Parses the vocabulary JSON string array and sorts it for binary lookup. */
static int load_token_map(const char* path, token_map_t* map) {
    FILE* file = NULL;
    long file_size = 0;
    size_t capacity = 256;
    int in_quote = 0;
    int escaping = 0;
    char* token_start = NULL;
    char* write_pointer = NULL;

    if (!path || !map)
        return -1;
    memset(map, 0, sizeof(*map));
    file = fopen(path, "rb");
    if (!file)
        return -1;
    if (fseek(file, 0, SEEK_END) != 0 || (file_size = ftell(file)) <= 0 || fseek(file, 0, SEEK_SET) != 0) {
        fclose(file);
        return -1;
    }
    map->storage = (char*)malloc((size_t)file_size + 1);
    map->entries = (token_entry_t*)malloc(capacity * sizeof(token_entry_t));
    if (!map->storage || !map->entries || fread(map->storage, 1, (size_t)file_size, file) != (size_t)file_size) {
        fclose(file);
        release_token_map(map);
        return -1;
    }
    fclose(file);
    map->storage[file_size] = '\0';

    for (long index = 0; index < file_size; ++index) {
        char value = map->storage[index];
        if (!in_quote) {
            if (value == '"') {
                in_quote = 1;
                token_start = map->storage + index + 1;
                write_pointer = token_start;
            }
            continue;
        }
        if (escaping) {
            *write_pointer++ = value;
            escaping = 0;
        } else if (value == '\\') {
            escaping = 1;
        } else if (value == '"') {
            token_entry_t* resized = NULL;
            *write_pointer = '\0';
            if ((size_t)map->count == capacity) {
                capacity *= 2;
                resized = (token_entry_t*)realloc(map->entries, capacity * sizeof(token_entry_t));
                if (!resized) {
                    release_token_map(map);
                    return -1;
                }
                map->entries = resized;
            }
            map->entries[map->count].text = token_start;
            map->entries[map->count].id = map->count;
            ++map->count;
            in_quote = 0;
        } else {
            *write_pointer++ = value;
        }
    }
    if (map->count <= HOTWORD_UNK_ID || strcmp(map->entries[HOTWORD_UNK_ID].text, "<unk>") != 0) {
        release_token_map(map);
        return -1;
    }
    qsort(map->entries, (size_t)map->count, sizeof(token_entry_t), compare_token_entries);
    return 0;
}

/** Finds one UTF-8 token ID and maps unknown pieces to the vocabulary UNK. */
static int32_t find_token_id(const token_map_t* map, const char* token) {
    int low = 0;
    int high = map ? map->count - 1 : -1;
    while (low <= high) {
        int middle = low + (high - low) / 2;
        int comparison = strcmp(token, map->entries[middle].text);
        if (comparison == 0)
            return map->entries[middle].id;
        if (comparison < 0)
            high = middle - 1;
        else
            low = middle + 1;
    }
    return HOTWORD_UNK_ID;
}

/** Releases a segmentation dictionary parsed in place. */
static void release_seg_table(seg_table_t* table) {
    if (!table)
        return;
    free(table->entries);
    free(table->storage);
    memset(table, 0, sizeof(*table));
}

/** Loads `seg_dict` into sorted key-to-piece-list entries. */
static int load_seg_table(const char* path, seg_table_t* table) {
    size_t size = 0;
    size_t capacity = 1;
    char* resized_storage = NULL;
    char* cursor = NULL;
    char* end = NULL;

    if (!path || !table)
        return -1;
    memset(table, 0, sizeof(*table));
    table->storage = (char*)load_binary_file(path, &size);
    if (!table->storage)
        return -1;
    resized_storage = (char*)realloc(table->storage, size + 1);
    if (!resized_storage) {
        release_seg_table(table);
        return -1;
    }
    table->storage = resized_storage;
    table->storage[size] = '\0';
    for (size_t index = 0; index < size; ++index) {
        if (table->storage[index] == '\n')
            ++capacity;
    }
    table->entries = (seg_entry_t*)calloc(capacity, sizeof(seg_entry_t));
    if (!table->entries) {
        release_seg_table(table);
        return -1;
    }

    cursor = table->storage;
    end = table->storage + size;
    while (cursor < end) {
        char* line = cursor;
        char* split = NULL;
        size_t line_length = 0;
        while (cursor < end && *cursor != '\n')
            ++cursor;
        if (cursor < end)
            *cursor++ = '\0';
        line_length = strlen(line);
        if (line_length > 0 && line[line_length - 1] == '\r') {
            line[line_length - 1] = '\0';
        }
        while (*line && isspace((unsigned char)*line))
            ++line;
        if (!*line)
            continue;
        split = line;
        while (*split && !isspace((unsigned char)*split))
            ++split;
        if (!*split)
            continue;
        *split++ = '\0';
        while (*split && isspace((unsigned char)*split))
            ++split;
        if (!*split)
            continue;
        table->entries[table->count].key = line;
        table->entries[table->count].pieces = split;
        ++table->count;
    }
    qsort(table->entries, (size_t)table->count, sizeof(seg_entry_t), compare_seg_entries);
    return table->count > 0 ? 0 : -1;
}

/** Finds a segmentation dictionary entry by exact UTF-8 key. */
static const char* find_seg_pieces(const seg_table_t* table, const char* key) {
    int low = 0;
    int high = table ? table->count - 1 : -1;
    while (low <= high) {
        int middle = low + (high - low) / 2;
        int comparison = strcmp(key, table->entries[middle].key);
        if (comparison == 0)
            return table->entries[middle].pieces;
        if (comparison < 0)
            high = middle - 1;
        else
            low = middle + 1;
    }
    return NULL;
}

/** Lowercases ASCII letters without changing UTF-8 multibyte sequences. */
static void ascii_lower_in_place(char* text) {
    for (; text && *text; ++text) {
        unsigned char value = (unsigned char)*text;
        if (value < 128)
            *text = (char)tolower(value);
    }
}

/** Returns the byte length and codepoint of one leading UTF-8 character. */
static int decode_utf8_character(const char* text, uint32_t* codepoint) {
    const unsigned char* data = (const unsigned char*)text;
    if (!text || !*text || !codepoint)
        return 0;
    if ((data[0] & 0x80) == 0) {
        *codepoint = data[0];
        return 1;
    }
    if ((data[0] & 0xe0) == 0xc0 && data[1]) {
        *codepoint = ((uint32_t)(data[0] & 0x1f) << 6) | (data[1] & 0x3f);
        return 2;
    }
    if ((data[0] & 0xf0) == 0xe0 && data[1] && data[2]) {
        *codepoint = ((uint32_t)(data[0] & 0x0f) << 12) | ((uint32_t)(data[1] & 0x3f) << 6) | (data[2] & 0x3f);
        return 3;
    }
    if ((data[0] & 0xf8) == 0xf0 && data[1] && data[2] && data[3]) {
        *codepoint = ((uint32_t)(data[0] & 0x07) << 18) | ((uint32_t)(data[1] & 0x3f) << 12) |
                     ((uint32_t)(data[2] & 0x3f) << 6) | (data[3] & 0x3f);
        return 4;
    }
    *codepoint = data[0];
    return 1;
}

/** Checks whether a word consists only of CJK characters or ASCII digits. */
static int is_cjk_or_digit_word(const char* word) {
    const char* cursor = word;
    if (!cursor || !*cursor)
        return 0;
    while (*cursor) {
        uint32_t codepoint = 0;
        int bytes = decode_utf8_character(cursor, &codepoint);
        if (bytes <= 0 || !((codepoint >= '0' && codepoint <= '9') || (codepoint >= 0x4e00 && codepoint <= 0x9fa5))) {
            return 0;
        }
        cursor += bytes;
    }
    return 1;
}

/** Appends whitespace-separated segmentation pieces as token IDs. */
static void append_seg_piece_ids(const char* pieces, const token_map_t* token_map, int32_t* ids, int* count,
                                 int limit) {
    const char* cursor = pieces;
    while (cursor && *cursor && *count < limit) {
        char token[256];
        size_t length = 0;
        size_t copy_length = 0;
        while (*cursor && isspace((unsigned char)*cursor))
            ++cursor;
        while (cursor[length] && !isspace((unsigned char)cursor[length]))
            ++length;
        if (length == 0)
            break;
        copy_length = length < sizeof(token) ? length : sizeof(token) - 1;
        memcpy(token, cursor, copy_length);
        token[copy_length] = '\0';
        ids[(*count)++] = find_token_id(token_map, token);
        cursor += length;
    }
}

/** Tokenizes one FunASR-style hotword line into vocabulary IDs. */
static int tokenize_hotword_line(char* line, const token_map_t* token_map, const seg_table_t* seg_table, int32_t* ids,
                                 int limit) {
    char* cursor = line;
    int count = 0;
    ascii_lower_in_place(line);
    while (*cursor && count < limit) {
        char* word = NULL;
        char saved = '\0';
        const char* pieces = NULL;
        while (*cursor && isspace((unsigned char)*cursor))
            ++cursor;
        if (!*cursor)
            break;
        word = cursor;
        while (*cursor && !isspace((unsigned char)*cursor))
            ++cursor;
        saved = *cursor;
        *cursor = '\0';
        pieces = find_seg_pieces(seg_table, word);
        if (pieces) {
            append_seg_piece_ids(pieces, token_map, ids, &count, limit);
        } else if (is_cjk_or_digit_word(word)) {
            const char* character = word;
            while (*character && count < limit) {
                char utf8[8];
                uint32_t codepoint = 0;
                int bytes = decode_utf8_character(character, &codepoint);
                if (bytes <= 0 || bytes >= (int)sizeof(utf8))
                    break;
                memcpy(utf8, character, (size_t)bytes);
                utf8[bytes] = '\0';
                pieces = find_seg_pieces(seg_table, utf8);
                if (pieces)
                    append_seg_piece_ids(pieces, token_map, ids, &count, limit);
                else
                    ids[count++] = find_token_id(token_map, utf8);
                character += bytes;
            }
        } else {
            ids[count++] = find_token_id(token_map, word);
        }
        *cursor = saved;
    }
    return count;
}

/** Reads text hotwords and appends the SOS-seeded no-bias context row. */
static int load_hotword_ids(const char* hotwords_path, const char* tokens_path, const char* seg_dict_path,
                            int max_hotwords, int max_hotword_len, int32_t** ids, int32_t** lengths,
                            int* hotword_count) {
    token_map_t token_map;
    seg_table_t seg_table;
    FILE* file = NULL;
    char line[HOTWORD_LINE_SIZE];
    int count = 0;
    int result = -1;

    memset(&token_map, 0, sizeof(token_map));
    memset(&seg_table, 0, sizeof(seg_table));
    if (!ids || !lengths || !hotword_count)
        return -1;
    *ids = NULL;
    *lengths = NULL;
    *hotword_count = 0;
    if (max_hotwords < 1 || max_hotword_len < 1 || load_token_map(tokens_path, &token_map) != 0 ||
        load_seg_table(seg_dict_path, &seg_table) != 0) {
        goto cleanup;
    }
    *ids = (int32_t*)calloc((size_t)max_hotwords * max_hotword_len, sizeof(int32_t));
    *lengths = (int32_t*)calloc((size_t)max_hotwords, sizeof(int32_t));
    if (!*ids || !*lengths)
        goto cleanup;

    file = fopen(hotwords_path, "rb");
    if (!file)
        goto cleanup;
    while (count < max_hotwords - 1 && fgets(line, sizeof(line), file)) {
        size_t length = strlen(line);
        int token_count = 0;
        if (length == sizeof(line) - 1 && line[length - 1] != '\n' && !feof(file)) {
            printf("Error: hotword line exceeds %u bytes\n", HOTWORD_LINE_SIZE - 1);
            goto cleanup;
        }
        while (length > 0 && isspace((unsigned char)line[length - 1])) {
            line[--length] = '\0';
        }
        if (length == 0)
            continue;
        token_count = tokenize_hotword_line(line, &token_map, &seg_table, *ids + (size_t)count * max_hotword_len,
                                            max_hotword_len);
        if (token_count > 0) {
            (*lengths)[count] = token_count;
            ++count;
        }
    }
    (*ids)[(size_t)count * max_hotword_len] = HOTWORD_SOS_ID;
    (*lengths)[count] = 1;
    ++count;
    *hotword_count = count;
    result = 0;

cleanup:
    if (file)
        fclose(file);
    release_seg_table(&seg_table);
    release_token_map(&token_map);
    if (result != 0) {
        free(*ids);
        free(*lengths);
        *ids = NULL;
        *lengths = NULL;
    }
    return result;
}

/** Releases the shared FP32 BiasEncoder parameter buffer. */
static void release_bias_weights(bias_encoder_weights_t* weights) {
    if (!weights)
        return;
    free(weights->storage);
    memset(weights, 0, sizeof(*weights));
}

/** Loads one packed FP32 file and binds the embedding and two LSTM layers. */
static int load_bias_weights(const char* path, bias_encoder_weights_t* weights) {
    const size_t embedding_count = (size_t)HOTWORD_VOCAB_SIZE * HOTWORD_HIDDEN_SIZE;
    const size_t matrix_count = (size_t)HOTWORD_GATE_SIZE * HOTWORD_HIDDEN_SIZE;
    const size_t bias_count = HOTWORD_GATE_SIZE;
    const size_t expected_count =
        embedding_count + HOTWORD_LSTM_LAYERS * (matrix_count * 2 + bias_count * 2);
    size_t bytes = 0;
    float* cursor = NULL;
    if (!path || !weights)
        return -1;
    memset(weights, 0, sizeof(*weights));
    weights->storage = (float*)load_binary_file(path, &bytes);
    if (!weights->storage || bytes != expected_count * sizeof(float)) {
        printf("Error: BiasEncoder parameter size mismatch for %s: got=%zu expected=%zu\n", path, bytes,
               expected_count * sizeof(float));
        release_bias_weights(weights);
        return -1;
    }

    cursor = weights->storage;
    weights->embedding = cursor;
    cursor += embedding_count;
    for (int layer = 0; layer < HOTWORD_LSTM_LAYERS; ++layer) {
        weights->weight_ih[layer] = cursor;
        cursor += matrix_count;
        weights->weight_hh[layer] = cursor;
        cursor += matrix_count;
        weights->bias_ih[layer] = cursor;
        cursor += bias_count;
        weights->bias_hh[layer] = cursor;
        cursor += bias_count;
    }
    return 0;
}

/** Computes a numerically stable sigmoid for one LSTM gate value. */
static float sigmoid_value(float value) {
    if (value >= 0.0f) {
        float term = expf(-value);
        return 1.0f / (1.0f + term);
    }
    {
        float term = expf(value);
        return term / (1.0f + term);
    }
}

/** Computes `output = weight * input + bias` for one row-major matrix. */
static void matrix_vector_add(const float* weight, const float* input, const float* bias, float* output, int input_size,
                              int output_size) {
    for (int row = 0; row < output_size; ++row) {
        const float* weight_row = weight + (size_t)row * input_size;
        float sum = bias ? bias[row] : 0.0f;
        for (int column = 0; column < input_size; ++column) {
            sum += weight_row[column] * input[column];
        }
        output[row] = sum;
    }
}

/** Runs one LSTM layer and skips padded timesteps after each phrase length. */
static int run_lstm_layer(const float* weight_ih, const float* weight_hh, const float* bias_ih, const float* bias_hh,
                          const float* input_sequence, const int32_t* lengths, int sequence_length, int batch_size,
                          float* output_sequence) {
    float* hidden = NULL;
    float* cell = NULL;
    float* input_gates = NULL;
    float* hidden_gates = NULL;
    float* gates = NULL;
    int result = -1;

    hidden = (float*)calloc((size_t)batch_size * HOTWORD_HIDDEN_SIZE, sizeof(float));
    cell = (float*)calloc((size_t)batch_size * HOTWORD_HIDDEN_SIZE, sizeof(float));
    input_gates = (float*)malloc(HOTWORD_GATE_SIZE * sizeof(float));
    hidden_gates = (float*)malloc(HOTWORD_GATE_SIZE * sizeof(float));
    gates = (float*)malloc(HOTWORD_GATE_SIZE * sizeof(float));
    if (!hidden || !cell || !input_gates || !hidden_gates || !gates)
        goto cleanup;

    memset(output_sequence, 0, (size_t)sequence_length * batch_size * HOTWORD_HIDDEN_SIZE * sizeof(float));
    for (int time = 0; time < sequence_length; ++time) {
        for (int batch = 0; batch < batch_size; ++batch) {
            const float* input = NULL;
            float* hidden_row = NULL;
            float* cell_row = NULL;
            float* output = NULL;
            if (time >= lengths[batch])
                continue;
            input = input_sequence + ((size_t)time * batch_size + batch) * HOTWORD_HIDDEN_SIZE;
            hidden_row = hidden + (size_t)batch * HOTWORD_HIDDEN_SIZE;
            cell_row = cell + (size_t)batch * HOTWORD_HIDDEN_SIZE;
            output = output_sequence + ((size_t)time * batch_size + batch) * HOTWORD_HIDDEN_SIZE;

            matrix_vector_add(weight_ih, input, bias_ih, input_gates, HOTWORD_HIDDEN_SIZE, HOTWORD_GATE_SIZE);
            matrix_vector_add(weight_hh, hidden_row, bias_hh, hidden_gates, HOTWORD_HIDDEN_SIZE, HOTWORD_GATE_SIZE);
            for (int gate = 0; gate < HOTWORD_GATE_SIZE; ++gate) {
                gates[gate] = input_gates[gate] + hidden_gates[gate];
            }
            for (int dimension = 0; dimension < HOTWORD_HIDDEN_SIZE; ++dimension) {
                float input_gate = sigmoid_value(gates[dimension]);
                float forget_gate = sigmoid_value(gates[HOTWORD_HIDDEN_SIZE + dimension]);
                float cell_gate = tanhf(gates[2 * HOTWORD_HIDDEN_SIZE + dimension]);
                float output_gate = sigmoid_value(gates[3 * HOTWORD_HIDDEN_SIZE + dimension]);
                float next_cell = forget_gate * cell_row[dimension] + input_gate * cell_gate;
                cell_row[dimension] = next_cell;
                output[dimension] = output_gate * tanhf(next_cell);
                hidden_row[dimension] = output[dimension];
            }
        }
    }
    result = 0;

cleanup:
    free(gates);
    free(hidden_gates);
    free(input_gates);
    free(cell);
    free(hidden);
    return result;
}

/** Runs the CPU BiasEncoder and gathers each phrase's final valid timestep. */
static int run_bias_encoder(const bias_encoder_weights_t* weights, const int32_t* ids, const int32_t* lengths,
                            int hotword_count, int max_hotword_len, float* contextual_info) {
    size_t sequence_count = (size_t)max_hotword_len * hotword_count * HOTWORD_HIDDEN_SIZE;
    float* embedded = (float*)calloc(sequence_count, sizeof(float));
    float* buffer_a = (float*)malloc(sequence_count * sizeof(float));
    float* buffer_b = (float*)malloc(sequence_count * sizeof(float));
    const float* layer_input = embedded;
    int result = -1;

    if (!embedded || !buffer_a || !buffer_b)
        goto cleanup;
    for (int batch = 0; batch < hotword_count; ++batch) {
        int valid_length = min_int(lengths[batch], max_hotword_len);
        for (int time = 0; time < valid_length; ++time) {
            int32_t id = ids[(size_t)batch * max_hotword_len + time];
            const float* source = NULL;
            float* destination = NULL;
            if (id < 0 || id >= HOTWORD_VOCAB_SIZE)
                id = 0;
            source = weights->embedding + (size_t)id * HOTWORD_HIDDEN_SIZE;
            destination = embedded + ((size_t)time * hotword_count + batch) * HOTWORD_HIDDEN_SIZE;
            memcpy(destination, source, HOTWORD_HIDDEN_SIZE * sizeof(float));
        }
    }

    for (int layer = 0; layer < HOTWORD_LSTM_LAYERS; ++layer) {
        float* layer_output = layer == 0 ? buffer_a : buffer_b;
        if (run_lstm_layer(weights->weight_ih[layer], weights->weight_hh[layer], weights->bias_ih[layer],
                           weights->bias_hh[layer], layer_input, lengths, max_hotword_len, hotword_count,
                           layer_output) != 0) {
            goto cleanup;
        }
        layer_input = layer_output;
    }

    for (int batch = 0; batch < hotword_count; ++batch) {
        int valid_length = lengths[batch];
        const float* source = NULL;
        if (valid_length < 1)
            valid_length = 1;
        if (valid_length > max_hotword_len)
            valid_length = max_hotword_len;
        source = layer_input + ((size_t)(valid_length - 1) * hotword_count + batch) * HOTWORD_HIDDEN_SIZE;
        memcpy(contextual_info + (size_t)batch * HOTWORD_HIDDEN_SIZE, source, HOTWORD_HIDDEN_SIZE * sizeof(float));
    }
    result = 0;

cleanup:
    free(buffer_b);
    free(buffer_a);
    free(embedded);
    return result;
}

/** Releases tensor memory and the RKNN3 context owned by one optional model. */
static void release_npu_model(hotword_npu_model_t* model) {
    if (!model)
        return;
    for (uint32_t index = 0; index < model->input_count; ++index) {
        if (model->input_mem[index]) {
            rknn3_destroy_mem(model->context, model->input_mem[index]);
        }
    }
    if (model->output_mem) {
        rknn3_destroy_mem(model->context, model->output_mem);
    }
    if (model->context) {
        rknn3_destroy(model->context);
    }
    memset(model, 0, sizeof(*model));
}

/** Verifies that an optional model was compiled for the requested NPU mask. */
static int validate_core_mask(rknn3_context context, const char* name, uint32_t core_mask) {
    uint32_t core_count = 0;
    uint32_t expected_mask = 0;
    int ret = rknn3_query(context, RKNN3_QUERY_CORE_NUMBER, &core_count, sizeof(core_count));
    if (ret != RKNN3_SUCCESS || core_count == 0 || core_count > 32) {
        printf("Error: cannot query %s core count: ret=%d count=%u\n", name, ret, core_count);
        return -1;
    }
    expected_mask = core_count == 32 ? UINT32_MAX : (uint32_t)((1ULL << core_count) - 1ULL);
    if (core_mask != expected_mask) {
        printf("Error: %s is a %u-core model but core mask is 0x%x, expected 0x%x\n", name, core_count, core_mask,
               expected_mask);
        return -1;
    }
    return 0;
}

/** Validates direct-memory FP16 input and output attributes. */
static int validate_npu_attributes(const hotword_npu_model_t* model) {
    for (uint32_t index = 0; index < model->input_count; ++index) {
        const rknn3_tensor_attr* attr = &model->input_attrs[index];
        if (attr->dtype != RKNN3_TENSOR_FLOAT16 || attr->layout != RKNN3_TENSOR_UNDEFINED) {
            printf("Error: %s input%u must be FP16 with undefined layout, got dtype=%d layout=%d\n", model->name, index,
                   attr->dtype, attr->layout);
            return -1;
        }
    }
    if (model->output_attr.dtype != RKNN3_TENSOR_FLOAT16 || model->output_attr.layout != RKNN3_TENSOR_UNDEFINED) {
        printf("Error: %s output must be FP16 with undefined layout, got dtype=%d layout=%d\n", model->name,
               model->output_attr.dtype, model->output_attr.layout);
        return -1;
    }
    return 0;
}

/** Loads one SeaCo/ASF RKNN pair and records every available shape. */
static int init_npu_model(const char* model_path, const char* weight_path, const char* name, uint32_t expected_inputs,
                          uint32_t core_mask, hotword_npu_model_t* model) {
    rknn3_input_output_num io_count;
    rknn3_shape_config shape_config;
    rknn3_config config;
    int ret = -1;

    if (!model_path || !weight_path || !name || !model)
        return -1;
    memset(model, 0, sizeof(*model));
    model->name = name;
    ret = rknn3_init(&model->context, NULL);
    if (ret != RKNN3_SUCCESS)
        goto failed;

    ret = rknn3_load_model_from_path(model->context, model_path, weight_path);
    if (ret != RKNN3_SUCCESS || validate_core_mask(model->context, name, core_mask) != 0) {
        goto failed;
    }

    memset(&config, 0, sizeof(config));
    config.run_core_mask = core_mask;
    config.user_mem_internal = 1;
    ret = rknn3_model_init(model->context, &config);
    if (ret != RKNN3_SUCCESS)
        goto failed;

    memset(&io_count, 0, sizeof(io_count));
    ret = rknn3_query(model->context, RKNN3_QUERY_IN_OUT_NUM, &io_count, sizeof(io_count));
    if (ret != RKNN3_SUCCESS || io_count.n_input != expected_inputs || io_count.n_output != 1 ||
        expected_inputs > HOTWORD_MAX_NPU_INPUTS) {
        printf("Error: invalid %s I/O count: inputs=%u outputs=%u expected=%u/1\n", name, io_count.n_input,
               io_count.n_output, expected_inputs);
        goto failed;
    }
    model->input_count = io_count.n_input;
    for (uint32_t index = 0; index < model->input_count; ++index) {
        model->input_attrs[index].index = index;
        ret =
            rknn3_query(model->context, RKNN3_QUERY_INPUT_ATTR, &model->input_attrs[index], sizeof(rknn3_tensor_attr));
        if (ret != RKNN3_SUCCESS)
            goto failed;
        model->input_mem[index] = rknn3_create_mem(model->context, model->input_attrs[index].aligned_size,
                                                   model->input_attrs[index].core_id, RKNN3_FLAG_MEMORY_CACHEABLE);
        if (!model->input_mem[index])
            goto failed;
    }
    model->output_attr.index = 0;
    ret = rknn3_query(model->context, RKNN3_QUERY_OUTPUT_ATTR, &model->output_attr, sizeof(model->output_attr));
    if (ret != RKNN3_SUCCESS || validate_npu_attributes(model) != 0)
        goto failed;
    model->output_mem = rknn3_create_mem(model->context, model->output_attr.aligned_size, model->output_attr.core_id,
                                         RKNN3_FLAG_MEMORY_CACHEABLE);
    if (!model->output_mem)
        goto failed;

    memset(&shape_config, 0, sizeof(shape_config));
    ret = rknn3_query(model->context, RKNN3_QUERY_DYNAMIC_SHAPE_CONFIG, &shape_config, sizeof(shape_config));
    if (ret == RKNN3_SUCCESS && shape_config.n_shapes > 1) {
        if (shape_config.n_shapes > HOTWORD_MAX_NPU_SHAPES) {
            printf("Error: %s shape count %u exceeds %u\n", name, shape_config.n_shapes, HOTWORD_MAX_NPU_SHAPES);
            goto failed;
        }
        model->is_dynamic = 1;
        model->shape_count = shape_config.n_shapes;
        for (uint32_t shape = 0; shape < model->shape_count; ++shape) {
            int32_t shape_id = (int32_t)shape;
            ret = rknn3_set_shape(model->context, shape_id);
            if (ret != RKNN3_SUCCESS)
                goto failed;
            model->shape_ids[shape] = shape_id;
            for (uint32_t input = 0; input < model->input_count; ++input) {
                rknn3_tensor_attr* attr = &model->shape_input_attrs[shape][input];
                memset(attr, 0, sizeof(*attr));
                attr->index = input;
                ret = rknn3_query(model->context, RKNN3_QUERY_INPUT_ATTR, attr, sizeof(*attr));
                if (ret != RKNN3_SUCCESS || attr->dtype != RKNN3_TENSOR_FLOAT16 ||
                    attr->layout != RKNN3_TENSOR_UNDEFINED) {
                    goto failed;
                }
            }
            memset(&model->shape_output_attrs[shape], 0, sizeof(rknn3_tensor_attr));
            model->shape_output_attrs[shape].index = 0;
            ret = rknn3_query(model->context, RKNN3_QUERY_OUTPUT_ATTR, &model->shape_output_attrs[shape],
                              sizeof(rknn3_tensor_attr));
            if (ret != RKNN3_SUCCESS || model->shape_output_attrs[shape].dtype != RKNN3_TENSOR_FLOAT16 ||
                model->shape_output_attrs[shape].layout != RKNN3_TENSOR_UNDEFINED) {
                goto failed;
            }
        }
    } else {
        model->is_dynamic = 0;
        model->shape_count = 1;
        model->shape_ids[0] = 0;
        for (uint32_t input = 0; input < model->input_count; ++input) {
            model->shape_input_attrs[0][input] = model->input_attrs[input];
        }
        model->shape_output_attrs[0] = model->output_attr;
    }
    printf("[%s] initialized: inputs=%u shapes=%u mode=%s\n", name, model->input_count, model->shape_count,
           model->is_dynamic ? "dynamic_bucket" : "static");
    return 0;

failed:
    printf("Error: failed to initialize %s: ret=%d\n", name, ret);
    release_npu_model(model);
    return -1;
}

/** Finds the smallest model shape that contains two sequence dimensions. */
static int find_npu_shape(const hotword_npu_model_t* model, int first_length, int second_length) {
    int best = -1;
    int best_score = INT32_MAX;
    for (uint32_t shape = 0; shape < model->shape_count; ++shape) {
        int first_bucket = tensor_time(&model->shape_input_attrs[shape][0]);
        int second_bucket = tensor_time(&model->shape_input_attrs[shape][1]);
        int score = first_bucket + second_bucket;
        if (first_bucket >= first_length && second_bucket >= second_length && score < best_score) {
            best = (int)shape;
            best_score = score;
        }
    }
    return best;
}

/** Writes a zero-padded rank-three feature tensor and synchronizes it. */
static int write_feature_input(hotword_npu_model_t* model, uint32_t index, const rknn3_tensor_attr* attr,
                               const float* source, size_t source_count, int valid_time, int source_dimension) {
    int destination_time = tensor_time(attr);
    int destination_dimension = tensor_last_dim(attr);
    size_t copy_count = 0;
    int ret = -1;
    if (!source || destination_dimension != source_dimension || valid_time > destination_time ||
        index >= model->input_count) {
        return -1;
    }
    copy_count = (size_t)valid_time * source_dimension;
    if (source_count < copy_count)
        return -1;
    memset(model->input_mem[index]->virt_addr, 0, model->input_mem[index]->size);
    paraformer_fp32_to_fp16(source, (uint16_t*)model->input_mem[index]->virt_addr, copy_count);
    ret = rknn3_mem_sync(model->context, model->input_mem[index], RKNN3_MEMORY_SYNC_TO_DEVICE);
    return ret == RKNN3_SUCCESS ? 0 : -1;
}

/** Writes a leading-one FP16 sequence mask and synchronizes it. */
static int write_mask_input(hotword_npu_model_t* model, uint32_t index, const rknn3_tensor_attr* attr,
                            int valid_length) {
    size_t element_count = tensor_numel(attr);
    uint16_t* destination = NULL;
    int ret = -1;
    if (index >= model->input_count || valid_length < 0 || (size_t)valid_length > element_count) {
        return -1;
    }
    memset(model->input_mem[index]->virt_addr, 0, model->input_mem[index]->size);
    destination = (uint16_t*)model->input_mem[index]->virt_addr;
    for (int position = 0; position < valid_length; ++position) {
        destination[position] = 0x3c00;
    }
    ret = rknn3_mem_sync(model->context, model->input_mem[index], RKNN3_MEMORY_SYNC_TO_DEVICE);
    return ret == RKNN3_SUCCESS ? 0 : -1;
}

/** Runs the five-input SeaCo decoder and returns FP32 logits. */
static int run_seaco_decoder(hotword_npu_model_t* model, const float* contextual_info, int hotword_count,
                             const float* predictor_embeds, size_t predictor_embed_count, const float* decoder_hidden,
                             size_t decoder_hidden_count, int token_num, float** logits, size_t* logit_count,
                             int* output_time, int* vocab_size) {
    int shape = find_npu_shape(model, hotword_count, token_num);
    rknn3_tensor inputs[5];
    rknn3_tensor output;
    rknn3_tensor_attr* attrs = NULL;
    rknn3_tensor_attr* output_attr = NULL;
    int ret = -1;
    int64_t begin_us = 0;

    if (shape < 0 || !logits || !logit_count || !output_time || !vocab_size) {
        printf("Error: no SeaCo bucket for hotwords=%d tokens=%d\n", hotword_count, token_num);
        return -1;
    }
    if (model->is_dynamic && rknn3_set_shape(model->context, model->shape_ids[shape]) != RKNN3_SUCCESS) {
        return -1;
    }
    attrs = model->shape_input_attrs[shape];
    output_attr = &model->shape_output_attrs[shape];
    if (write_feature_input(model, 0, &attrs[0], contextual_info, (size_t)hotword_count * HOTWORD_HIDDEN_SIZE,
                            hotword_count, HOTWORD_HIDDEN_SIZE) != 0 ||
        write_feature_input(model, 1, &attrs[1], predictor_embeds, predictor_embed_count, token_num,
                            HOTWORD_HIDDEN_SIZE) != 0 ||
        write_feature_input(model, 2, &attrs[2], decoder_hidden, decoder_hidden_count, token_num,
                            HOTWORD_HIDDEN_SIZE) != 0 ||
        write_mask_input(model, 3, &attrs[3], token_num) != 0 ||
        write_mask_input(model, 4, &attrs[4], hotword_count) != 0) {
        return -1;
    }

    memset(inputs, 0, sizeof(inputs));
    for (uint32_t index = 0; index < model->input_count; ++index) {
        inputs[index].mem = model->input_mem[index];
        inputs[index].attr = &attrs[index];
    }
    memset(&output, 0, sizeof(output));
    output.mem = model->output_mem;
    output.attr = output_attr;
    begin_us = getCurrentTimeUs();
    ret = rknn3_run(model->context, inputs, model->input_count, &output, 1);
    if (ret != RKNN3_SUCCESS)
        return -1;
    printf("[SeaCo] inference=%.2f ms hotwords=%d tokens=%d bucket=%d\n",
           (getCurrentTimeUs() - begin_us) / 1000.0,
           hotword_count, token_num, shape);
    ret = rknn3_mem_sync(model->context, model->output_mem, RKNN3_MEMORY_SYNC_FROM_DEVICE);
    if (ret != RKNN3_SUCCESS)
        return -1;

    *logit_count = tensor_numel(output_attr);
    *output_time = tensor_time(output_attr);
    *vocab_size = tensor_last_dim(output_attr);
    *logits = (float*)malloc(*logit_count * sizeof(float));
    if (!*logits)
        return -1;
    paraformer_fp16_to_fp32((const uint16_t*)model->output_mem->virt_addr, *logits, *logit_count);
    return 0;
}

/** Runs ASF attention scoring and selects top real hotwords plus NO_BIAS. */
static int run_asf_scorer(hotword_npu_model_t* model, const float* contextual_info, int hotword_count,
                          const float* decoder_hidden, size_t decoder_hidden_count, int token_num, int nfilter,
                          float** selected_context, int* selected_count) {
    int shape = find_npu_shape(model, hotword_count, token_num);
    rknn3_tensor inputs[4];
    rknn3_tensor output;
    rknn3_tensor_attr* attrs = NULL;
    rknn3_tensor_attr* output_attr = NULL;
    float* scores = NULL;
    unsigned char* selected = NULL;
    int real_count = hotword_count - 1;
    int keep_count = min_int(nfilter, real_count);
    int ret = -1;
    int64_t begin_us = 0;

    if (shape < 0 || keep_count <= 0 || !selected_context || !selected_count) {
        printf("Error: no ASF bucket for hotwords=%d tokens=%d\n", hotword_count, token_num);
        return -1;
    }
    if (model->is_dynamic && rknn3_set_shape(model->context, model->shape_ids[shape]) != RKNN3_SUCCESS) {
        return -1;
    }
    attrs = model->shape_input_attrs[shape];
    output_attr = &model->shape_output_attrs[shape];
    if (write_feature_input(model, 0, &attrs[0], contextual_info, (size_t)hotword_count * HOTWORD_HIDDEN_SIZE,
                            hotword_count, HOTWORD_HIDDEN_SIZE) != 0 ||
        write_feature_input(model, 1, &attrs[1], decoder_hidden, decoder_hidden_count, token_num,
                            HOTWORD_HIDDEN_SIZE) != 0 ||
        write_mask_input(model, 2, &attrs[2], token_num) != 0 ||
        write_mask_input(model, 3, &attrs[3], hotword_count) != 0) {
        return -1;
    }

    memset(inputs, 0, sizeof(inputs));
    for (uint32_t index = 0; index < model->input_count; ++index) {
        inputs[index].mem = model->input_mem[index];
        inputs[index].attr = &attrs[index];
    }
    memset(&output, 0, sizeof(output));
    output.mem = model->output_mem;
    output.attr = output_attr;
    begin_us = getCurrentTimeUs();
    if (rknn3_run(model->context, inputs, model->input_count, &output, 1) != RKNN3_SUCCESS ||
        rknn3_mem_sync(model->context, model->output_mem, RKNN3_MEMORY_SYNC_FROM_DEVICE) != RKNN3_SUCCESS) {
        return -1;
    }
    scores = (float*)malloc(tensor_numel(output_attr) * sizeof(float));
    selected = (unsigned char*)calloc((size_t)real_count, 1);
    *selected_context = (float*)malloc((size_t)(keep_count + 1) * HOTWORD_HIDDEN_SIZE * sizeof(float));
    if (!scores || !selected || !*selected_context || (int)tensor_numel(output_attr) < hotword_count) {
        goto cleanup;
    }
    paraformer_fp16_to_fp32((const uint16_t*)model->output_mem->virt_addr, scores, tensor_numel(output_attr));
    for (int output_index = 0; output_index < keep_count; ++output_index) {
        int best_index = -1;
        float best_score = -INFINITY;
        for (int candidate = 0; candidate < real_count; ++candidate) {
            if (!selected[candidate] && scores[candidate] > best_score) {
                best_score = scores[candidate];
                best_index = candidate;
            }
        }
        if (best_index < 0)
            goto cleanup;
        selected[best_index] = 1;
        memcpy(*selected_context + (size_t)output_index * HOTWORD_HIDDEN_SIZE,
               contextual_info + (size_t)best_index * HOTWORD_HIDDEN_SIZE, HOTWORD_HIDDEN_SIZE * sizeof(float));
    }
    memcpy(*selected_context + (size_t)keep_count * HOTWORD_HIDDEN_SIZE,
           contextual_info + (size_t)(hotword_count - 1) * HOTWORD_HIDDEN_SIZE, HOTWORD_HIDDEN_SIZE * sizeof(float));
    *selected_count = keep_count + 1;
    printf("[ASF] inference=%.2f ms selected=%d/%d real hotwords + NO_BIAS\n",
           (getCurrentTimeUs() - begin_us) / 1000.0,
           keep_count, real_count);
    ret = 0;

cleanup:
    if (ret != 0) {
        free(*selected_context);
        *selected_context = NULL;
    }
    free(selected);
    free(scores);
    return ret;
}

/** Converts one logit row to log probabilities in caller-provided storage. */
static void log_softmax_row(const float* logits, int size, float* output) {
    float maximum = logits[0];
    double sum = 0.0;
    for (int index = 1; index < size; ++index) {
        if (logits[index] > maximum)
            maximum = logits[index];
    }
    for (int index = 0; index < size; ++index) {
        sum += exp((double)logits[index] - maximum);
    }
    {
        float normalization = maximum + (float)log(sum);
        for (int index = 0; index < size; ++index) {
            output[index] = logits[index] - normalization;
        }
    }
}

/** Applies NO_BIAS-aware posterior fusion for every valid decoder timestep. */
static int fuse_seaco_logprobs(float* decoder_logits, size_t decoder_count, const float* seaco_logits,
                               size_t seaco_count, int token_len, int vocab_size, float weight) {
    float* main_logprob = NULL;
    float* seaco_logprob = NULL;
    int result = -1;
    if (!decoder_logits || !seaco_logits || token_len <= 0 || vocab_size <= 0 ||
        decoder_count < (size_t)token_len * vocab_size || seaco_count < (size_t)token_len * vocab_size) {
        return -1;
    }
    main_logprob = (float*)malloc((size_t)vocab_size * sizeof(float));
    seaco_logprob = (float*)malloc((size_t)vocab_size * sizeof(float));
    if (!main_logprob || !seaco_logprob)
        goto cleanup;

    for (int time = 0; time < token_len; ++time) {
        float* main_row = decoder_logits + (size_t)time * vocab_size;
        const float* seaco_row = seaco_logits + (size_t)time * vocab_size;
        int seaco_argmax = 0;
        log_softmax_row(main_row, vocab_size, main_logprob);
        log_softmax_row(seaco_row, vocab_size, seaco_logprob);
        for (int token = 1; token < vocab_size; ++token) {
            if (seaco_logprob[token] > seaco_logprob[seaco_argmax]) {
                seaco_argmax = token;
            }
        }
        if (seaco_argmax == HOTWORD_NO_BIAS_ID) {
            memcpy(main_row, main_logprob, (size_t)vocab_size * sizeof(float));
        } else {
            for (int token = 0; token < vocab_size; ++token) {
                main_row[token] = (1.0f - weight) * main_logprob[token] + weight * seaco_logprob[token];
            }
        }
    }
    result = 0;

cleanup:
    free(seaco_logprob);
    free(main_logprob);
    return result;
}

int paraformer_hotword_create(const char* seaco_model_path, const char* seaco_weight_path, const char* bias_weights_path,
                              const char* hotwords_path, const char* tokens_path, const char* seg_dict_path,
                              int max_hotwords, int max_hotword_len, float seaco_weight, const char* asf_model_path,
                              const char* asf_weight_path, int asf_nfilter, uint32_t core_mask,
                              paraformer_hotword_context_t** context) {
    paraformer_hotword_context_t* instance = NULL;
    int64_t begin_us = 0;

    if (!seaco_model_path || !seaco_weight_path || !bias_weights_path || !hotwords_path || !tokens_path ||
        !seg_dict_path || !context || max_hotwords < 1 || max_hotwords > 128 || max_hotword_len < 1 ||
        max_hotword_len > 256 || seaco_weight < 0.0f || seaco_weight > 1.0f ||
        ((asf_model_path || asf_weight_path || asf_nfilter > 0) &&
         (!asf_model_path || !asf_weight_path || asf_nfilter <= 0))) {
        printf("Error: invalid hotword configuration\n");
        return -1;
    }
    *context = NULL;
    instance = (paraformer_hotword_context_t*)calloc(1, sizeof(*instance));
    if (!instance)
        return -1;
    instance->max_hotword_len = max_hotword_len;
    instance->seaco_weight = seaco_weight;
    instance->asf_nfilter = asf_nfilter;

    if (load_hotword_ids(hotwords_path, tokens_path, seg_dict_path, max_hotwords, max_hotword_len,
                         &instance->hotword_ids, &instance->hotword_lengths, &instance->hotword_count) != 0 ||
        load_bias_weights(bias_weights_path, &instance->bias_weights) != 0) {
        goto failed;
    }
    instance->contextual_info = (float*)malloc((size_t)instance->hotword_count * HOTWORD_HIDDEN_SIZE * sizeof(float));
    if (!instance->contextual_info)
        goto failed;

    begin_us = getCurrentTimeUs();
    if (run_bias_encoder(&instance->bias_weights, instance->hotword_ids, instance->hotword_lengths,
                         instance->hotword_count, instance->max_hotword_len, instance->contextual_info) != 0) {
        goto failed;
    }
    printf("[Hotword] BiasEncoder=%.2f ms phrases=%d max_length=%d\n",
           (getCurrentTimeUs() - begin_us) / 1000.0,
           instance->hotword_count, instance->max_hotword_len);
    release_bias_weights(&instance->bias_weights);
    free(instance->hotword_ids);
    free(instance->hotword_lengths);
    instance->hotword_ids = NULL;
    instance->hotword_lengths = NULL;

    if (init_npu_model(seaco_model_path, seaco_weight_path, "SeaCo", 5, core_mask, &instance->seaco) != 0) {
        goto failed;
    }
    if (asf_model_path && init_npu_model(asf_model_path, asf_weight_path, "ASF", 4, core_mask, &instance->asf) != 0) {
        goto failed;
    }

    *context = instance;
    return 0;

failed:
    paraformer_hotword_release(instance);
    return -1;
}

int paraformer_hotword_apply(paraformer_hotword_context_t* context, const float* predictor_embeds,
                             size_t predictor_embed_count, const float* decoder_hidden, size_t decoder_hidden_count,
                             int token_num, float* decoder_logits, size_t decoder_logit_count, int token_len,
                             int vocab_size) {
    const float* selected_context = NULL;
    float* filtered_context = NULL;
    float* seaco_logits = NULL;
    size_t seaco_logit_count = 0;
    int selected_count = 0;
    int seaco_time = 0;
    int seaco_vocab = 0;
    int result = -1;

    if (!context || !predictor_embeds || !decoder_hidden || !decoder_logits || token_num <= 0 || token_len <= 0 ||
        vocab_size <= 0) {
        return -1;
    }
    selected_context = context->contextual_info;
    selected_count = context->hotword_count;
    if (context->asf.context && context->asf_nfilter > 0 && context->hotword_count > context->asf_nfilter + 1) {
        if (run_asf_scorer(&context->asf, context->contextual_info, context->hotword_count, decoder_hidden,
                           decoder_hidden_count, token_num, context->asf_nfilter, &filtered_context,
                           &selected_count) != 0) {
            goto cleanup;
        }
        selected_context = filtered_context;
    }

    if (run_seaco_decoder(&context->seaco, selected_context, selected_count, predictor_embeds, predictor_embed_count,
                          decoder_hidden, decoder_hidden_count, token_num, &seaco_logits, &seaco_logit_count,
                          &seaco_time, &seaco_vocab) != 0 ||
        seaco_vocab != vocab_size || seaco_time < token_len ||
        fuse_seaco_logprobs(decoder_logits, decoder_logit_count, seaco_logits, seaco_logit_count, token_len, vocab_size,
                            context->seaco_weight) != 0) {
        goto cleanup;
    }
    printf("[Hotword] posterior fusion applied: weight=%.2f phrases=%d\n", context->seaco_weight, selected_count);
    result = 0;

cleanup:
    free(seaco_logits);
    free(filtered_context);
    return result;
}

int paraformer_hotword_get_npu_contexts(const paraformer_hotword_context_t* context,
                                        rknn3_context* model_contexts, uint32_t capacity,
                                        uint32_t* model_count) {
    uint32_t count = 0;

    if (!context || !model_contexts || !model_count) {
        return -1;
    }
    if (context->seaco.context) {
        if (count >= capacity) {
            return -1;
        }
        model_contexts[count++] = context->seaco.context;
    }
    if (context->asf.context) {
        if (count >= capacity) {
            return -1;
        }
        model_contexts[count++] = context->asf.context;
    }
    *model_count = count;
    return 0;
}

void paraformer_hotword_release(paraformer_hotword_context_t* context) {
    if (!context)
        return;
    release_npu_model(&context->asf);
    release_npu_model(&context->seaco);
    release_bias_weights(&context->bias_weights);
    free(context->contextual_info);
    free(context->hotword_lengths);
    free(context->hotword_ids);
    free(context);
}
