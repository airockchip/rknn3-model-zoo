// Copyright (c) 2025 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef SPEEDUP_H
#define SPEEDUP_H

#include <stdint.h>
#include <rknn3_api.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SPEEDUP_VERSION_MAJOR 1
#define SPEEDUP_VERSION_MINOR 0
#define SPEEDUP_VERSION_PATCH 1

typedef struct SpeedUPContext* SpeedUPHandle;

typedef struct {
    float tau;       // Automatic mode threshold.
    float min_ratio; // Lower bound of the output ratio.
    float max_ratio; // Upper bound of the output ratio.
    int stride;      // Feature stride.
} SpeedUPConfig;

typedef struct {
    int32_t patch_height;   // Runtime grid height before processing.
    int32_t patch_width;    // Runtime grid width before processing.
    int32_t patch_temporal; // Runtime temporal grid size. Use 1 for static input.
} SpeedUPMropeGrid;

typedef struct {
    void* image_embed;          // Input embeddings. The buffer is updated in place after processing.
    int n_images;               // Number of media inputs in the current request.
    int items_per_input;       // Number of embedding items for each media input.
    int embedding_dim;          // Embedding dimension for each item.
    int32_t model_width;        // Model input width, used for runtime shape setup.
    int32_t model_height;       // Model input height, used for runtime shape setup.
    float speedup_ratio;  // 0 disables acceleration, (0, 1) uses manual ratio, 1 enables automatic ratio.
    rknn3_session* llm_sess;    // LLM session handle. Reserved for interfaces that need session context.
} SpeedUPMultimodalParam;

/**
 * @brief Creates a SpeedUP instance.
 *
 * The returned handle stores per-request runtime state and callback state.
 * Each inference thread should use an independent handle.
 *
 * @param[in] config Pointer to SpeedUP configuration. If NULL, default configuration is used.
 * @return SpeedUPHandle Return a valid handle on success, or NULL on failure.
 *
 * @note The handle must be released by speedup_destroy() or speedup_cleanup().
 */
SpeedUPHandle speedup_create(const SpeedUPConfig* config);

/**
 * @brief Destroys a SpeedUP instance.
 *
 * @param[in] handle SpeedUP handle created by speedup_create().
 *
 * @note This function releases only resources owned by SpeedUP. If an MRoPE callback
 *       was attached to an RKNN3 session, call speedup_cleanup() when releasing the
 *       associated model/session.
 */
void speedup_destroy(SpeedUPHandle handle);

/**
 * @brief Gets the selection map of the latest single-input result.
 *
 * The returned array length equals the original item count.
 * Value 1 means the item is selected, and 0 means the item is skipped.
 *
 * @param[in] handle SpeedUP handle.
 * @return const int* Pointer to the selection map, or NULL if no valid result is available.
 *
 * @note The returned pointer is owned by SpeedUP and remains valid until the next
 *       processing call, reset call, or handle destruction.
 */
const int* speedup_get_keep_mask(SpeedUPHandle handle);

/**
 * @brief Gets the output item count from the latest result.
 *
 * @param[in] handle SpeedUP handle.
 * @return int Number of output items. Returns 0 if no valid result is available.
 */
int speedup_get_output_count(SpeedUPHandle handle);

/**
 * @brief Gets the input item count before processing.
 *
 * @param[in] handle SpeedUP handle.
 * @return int Number of original items. Returns 0 if no valid result is available.
 */
int speedup_get_input_count(SpeedUPHandle handle);

/**
 * @brief Resets runtime state for a new request.
 *
 * @param[in] handle SpeedUP handle.
 *
 * @note This function clears selection state and counters. It does not detach RKNN3 callbacks.
 */
void speedup_reset(SpeedUPHandle handle);

/**
 * @brief Gets the SpeedUP library version string.
 *
 * @return const char* Version string in "major.minor.patch" format.
 */
const char* speedup_get_version(void);

/**
 * @brief Cleans up SpeedUP resources associated with an RKNN3 LLM session.
 *
 * If SpeedUP has attached an MRoPE input callback, this function restores the
 * original callback before destroying the handle.
 *
 * @param[in] handle SpeedUP handle.
 * @param[in] llm_sess RKNN3 LLM session associated with the attached callback. May be NULL
 *                     if no callback was attached.
 *
 * @note Call this function only when the associated model/session is being released.
 *       Do not call it after each inference request.
 */
void speedup_cleanup(SpeedUPHandle handle, rknn3_session* llm_sess);

/**
 * @brief Processes embeddings with a total item count.
 *
 * This helper accepts the total item count for all media inputs and internally
 * dispatches to speedup_process_multi().
 *
 * @param[in] handle SpeedUP handle.
 * @param[in,out] img_embeds Pointer to embeddings. The buffer is updated in place on success.
 * @param[in] n_images Number of media inputs in the request. Must be greater than 0.
 * @param[in] num_items Total number of embedding items. Must be divisible by n_images.
 * @param[in] embed_dim Embedding dimension for each item.
 * @param[in] speedup_ratio Acceleration ratio control. 0 disables acceleration, (0, 1) uses manual
 *                                output ratio, and 1 enables automatic ratio.
 * @param[in] llm_sess RKNN3 LLM session handle. Reserved for integrations requiring session context.
 * @param[out] out_output_count Pointer to receive the total output item count. May be NULL.
 * @return int Return status code:
 *         - 0: Success
 *         - <0: Error code
 */
int speedup_process_n(SpeedUPHandle handle,
                           void* img_embeds,
                           int n_images,
                           int num_items,
                           int embed_dim,
                           float speedup_ratio,
                           rknn3_session* llm_sess,
                           int* out_output_count);

/**
 * @brief Processes embeddings with a fixed item count per media input.
 *
 * Manual mode uses the same output count for all media inputs. Automatic mode currently
 * handles single-input requests; multi-input automatic requests fall back to baseline mode.
 * On success, the embedding buffer is updated in place in contiguous media order.
 *
 * @param[in] handle SpeedUP handle.
 * @param[in,out] img_embeds Pointer to embeddings in contiguous media order.
 * @param[in] n_images Number of media inputs in the request. Must be greater than 0.
 * @param[in] items_per_input Number of embedding items for each media input.
 * @param[in] embed_dim Embedding dimension for each item.
 * @param[in] speedup_ratio Acceleration ratio control. 0 disables acceleration, (0, 1) uses manual
 *                                output ratio, and 1 enables automatic ratio.
 * @param[in] llm_sess RKNN3 LLM session handle. Reserved for integrations requiring session context.
 * @param[out] out_output_count_per_input Pointer to receive the output item count per media input.
 *                                         May be NULL.
 * @return int Return status code:
 *         - 0: Success
 *         - <0: Error code
 */
int speedup_process_multi(SpeedUPHandle handle,
                               void* img_embeds,
                               int n_images,
                               int items_per_input,
                               int embed_dim,
                               float speedup_ratio,
                               rknn3_session* llm_sess,
                               int* out_output_count_per_input);

/**
 * @brief Prepares multimodal embeddings for LLM inference.
 *
 * This function configures runtime shape metadata and then processes
 * param->image_embed. Model-specific auxiliary inputs should be updated by the caller
 * using the selection state returned by SpeedUP.
 *
 * @param[in] handle SpeedUP handle.
 * @param[in] param Pointer to multimodal preparation parameters. Must not be NULL.
 * @param[out] out_output_count_per_input Pointer to receive the output item count per media input.
 *                                         May be NULL.
 * @return int Return status code:
 *         - 0: Success
 *         - <0: Error code
 */
int speedup_prepare_multimodal(SpeedUPHandle handle,
                                    const SpeedUPMultimodalParam* param,
                                    int* out_output_count_per_input);

/**
 * @brief Prepares an RKNN3 multimodal tensor for LLM inference.
 *
 * This function derives embedding_dim from embed_shape, processes image_embed, and updates
 * the RKNN3 tensor item count to the output count per media input. Model-specific
 * auxiliary inputs should be updated by the caller after this function returns successfully.
 *
 * @param[in] handle SpeedUP handle.
 * @param[in,out] image_embed Pointer to embeddings. The buffer is updated in place on success.
 * @param[in] embed_shape Pointer to the embedding shape array.
 * @param[in] embed_ndims Number of dimensions in embed_shape.
 * @param[in,out] tensor Pointer to RKNN3 multimodal tensor. The related item count is
 *                       updated on success.
 * @param[in] model_width Model input width.
 * @param[in] model_height Model input height.
 * @param[in] speedup_ratio Acceleration ratio control. 0 disables acceleration, (0, 1) uses manual
 *                                output ratio, and 1 enables automatic ratio.
 * @param[in] llm_sess RKNN3 LLM session handle. Reserved for integrations requiring session context.
 * @param[out] out_output_count_per_input Pointer to receive the output item count per media input.
 *                                         May be NULL.
 * @return int Return status code:
 *         - 0: Success
 *         - <0: Error code
 */
int speedup_prepare_rknn_multimodal_tensor(SpeedUPHandle handle,
                                                void* image_embed,
                                                const uint32_t* embed_shape,
                                                uint32_t embed_ndims,
                                                rknn3_llm_multimodal_tensor* tensor,
                                                int32_t model_width,
                                                int32_t model_height,
                                                float speedup_ratio,
                                                rknn3_session* llm_sess,
                                                int* out_output_count_per_input);

/**
 * @brief Gets the selection map for one media input in the latest multi-input result.
 *
 * The returned array length equals items_per_input. Value 1 means the item is selected,
 * and 0 means the item is skipped.
 *
 * @param[in] handle SpeedUP handle.
 * @param[in] image_index Media input index in the current request.
 * @return const int* Pointer to the selection map, or NULL for invalid input or unavailable result.
 */
const int* speedup_get_keep_mask_for_image(SpeedUPHandle handle, int image_index);

/**
 * @brief Gets the media input count recorded in the latest result.
 *
 * @param[in] handle SpeedUP handle.
 * @return int Number of media inputs. Returns 0 if no valid result is available.
 */
int speedup_get_num_images(SpeedUPHandle handle);

/**
 * @brief Gets the output item count per media input from the latest result.
 *
 * @param[in] handle SpeedUP handle.
 * @return int Output item count per media input. Returns 0 if no valid result is available.
 */
int speedup_get_output_count_per_input(SpeedUPHandle handle);

// ===== RKNN3 runtime input callback support =====
// Some RKNN3 multimodal LLM models provide auxiliary runtime inputs through input_callback.
// SpeedUP maintains the required runtime data and fills each callback bucket according
// to LLMInputCallbackParam::pos and item count.

/**
 * @brief Sets runtime grid metadata for each media input.
 *
 * For each grid, patch_height * patch_width * patch_temporal must match the
 * corresponding item count before processing.
 *
 * @param[in] handle SpeedUP handle.
 * @param[in] grids Pointer to an array of runtime grid descriptions.
 * @param[in] n_images Number of entries in grids.
 * @return int Return status code:
 *         - 0: Success
 *         - <0: Error code
 */
int speedup_set_mrope_grids(SpeedUPHandle handle,
                                 const SpeedUPMropeGrid* grids,
                                 int n_images);

/**
 * @brief Sets runtime grid metadata for one media input.
 *
 * @param[in] handle SpeedUP handle.
 * @param[in] image_index Media input index in the current request.
 * @param[in] patch_height Runtime grid height before processing.
 * @param[in] patch_width Runtime grid width before processing.
 * @param[in] patch_temporal Runtime temporal grid size. Use 1 for static input.
 * @return int Return status code:
 *         - 0: Success
 *         - <0: Error code
 */
int speedup_set_mrope_grid(SpeedUPHandle handle,
                                int image_index,
                                int32_t patch_height,
                                int32_t patch_width,
                                int32_t patch_temporal);

/**
 * @brief Infers and sets runtime grid metadata from model input size.
 *
 * @param[in] handle SpeedUP handle.
 * @param[in] image_index Media input index in the current request.
 * @param[in] items_per_input Number of embedding items for the media input.
 * @param[in] model_width Model input width.
 * @param[in] model_height Model input height.
 * @return int Return status code:
 *         - 0: Success
 *         - <0: Error code
 */
int speedup_set_mrope_grid_from_image_size(SpeedUPHandle handle,
                                                int image_index,
                                                int items_per_input,
                                                int32_t model_width,
                                                int32_t model_height);

/**
 * @brief Clears callback runtime state for the current request.
 *
 * @param[in] handle SpeedUP handle.
 *
 * @note Configured grids are preserved.
 */
void speedup_reset_mrope(SpeedUPHandle handle);

/**
 * @brief Fills one auxiliary input bucket in a custom RKNN3 input_callback implementation.
 *
 * @param[in] handle SpeedUP handle.
 * @param[out] dst Destination buffer for runtime values.
 * @param[in] dst_num_items Capacity of dst measured in items. dst must have space for
 *                           dst_num_items * 3 int32_t values.
 * @param[in] param Pointer to RKNN3 input callback parameters.
 * @return int Return status code:
 *         - 0: Success
 *         - <0: Error code
 */
int speedup_fill_mrope_input(SpeedUPHandle handle,
                                  int32_t* dst,
                                  int32_t dst_num_items,
                                  const LLMInputCallbackParam* param);

/**
 * @brief RKNN3 input callback used to fill auxiliary runtime tensors.
 *
 * Pass SpeedUPHandle as userdata. The function fills the matching runtime tensors and leaves other tensors unchanged.
 *
 * @param[in] userdata SpeedUPHandle passed as callback userdata.
 * @param[in,out] input_tensors Pointer to RKNN3 input tensors supplied by the runtime.
 * @param[in] n_input_tensors Number of input tensors.
 * @param[in] param RKNN3 input callback parameters.
 * @return int Return status code:
 *         - 0: Success
 *         - <0: Error code
 */
int speedup_mrope_input_callback(void* userdata,
                                      rknn3_tensor* input_tensors,
                                      uint32_t n_input_tensors,
                                      LLMInputCallbackParam param);

/**
 * @brief Attaches SpeedUP runtime input callback to an RKNN3 LLM session.
 *
 * The public wrapper passes sizeof(rknn3_tensor_attr) from the application build to
 * libSpeedUP. This keeps the prebuilt library independent of later, compatible
 * rknn3_tensor_attr size extensions.
 *
 * @param[in] handle SpeedUP handle.
 * @param[in] rknn_ctx RKNN3 context handle used by the LLM model.
 * @param[in] llm_sess RKNN3 LLM session handle.
 * @param[in] base_callback Pointer to the existing callback configuration. May be NULL.
 * @return int Return status code:
 *         - 0: Success
 *         - <0: Error code
 *
 * @note Attach once during model/session initialization. Do not repeatedly attach the callback
 *       for every inference request.
 */
int speedup_attach_mrope_callback_with_attr_size(SpeedUPHandle handle,
                                                 rknn3_context rknn_ctx,
                                                 rknn3_session* llm_sess,
                                                 const RKLLMCallback* base_callback,
                                                 uint64_t tensor_attr_size);

static inline int speedup_attach_mrope_callback(SpeedUPHandle handle,
                                                rknn3_context rknn_ctx,
                                                rknn3_session* llm_sess,
                                                const RKLLMCallback* base_callback) {
    return speedup_attach_mrope_callback_with_attr_size(
        handle, rknn_ctx, llm_sess, base_callback, sizeof(rknn3_tensor_attr));
}

/**
 * @brief Restores the callback supplied to speedup_attach_mrope_callback().
 *
 * @param[in] handle SpeedUP handle.
 * @param[in] llm_sess RKNN3 LLM session handle.
 * @return int Return status code:
 *         - 0: Success
 *         - <0: Error code
 *
 * @note Detach only when the associated model/session is being released.
 */
int speedup_detach_mrope_callback(SpeedUPHandle handle,
                                       rknn3_session* llm_sess);

#ifdef __cplusplus
}
#endif

#endif // SPEEDUP_H
