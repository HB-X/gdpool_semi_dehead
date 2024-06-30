// Copyright (c) OpenMMLab. All rights reserved
#include "pytorch_cpp_helper.hpp"

#ifdef MMCV_WITH_CUDA
void GeneralDeformRoIPoolForwardCUDAKernelLauncher(Tensor input, Tensor rois, Tensor rescale,
                                                   Tensor offset, Tensor output,
                                                   int pooled_height, int pooled_width,
                                                   float spatial_scale,
                                                   int sampling_ratio, float gamma, float sigma, float beta);

void GeneralDeformRoIPoolBackwardCUDAKernelLauncher(
    Tensor grad_output, Tensor input, Tensor rois, Tensor rescale, Tensor offset,
    Tensor grad_input, Tensor grad_rescale, Tensor grad_offset, int pooled_height, int pooled_width,
    float spatial_scale, int sampling_ratio, float gamma, float sigma, float beta);

void general_deform_roi_pool_forward_cuda(Tensor input, Tensor rois, Tensor rescale, Tensor offset,
                                          Tensor output, int pooled_height,
                                          int pooled_width, float spatial_scale,
                                          int sampling_ratio, float gamma, float sigma, float beta) {
  GeneralDeformRoIPoolForwardCUDAKernelLauncher(input, rois, rescale, offset, output,
                                                pooled_height, pooled_width,
                                                spatial_scale, sampling_ratio, gamma, sigma, beta);
}

void general_deform_roi_pool_backward_cuda(Tensor grad_output, Tensor input,
                                           Tensor rois, Tensor rescale, Tensor offset,
                                           Tensor grad_input, Tensor grad_rescale, Tensor grad_offset,
                                           int pooled_height, int pooled_width,
                                           float spatial_scale, int sampling_ratio, float gamma, float sigma, float beta) {
  GeneralDeformRoIPoolBackwardCUDAKernelLauncher(
      grad_output, input, rois, rescale, offset, grad_input, grad_rescale, grad_offset, pooled_height,
      pooled_width, spatial_scale, sampling_ratio, gamma, sigma, beta);
}
#endif

void general_deform_roi_pool_forward(Tensor input, Tensor rois, Tensor rescale, Tensor offset,
                                     Tensor output, int pooled_height, int pooled_width,
                                     float spatial_scale, int sampling_ratio, float gamma, float sigma, float beta) {
  if (input.device().is_cuda()) {
#ifdef MMCV_WITH_CUDA
    CHECK_CUDA_INPUT(input);
    CHECK_CUDA_INPUT(rois);
    CHECK_CUDA_INPUT(rescale);
    CHECK_CUDA_INPUT(offset);
    CHECK_CUDA_INPUT(output);

    general_deform_roi_pool_forward_cuda(input, rois, rescale, offset, output, pooled_height,
                                         pooled_width, spatial_scale, sampling_ratio, gamma, sigma, beta);
#else
    AT_ERROR("GeneralDeformRoIPool is not compiled with GPU support");
#endif
  } else {
    AT_ERROR("GeneralDeformRoIPool is not implemented on CPU");
  }
}

void general_deform_roi_pool_backward(Tensor grad_output, Tensor input, Tensor rois,
                                      Tensor rescale, Tensor offset, Tensor grad_input,
                                      Tensor grad_rescale, Tensor grad_offset, int pooled_height,
                                      int pooled_width, float spatial_scale,
                                      int sampling_ratio, float gamma, float sigma, float beta) {
  if (grad_output.device().is_cuda()) {
#ifdef MMCV_WITH_CUDA
    CHECK_CUDA_INPUT(grad_output);
    CHECK_CUDA_INPUT(input);
    CHECK_CUDA_INPUT(rois);
    CHECK_CUDA_INPUT(rescale);
    CHECK_CUDA_INPUT(offset);
    CHECK_CUDA_INPUT(grad_input);
    CHECK_CUDA_INPUT(grad_rescale);
    CHECK_CUDA_INPUT(grad_offset);

    general_deform_roi_pool_backward_cuda(grad_output, input, rois, rescale, offset, grad_input,
                                          grad_rescale, grad_offset, pooled_height, pooled_width,
                                          spatial_scale, sampling_ratio, gamma, sigma, beta);
#else
    AT_ERROR("GeneralDeformRoIPool is not compiled with GPU support");
#endif
  } else {
    AT_ERROR("GeneralDeformRoIPool is not implemented on CPU");
  }
}
