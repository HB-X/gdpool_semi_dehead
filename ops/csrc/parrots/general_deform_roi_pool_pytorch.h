// Copyright (c) OpenMMLab. All rights reserved
#ifndef GENERAL_DEFORM_ROI_POOL_PYTORCH_H
#define GENERAL_DEFORM_ROI_POOL_PYTORCH_H
#include <torch/extension.h>
using namespace at;

void general_deform_roi_pool_forward_cuda(Tensor input, Tensor rois, Tensor rescale, Tensor offset,
                                          Tensor output, int pooled_height,
                                          int pooled_width, float spatial_scale,
                                          int sampling_ratio, float gamma, float sigma, float beta);

void general_deform_roi_pool_backward_cuda(Tensor grad_output, Tensor input,
                                           Tensor rois, Tensor rescale, Tensor offset,
                                           Tensor grad_input, Tensor grad_rescale, Tensor grad_offset,
                                           int pooled_height, int pooled_width,
                                           float spatial_scale, int sampling_ratio,
                                           float gamma, float sigma, float beta);
#endif  // GENERAL_DEFORM_ROI_POOL_PYTORCH_H
