// Copyright (c) 2020 SenseTime. All Rights Reserved.
// Licensed under the Apache License, Version 2.0 [see LICENSE for details].
// Modified: declarations for the standalone Metal backend.
#pragma once

#include <ATen/ATen.h>
#include <vector>

at::Tensor ms_deform_attn_mps_forward(const at::Tensor &value,
                                      const at::Tensor &shapes,
                                      const at::Tensor &starts,
                                      const at::Tensor &locations,
                                      const at::Tensor &weights, int step);

std::vector<at::Tensor> ms_deform_attn_mps_backward(
    const at::Tensor &value, const at::Tensor &shapes, const at::Tensor &starts,
    const at::Tensor &locations, const at::Tensor &weights,
    const at::Tensor &grad, int step);
