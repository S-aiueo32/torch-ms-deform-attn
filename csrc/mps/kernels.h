// Copyright (c) 2020 SenseTime. All Rights Reserved.
// Licensed under the Apache License, Version 2.0 [see LICENSE for details].
// Modified: Metal sampling/reduction and first-order gradient kernels.
#pragma once

// Embedded source keeps installed wheels independent of source-tree paths and
// the offline Metal compiler. Arithmetic deliberately matches the CPU kernel.
constexpr const char *msda_metal_source = R"METAL(
#include <metal_stdlib>
using namespace metal;

struct Dimensions { ulong N, S, M, D, Q, L, P; };

// CAS on integer bits works on all supported Apple GPUs, including devices
// without native floating-point atomic add. Comparing bits also handles NaNs.
inline void add_float(device atomic_uint *address, float increment) {
    uint old = atomic_load_explicit(address, memory_order_relaxed);
    while (!atomic_compare_exchange_weak_explicit(
        address, &old, as_type<uint>(as_type<float>(old) + increment),
        memory_order_relaxed, memory_order_relaxed)) {}
}

struct Sample {
    long x0, y0;
    float dx, dy;
    bool active;
};

inline Sample sample_at(float2 location, long h, long w) {
    float x = location.x * float(w) - 0.5f;
    float y = location.y * float(h) - 0.5f;
    Sample s;
    s.active = x > -1 && x < float(w) && y > -1 && y < float(h);
    if (s.active) {
        s.x0 = long(floor(x)); s.y0 = long(floor(y));
        s.dx = x - float(s.x0); s.dy = y - float(s.y0);
    }
    return s;
}

inline float pixel(device const float *value, ulong base, long x, long y,
                   long h, long w, ulong stride) {
    return x >= 0 && x < w && y >= 0 && y < h
        ? value[base + ulong(y * w + x) * stride] : 0.0f;
}

kernel void msda_forward(
    device const float *value [[buffer(0)]],
    device const long *shapes [[buffer(1)]],
    device const long *starts [[buffer(2)]],
    device const float2 *locations [[buffer(3)]],
    device const float *weights [[buffer(4)]],
    device float *output [[buffer(5)]],
    constant Dimensions &d [[buffer(6)]],
    constant ulong &begin [[buffer(7)]],
    uint index [[thread_position_in_grid]]) {
    ulong oi = begin + ulong(index);
    if (oi >= d.N * d.Q * d.M * d.D) return;
    ulong c = oi % d.D, m = (oi / d.D) % d.M;
    ulong q = (oi / (d.D * d.M)) % d.Q;
    ulong n = oi / (d.D * d.M * d.Q);
    float sum = 0;
    for (ulong l = 0; l < d.L; ++l) {
        long h = shapes[2*l], w = shapes[2*l+1];
        ulong base = (n * d.S + ulong(starts[l])) * d.M * d.D + m * d.D + c;
        for (ulong p = 0; p < d.P; ++p) {
            ulong i = (((n * d.Q + q) * d.M + m) * d.L + l) * d.P + p;
            Sample s = sample_at(locations[i], h, w);
            if (!s.active) continue;
            float lx = 1 - s.dx, ly = 1 - s.dy;
            float v00 = pixel(value, base, s.x0, s.y0, h, w, d.M*d.D);
            float v01 = pixel(value, base, s.x0+1, s.y0, h, w, d.M*d.D);
            float v10 = pixel(value, base, s.x0, s.y0+1, h, w, d.M*d.D);
            float v11 = pixel(value, base, s.x0+1, s.y0+1, h, w, d.M*d.D);
            sum += (v00*(lx*ly) + v01*(s.dx*ly) + v10*(lx*s.dy)
                    + v11*(s.dx*s.dy)) * weights[i];
        }
    }
    output[oi] = sum;
}

inline void scatter(device atomic_uint *grad, ulong base, long x, long y,
                    long h, long w, ulong stride, float amount) {
    if (x >= 0 && x < w && y >= 0 && y < h)
        add_float(grad + base + ulong(y*w+x)*stride, amount);
}

// One SIMD group owns one sampling point. Channels stride across its lanes;
// only the value gradient needs atomics, even for overlapping feature levels.
kernel void msda_backward(
    device const float *value [[buffer(0)]],
    device const long *shapes [[buffer(1)]],
    device const long *starts [[buffer(2)]],
    device const float2 *locations [[buffer(3)]],
    device const float *weights [[buffer(4)]],
    device const float *grad [[buffer(5)]],
    device atomic_uint *grad_value [[buffer(6)]],
    device float2 *grad_locations [[buffer(7)]],
    device float *grad_weights [[buffer(8)]],
    constant Dimensions &d [[buffer(9)]],
    constant ulong &begin [[buffer(10)]],
    uint index [[thread_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort width [[threads_per_simdgroup]]) {
    ulong i = begin + ulong(index) / width;
    if (i >= d.N * d.Q * d.M * d.L * d.P) return;
    ulong l = (i / d.P) % d.L;
    ulong m = (i / (d.P*d.L)) % d.M;
    ulong q = (i / (d.P*d.L*d.M)) % d.Q;
    ulong n = i / (d.P*d.L*d.M*d.Q);
    long h = shapes[2*l], w = shapes[2*l+1];
    Sample s = sample_at(locations[i], h, w);
    float ga = 0, gx = 0, gy = 0;
    if (s.active) {
        float lx = 1 - s.dx, ly = 1 - s.dy;
        for (ulong c = lane; c < d.D; c += width) {
            ulong base = (n*d.S + ulong(starts[l]))*d.M*d.D + m*d.D+c;
            float v00 = pixel(value, base, s.x0, s.y0, h, w, d.M*d.D);
            float v01 = pixel(value, base, s.x0+1, s.y0, h, w, d.M*d.D);
            float v10 = pixel(value, base, s.x0, s.y0+1, h, w, d.M*d.D);
            float v11 = pixel(value, base, s.x0+1, s.y0+1, h, w, d.M*d.D);
            float g = grad[((n*d.Q+q)*d.M+m)*d.D+c];
            float gs = g * weights[i];
            ga += g * (v00*(lx*ly) + v01*(s.dx*ly) + v10*(lx*s.dy) + v11*(s.dx*s.dy));
            gx += gs * (ly*(v01-v00) + s.dy*(v11-v10));
            gy += gs * (lx*(v10-v00) + s.dx*(v11-v01));
            scatter(grad_value, base, s.x0, s.y0, h, w, d.M*d.D, gs*(lx*ly));
            scatter(grad_value, base, s.x0+1, s.y0, h, w, d.M*d.D, gs*(s.dx*ly));
            scatter(grad_value, base, s.x0, s.y0+1, h, w, d.M*d.D, gs*(lx*s.dy));
            scatter(grad_value, base, s.x0+1, s.y0+1, h, w, d.M*d.D, gs*(s.dx*s.dy));
        }
    }
    ga = simd_sum(ga); gx = simd_sum(gx); gy = simd_sum(gy);
    if (lane == 0) {
        grad_weights[i] = ga;
        grad_locations[i] = float2(gx * float(w), gy * float(h));
    }
}
)METAL";
