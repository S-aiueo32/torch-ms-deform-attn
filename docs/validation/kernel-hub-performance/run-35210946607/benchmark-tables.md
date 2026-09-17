# Kernel Hub benchmark results

Median of repetition medians; wall time in milliseconds.
FP32-compute rows include input/output casts. Native rows use different arithmetic.

## fp32-compute / forward

| Case / dtype | torch-ms-deform-attn | kernel-hub-adapter | hf-native | mmcv-source | msda-triton-rziga | pytorch-reference | upstream-native-control |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| decoder / bfloat16 | 0.0920 | 0.0952 | 0.0985 | — | — | — | 0.0840 |
| decoder / float16 | 0.0957 | 0.0929 | 0.1003 | — | — | — | 0.0829 |
| decoder / float32 | 0.0420 | 0.0419 | 0.0392 | — | — | — | 0.0419 |
| encoder / bfloat16 | 0.8254 | 0.8297 | 0.7583 | — | — | — | 0.8296 |
| encoder / float16 | 0.8323 | 0.8319 | 0.7650 | — | — | — | 0.8302 |
| encoder / float32 | 0.6596 | 0.6681 | 0.5973 | — | — | — | 0.6654 |
| small / bfloat16 | 0.0921 | 0.0950 | 0.0930 | — | — | — | 0.0850 |
| small / float16 | 0.0914 | 0.0943 | 0.0984 | — | — | — | 0.0800 |
| small / float32 | 0.0341 | 0.0333 | 0.0420 | — | — | — | 0.0254 |

## fp32-compute / backward

| Case / dtype | torch-ms-deform-attn | kernel-hub-adapter | hf-native | mmcv-source | msda-triton-rziga | pytorch-reference | upstream-native-control |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| decoder / bfloat16 | 0.2482 | 0.2528 | 0.2884 | — | — | — | 0.2250 |
| decoder / float16 | 0.2501 | 0.2419 | 0.2872 | — | — | — | 0.3338 |
| decoder / float32 | 0.2039 | 0.2098 | 0.1918 | — | — | — | 0.1911 |
| encoder / bfloat16 | 2.3318 | 2.3411 | 2.2491 | — | — | — | 2.3320 |
| encoder / float16 | 2.3527 | 2.3530 | 2.2670 | — | — | — | 2.3482 |
| encoder / float32 | 2.2175 | 2.2180 | 2.1304 | — | — | — | 2.2203 |
| small / bfloat16 | 0.2464 | 0.2433 | 0.2862 | — | — | — | 0.3306 |
| small / float16 | 0.2497 | 0.3074 | 0.2733 | — | — | — | 0.2756 |
| small / float32 | 0.2365 | 0.2239 | 0.1969 | — | — | — | 0.2185 |

## fp32-compute / forward_backward

| Case / dtype | torch-ms-deform-attn | kernel-hub-adapter | hf-native | mmcv-source | msda-triton-rziga | pytorch-reference | upstream-native-control |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| decoder / bfloat16 | 0.9334 | 0.9207 | 0.8072 | — | — | — | 0.6584 |
| decoder / float16 | 0.8549 | 0.8862 | 0.7461 | — | — | — | 0.6926 |
| decoder / float32 | 0.5392 | 0.5683 | 0.4225 | — | — | — | 0.3531 |
| encoder / bfloat16 | 3.1507 | 3.1599 | 2.9912 | — | — | — | 3.1539 |
| encoder / float16 | 3.1690 | 3.1727 | 3.0084 | — | — | — | 3.1581 |
| encoder / float32 | 3.0319 | 3.0470 | 2.9176 | — | — | — | 3.0366 |
| small / bfloat16 | 0.8891 | 0.8524 | 0.8394 | — | — | — | 0.7469 |
| small / float16 | 0.9448 | 0.9108 | 0.7530 | — | — | — | 0.7333 |
| small / float32 | 0.5216 | 0.5419 | 0.4118 | — | — | — | 0.3849 |

## native / forward

| Case / dtype | torch-ms-deform-attn | kernel-hub-adapter | hf-native | mmcv-source | msda-triton-rziga | pytorch-reference | upstream-native-control |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |

## native / backward

| Case / dtype | torch-ms-deform-attn | kernel-hub-adapter | hf-native | mmcv-source | msda-triton-rziga | pytorch-reference | upstream-native-control |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |

## native / forward_backward

| Case / dtype | torch-ms-deform-attn | kernel-hub-adapter | hf-native | mmcv-source | msda-triton-rziga | pytorch-reference | upstream-native-control |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |

## Incremental peak allocation

MiB, FP32-compute encoder forward+backward; includes outputs and temporary tensors.

| dtype | torch-ms-deform-attn | kernel-hub-adapter | hf-native | mmcv-source | msda-triton-rziga | pytorch-reference | upstream-native-control |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| float32 | 37.19 | 37.19 | 37.19 | — | — | — | 37.19 |
| float16 | 69.06 | 69.06 | 69.06 | — | — | — | 69.06 |
| bfloat16 | 69.06 | 69.06 | 69.06 | — | — | — | 69.06 |

Missing entries are not zero latency. Consult the JSON for incorrect/unsupported cases.
The JSON also retains IQRs, event intervals, HF ratios and investigation flags.
