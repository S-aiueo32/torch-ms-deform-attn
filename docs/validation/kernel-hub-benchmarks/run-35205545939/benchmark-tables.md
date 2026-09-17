# Kernel Hub benchmark results

Median of repetition medians; wall time in milliseconds.
FP32-compute rows include input/output casts. Native rows use different arithmetic.

## fp32-compute / forward

| Case / dtype | torch-ms-deform-attn | kernel-hub-adapter | hf-native | mmcv-source | msda-triton-rziga | pytorch-reference |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| decoder / bfloat16 | 0.0664 | 0.0664 | 0.0639 | 0.0640 | — | 0.3717 |
| decoder / float16 | 0.0664 | 0.0664 | 0.0636 | 0.0643 | — | 0.3628 |
| decoder / float32 | 0.0412 | 0.0412 | 0.0390 | 0.0379 | — | 0.3228 |
| encoder / bfloat16 | 0.8408 | 0.8381 | 0.7787 | 0.7537 | — | 8.6610 |
| encoder / float16 | 0.8406 | 0.8402 | 0.7818 | 0.7519 | — | 8.6635 |
| encoder / float32 | 0.6856 | 0.6783 | 0.6049 | 0.5996 | — | 8.5296 |
| small / bfloat16 | 0.0413 | 0.0417 | 0.0374 | 0.0347 | — | 0.1071 |
| small / float16 | 0.0412 | 0.0414 | 0.0372 | 0.0345 | — | 0.1060 |
| small / float32 | 0.0168 | 0.0175 | 0.0152 | 0.0133 | — | 0.0806 |

## fp32-compute / backward

| Case / dtype | torch-ms-deform-attn | kernel-hub-adapter | hf-native | mmcv-source | msda-triton-rziga | pytorch-reference |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| decoder / bfloat16 | 0.1803 | 0.1379 | 0.1387 | 0.1965 | — | 0.5873 |
| decoder / float16 | 0.1377 | 0.1389 | 0.1493 | 0.1420 | — | 0.5904 |
| decoder / float32 | 0.1271 | 0.1260 | 0.1238 | 0.1249 | — | 0.5152 |
| encoder / bfloat16 | 2.3448 | 2.3506 | 2.2689 | 2.3039 | — | 11.6425 |
| encoder / float16 | 2.3663 | 2.3562 | 2.2507 | 2.3442 | — | 11.6816 |
| encoder / float32 | 2.2407 | 2.2421 | 2.1452 | 2.2008 | — | 11.6202 |
| small / bfloat16 | 0.1269 | 0.1292 | 0.1060 | 0.1131 | — | 0.2389 |
| small / float16 | 0.1272 | 0.1288 | 0.1150 | 0.1114 | — | 0.2371 |
| small / float32 | 0.0934 | 0.0933 | 0.0726 | 0.1042 | — | 0.2074 |

## fp32-compute / forward_backward

| Case / dtype | torch-ms-deform-attn | kernel-hub-adapter | hf-native | mmcv-source | msda-triton-rziga | pytorch-reference |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| decoder / bfloat16 | 0.6743 | 0.6356 | 0.3647 | 0.4233 | — | 1.4189 |
| decoder / float16 | 0.6279 | 0.6386 | 0.4217 | 0.4229 | — | 1.7083 |
| decoder / float32 | 0.3292 | 0.3360 | 0.2730 | 0.2015 | — | 1.4842 |
| encoder / bfloat16 | 3.2634 | 3.2359 | 2.9994 | 3.0522 | — | 20.1772 |
| encoder / float16 | 3.2697 | 3.2636 | 3.0594 | 3.0833 | — | 20.1387 |
| encoder / float32 | 3.0420 | 3.0197 | 2.9100 | 2.9202 | — | 20.2044 |
| small / bfloat16 | 0.6153 | 0.6271 | 0.4426 | 0.4175 | — | 1.0157 |
| small / float16 | 0.4533 | 0.2799 | 0.2070 | 0.1949 | — | 1.0024 |
| small / float32 | 0.4098 | 0.4212 | 0.2171 | 0.2434 | — | 0.9017 |

## native / forward

| Case / dtype | torch-ms-deform-attn | kernel-hub-adapter | hf-native | mmcv-source | msda-triton-rziga | pytorch-reference |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |

## native / backward

| Case / dtype | torch-ms-deform-attn | kernel-hub-adapter | hf-native | mmcv-source | msda-triton-rziga | pytorch-reference |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |

## native / forward_backward

| Case / dtype | torch-ms-deform-attn | kernel-hub-adapter | hf-native | mmcv-source | msda-triton-rziga | pytorch-reference |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |

## Incremental peak allocation

MiB, FP32-compute encoder forward+backward; includes outputs and temporary tensors.

| dtype | torch-ms-deform-attn | kernel-hub-adapter | hf-native | mmcv-source | msda-triton-rziga | pytorch-reference |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| float32 | 37.19 | 37.19 | 37.19 | 37.19 | — | 563.12 |
| float16 | 69.06 | 69.06 | 69.06 | 69.06 | — | 585.38 |
| bfloat16 | 69.06 | 69.06 | 69.06 | 69.06 | — | 585.38 |

Missing entries are not zero latency. Consult the JSON for incorrect/unsupported cases.
The JSON also retains IQRs, event intervals, HF ratios and investigation flags.
