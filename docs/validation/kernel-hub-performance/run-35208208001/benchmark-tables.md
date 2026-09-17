# Kernel Hub benchmark results

Median of repetition medians; wall time in milliseconds.
FP32-compute rows include input/output casts. Native rows use different arithmetic.

## fp32-compute / forward

| Case / dtype | torch-ms-deform-attn | kernel-hub-adapter | hf-native | mmcv-source | msda-triton-rziga | pytorch-reference | upstream-native-control |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| decoder / bfloat16 | 0.0632 | 0.0631 | 0.0604 | 0.0600 | 0.0825 | 0.3204 | 0.0633 |
| decoder / float16 | 0.0630 | 0.0629 | 0.0604 | 0.0599 | 0.0833 | 0.3216 | 0.0630 |
| decoder / float32 | — | — | 0.0371 | 0.0368 | — | 0.3010 | — |
| encoder / bfloat16 | 0.7471 | 0.7465 | 0.6916 | 0.6771 | 0.5563 | 8.5093 | 0.7467 |
| encoder / float16 | 0.7484 | 0.7521 | 0.6991 | 0.6855 | 0.5624 | 8.5184 | 0.7529 |
| encoder / float32 | — | — | 0.5799 | 0.5635 | — | 8.4424 | — |
| small / bfloat16 | 0.0492 | 0.0499 | 0.0508 | 0.0464 | 0.0802 | 0.1333 | 0.0428 |
| small / float16 | 0.0489 | 0.0494 | 0.0495 | 0.0471 | 0.0796 | 0.1356 | 0.0419 |
| small / float32 | 0.0166 | 0.0176 | 0.0195 | 0.0171 | 0.0414 | 0.1015 | 0.0128 |

## fp32-compute / backward

| Case / dtype | torch-ms-deform-attn | kernel-hub-adapter | hf-native | mmcv-source | msda-triton-rziga | pytorch-reference | upstream-native-control |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| decoder / bfloat16 | 0.1845 | 0.1553 | 0.1349 | 0.1488 | 0.2331 | 0.5666 | 0.1313 |
| decoder / float16 | 0.1492 | 0.1851 | 0.1307 | 0.1616 | 0.1900 | 0.5728 | 0.1351 |
| decoder / float32 | — | — | 0.1269 | 0.1154 | — | 0.4911 | — |
| encoder / bfloat16 | 2.2437 | 2.2422 | 2.1497 | 2.2122 | 2.8276 | 10.9035 | 2.2349 |
| encoder / float16 | 2.2533 | 2.2675 | 2.1741 | 2.2279 | 2.8334 | 11.0041 | 2.2504 |
| encoder / float32 | — | — | 2.0601 | 2.1281 | — | 10.9970 | — |
| small / bfloat16 | 0.1489 | 0.1505 | 0.1308 | 0.1273 | 0.1879 | 0.2982 | 0.1260 |
| small / float16 | 0.1843 | 0.1501 | 0.1093 | 0.1400 | 0.1828 | 0.3008 | 0.1226 |
| small / float32 | 0.1022 | 0.1022 | 0.0854 | 0.0893 | 0.1750 | 0.2547 | 0.0796 |

## fp32-compute / forward_backward

| Case / dtype | torch-ms-deform-attn | kernel-hub-adapter | hf-native | mmcv-source | msda-triton-rziga | pytorch-reference | upstream-native-control |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| decoder / bfloat16 | 0.6231 | 0.5993 | 0.4957 | 0.5017 | 0.7588 | 1.8308 | 0.5133 |
| decoder / float16 | 0.7043 | 0.6813 | 0.5348 | 0.5170 | 0.7383 | 2.0557 | 0.3162 |
| decoder / float32 | — | — | 0.3170 | 0.1521 | — | 1.7521 | — |
| encoder / bfloat16 | 3.0680 | 3.0681 | 2.9131 | 2.9650 | 3.4483 | 19.5165 | 3.0562 |
| encoder / float16 | 3.0700 | 3.0972 | 2.9407 | 2.9923 | 3.4638 | 19.5885 | 3.0936 |
| encoder / float32 | — | — | 2.8094 | 2.8174 | — | 19.5784 | — |
| small / bfloat16 | 0.6513 | 0.5784 | 0.4642 | 0.5190 | 0.5395 | 1.3223 | 0.4424 |
| small / float16 | 0.6493 | 0.6241 | 0.3801 | 0.2379 | 0.7378 | 1.3194 | 0.4972 |
| small / float32 | 0.4513 | 0.3605 | 0.2633 | 0.2232 | 0.6271 | 1.1054 | 0.0948 |

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
| float32 | — | — | 37.19 | 37.19 | — | 563.12 | — |
| float16 | 69.06 | 69.06 | 69.06 | 69.06 | 69.06 | 586.00 | 69.06 |
| bfloat16 | 69.06 | 69.06 | 69.06 | 69.06 | 69.06 | 586.00 | 69.06 |

Missing entries are not zero latency. Consult the JSON for incorrect/unsupported cases.
The JSON also retains IQRs, event intervals, HF ratios and investigation flags.
