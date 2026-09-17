# Kernel Hub benchmark results

Median of repetition medians; wall time in milliseconds.
FP32-compute rows include input/output casts. Native rows use different arithmetic.

## fp32-compute / forward

| Case / dtype | torch-ms-deform-attn | kernel-hub-adapter | hf-native | mmcv-source | msda-triton-rziga | pytorch-reference | upstream-native-control | upstream-before-perf |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| decoder / float16 | 0.0665 | 0.0663 | 0.0642 | — | — | — | — | 0.0663 |
| decoder / float32 | 0.0416 | 0.0416 | 0.0381 | — | — | — | — | 0.0413 |
| encoder / float16 | 0.8339 | 0.8332 | 0.7655 | — | — | — | — | 0.8311 |
| encoder / float32 | 0.6715 | 0.6719 | 0.5984 | — | — | — | — | 0.6678 |

## fp32-compute / backward

| Case / dtype | torch-ms-deform-attn | kernel-hub-adapter | hf-native | mmcv-source | msda-triton-rziga | pytorch-reference | upstream-native-control | upstream-before-perf |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |

## fp32-compute / forward_backward

| Case / dtype | torch-ms-deform-attn | kernel-hub-adapter | hf-native | mmcv-source | msda-triton-rziga | pytorch-reference | upstream-native-control | upstream-before-perf |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| decoder / float16 | 0.4793 | 0.4655 | 0.3999 | — | — | — | — | 0.5381 |
| decoder / float32 | 0.1911 | 0.2890 | 0.1626 | — | — | — | — | 0.3299 |
| encoder / float16 | 3.2150 | 3.2100 | 3.0365 | — | — | — | — | 3.2099 |
| encoder / float32 | 3.0571 | 3.0579 | 2.9148 | — | — | — | — | 3.0607 |

## native / forward

| Case / dtype | torch-ms-deform-attn | kernel-hub-adapter | hf-native | mmcv-source | msda-triton-rziga | pytorch-reference | upstream-native-control | upstream-before-perf |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |

## native / backward

| Case / dtype | torch-ms-deform-attn | kernel-hub-adapter | hf-native | mmcv-source | msda-triton-rziga | pytorch-reference | upstream-native-control | upstream-before-perf |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |

## native / forward_backward

| Case / dtype | torch-ms-deform-attn | kernel-hub-adapter | hf-native | mmcv-source | msda-triton-rziga | pytorch-reference | upstream-native-control | upstream-before-perf |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |

## Incremental peak allocation

MiB, FP32-compute encoder forward+backward; includes outputs and temporary tensors.

| dtype | torch-ms-deform-attn | kernel-hub-adapter | hf-native | mmcv-source | msda-triton-rziga | pytorch-reference | upstream-native-control | upstream-before-perf |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| float32 | 37.19 | 37.19 | 37.19 | — | — | — | — | 37.19 |
| float16 | 69.06 | 69.06 | 69.06 | — | — | — | — | 69.06 |
| bfloat16 | — | — | — | — | — | — | — | — |

Missing entries are not zero latency. Consult the JSON for unselected, incorrect or unsupported cases.
The JSON also retains IQRs, event intervals, HF ratios and investigation flags.
