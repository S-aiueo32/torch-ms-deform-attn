# Kernel Hub benchmark results

Median of repetition medians; wall time in milliseconds.
FP32-compute rows include input/output casts. Native rows use different arithmetic.

## fp32-compute / forward

| Case / dtype | torch-ms-deform-attn | kernel-hub-adapter | hf-native | mmcv-source | msda-triton-rziga | pytorch-reference |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| decoder / bfloat16 | 0.0663 | 0.0668 | 0.0638 | 0.0641 | 0.0696 | 0.3471 |
| decoder / float16 | 0.0665 | 0.0665 | 0.0640 | 0.0643 | 0.0688 | 0.3484 |
| decoder / float32 | 0.0412 | 0.0412 | 0.0377 | 0.0375 | 0.0368 | 0.3137 |
| encoder / bfloat16 | 0.7874 | 0.7923 | 0.7493 | 0.7240 | 0.5999 | 8.6296 |
| encoder / float16 | 0.7952 | 0.7983 | 0.7551 | 0.7338 | 0.6054 | 8.6472 |
| encoder / float32 | 0.6713 | 0.6829 | 0.6101 | 0.6041 | 0.4615 | 8.5364 |
| small / bfloat16 | 0.0420 | 0.0421 | 0.0380 | 0.0348 | 0.0687 | 0.1079 |
| small / float16 | 0.0419 | 0.0420 | 0.0375 | 0.0345 | 0.0676 | 0.1092 |
| small / float32 | 0.0169 | 0.0173 | 0.0157 | 0.0135 | 0.0377 | 0.0821 |

## fp32-compute / backward

| Case / dtype | torch-ms-deform-attn | kernel-hub-adapter | hf-native | mmcv-source | msda-triton-rziga | pytorch-reference |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| decoder / bfloat16 | 0.1387 | 0.1384 | 0.1560 | 0.1470 | 0.1766 | 0.5942 |
| decoder / float16 | 0.1391 | 0.1385 | 0.1422 | 0.1885 | 0.1774 | 0.6082 |
| decoder / float32 | 0.1268 | 0.1285 | 0.1222 | 0.1284 | 0.1612 | 0.5188 |
| encoder / bfloat16 | 2.3170 | 2.3145 | 2.2304 | 2.3055 | 2.9532 | 11.5438 |
| encoder / float16 | 2.3359 | 2.3406 | 2.2525 | 2.3112 | 2.9639 | 11.6769 |
| encoder / float32 | 2.2404 | 2.2667 | 2.1407 | 2.2049 | 2.8368 | 11.5525 |
| small / bfloat16 | 0.1296 | 0.1887 | 0.1131 | 0.1209 | 0.1581 | 0.2425 |
| small / float16 | 0.1290 | 0.1696 | 0.1111 | 0.1505 | 0.1923 | 0.2437 |
| small / float32 | 0.1052 | 0.1072 | 0.0809 | 0.0735 | 0.1329 | 0.2608 |

## fp32-compute / forward_backward

| Case / dtype | torch-ms-deform-attn | kernel-hub-adapter | hf-native | mmcv-source | msda-triton-rziga | pytorch-reference |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| decoder / bfloat16 | 0.4645 | 0.4760 | 0.4224 | 0.4270 | 0.7205 | 1.4542 |
| decoder / float16 | 0.6262 | 0.6361 | 0.3725 | 0.4080 | 0.7197 | 1.6356 |
| decoder / float32 | 0.1993 | 0.3665 | 0.1614 | 0.2522 | 0.5007 | 1.4884 |
| encoder / bfloat16 | 3.2029 | 3.1987 | 3.0286 | 3.0828 | 3.5852 | 20.1500 |
| encoder / float16 | 3.2288 | 3.2329 | 3.0305 | 3.1119 | 3.5720 | 20.1260 |
| encoder / float32 | 3.0618 | 3.0449 | 2.8773 | 2.9201 | 3.4463 | 20.0102 |
| small / bfloat16 | 0.2784 | 0.6187 | 0.4261 | 0.4213 | 0.7038 | 1.0163 |
| small / float16 | 0.6157 | 0.6265 | 0.4546 | 0.4212 | 0.6212 | 1.0332 |
| small / float32 | 0.1937 | 0.1982 | 0.1378 | 0.1251 | 0.5248 | 0.9474 |

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
| float32 | 37.19 | 37.19 | 37.19 | 37.19 | 37.19 | 563.12 |
| float16 | 69.06 | 69.06 | 69.06 | 69.06 | 69.06 | 586.00 |
| bfloat16 | 69.06 | 69.06 | 69.06 | 69.06 | 69.06 | 586.00 |

Missing entries are not zero latency. Consult the JSON for incorrect/unsupported cases.
The JSON also retains IQRs, event intervals, HF ratios and investigation flags.
