# Runpod integration attempts

These failed infrastructure attempts do not establish CUDA correctness.
Actions artifacts retain the controller log and Pod cleanup state.

| Run / attempt | Outcome | GPU cleanup |
| --- | --- | --- |
| [35187020107 / 1](https://github.com/S-aiueo32/torch-ms-deform-attn/actions/runs/35187020107) | Runtime dependencies installed; apt failed under nonroot user. Moved system packages to root bootstrap. | `6pch77z9g77n1j` deleted |
| [35187438213 / 1](https://github.com/S-aiueo32/torch-ms-deform-attn/actions/runs/35187438213) | Pod creation rejected with HTTP 400 while L4 capacity was low. | No Pod created |
| [35187438213 / 2](https://github.com/S-aiueo32/torch-ms-deform-attn/actions/runs/35187438213) | Builder 0.16.0 rejected `general.edition`. Removed the field after checking its actual schema. | `fsyw3924kgfual` deleted |
| [35188407826 / 1](https://github.com/S-aiueo32/torch-ms-deform-attn/actions/runs/35188407826) | Configuration generation passed. Generated setup.py lacked `build_kernel`; switched to the pinned builder's CMake build and `local_install` targets. | `rolgy4w6luelfh` deleted |
| [35189678153 / 1](https://github.com/S-aiueo32/torch-ms-deform-attn/actions/runs/35189678153) | Cancelled during CPU preparation to remove redundant Rust compilation from the GPU workload. | GPU step skipped |
| [35189934488 / 1](https://github.com/S-aiueo32/torch-ms-deform-attn/actions/runs/35189934488) | CMake configured and CUDA source compiled; C++ binding failed on `torch::kCUDA`. Switched to `c10::DispatchKey::CUDA` and verified C++ syntax locally. | `5iuei6ls7tgq9l` deleted |
| [35190997354 / 1](https://github.com/S-aiueo32/torch-ms-deform-attn/actions/runs/35190997354) | CUDA artifact built and loaded. Layer eager/compile and autocast/validation tests passed. FP32/FP64 published-HF comparisons passed; FP16 gradient comparison failed (maximum reported absolute difference 0.009765625 versus 0.002 tolerance). E2E failed because resolving `__file__` followed a snapshot symlink into the shared blobs directory. Fixed snapshot directory handling without relaxing numerical tolerances. | `44yzubkk1cauz7` deleted |

The workflow now caches builder 0.16.0 on the CPU runner, generates the exported
project before GPU rental, and transfers it with the matching source revision.
The GPU executes dependency installation, CMake compilation and validation only.
