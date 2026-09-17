この調査は PR #16 の `2d604d2d9e2871ce2dc957278bbdae211e4b3fa8` を対象に、GHA を介さず Runpod の L4 で実行した。以下は同じ CUDA 環境で計測を再実行する手順。diagnostic variant は固定した正常入力専用で、公開 API の代替ではない。

**実行環境**

NVIDIA L4、driver 595.91.07、CUDA toolkit 12.6.85、Python 3.11.13、PyTorch 2.10.0+cu126、kernels 0.16.0。HF revision は probe 内で `abfd4042216fa4f84c9c5c4e3e844a3143c70ad5` に固定している。依存パッケージ一覧と build log はアーカイブ内の `results/python-packages.txt`、`results/setup.log` にある。

新しい GPU 環境の `/workspace/ci/source` に上記 commit の checkout と、このディレクトリの `diagnostics/` を配置する。`setup.sh` は `/workspace/ci/profile-env` に venv を作成し、依存関係と canonical extension をインストールする。Pod の作成・削除はこれらの診断スクリプトに含まれない。

```bash
cd /workspace/ci/source
mkdir -p /workspace/ci/results
bash diagnostics/setup.sh
export PATH=/workspace/ci/profile-env/bin:$PATH
export CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=8.9 MAX_JOBS=2
python diagnostics/prepare_variants.py --repo . --output /workspace/ci/variants --build
```

**実際の測定コマンド**

run2 は decoder/encoder、FP32/FP16、forward/forward＋backward を比較した。run3 は別プロセス・profiler 無効・より長い測定窓で encoder の再現性を確認した。いずれも一つの非デフォルト CUDA stream を入力生成から計測終了まで使用し、各 backend の出力と全3勾配を FP64 oracle で検証してから計時する。

```bash
python diagnostics/probe.py --output-dir /workspace/ci/results/run2 \
  --variants-dir /workspace/ci/variants --repeats 5 --min-run-time 0.12
python diagnostics/probe.py --output-dir /workspace/ci/results/run3 \
  --variants-dir /workspace/ci/variants --cases encoder \
  --backends native_control hf baseline int32 no_metadata int32_no_metadata \
  --repeats 7 --min-run-time 0.4 --skip-profile
python diagnostics/host_probe.py --output-dir /workspace/ci/results/host-probe \
  --repeats 7 --min-run-time 0.4 --graph
bash diagnostics/single_cases.sh
python diagnostics/summarize.py /workspace/ci/results/run3/report.json \
  --variants-dir /workspace/ci/variants --quiet
python diagnostics/collect_binary.py /workspace/ci/results/run2/report.json \
  --variants-dir /workspace/ci/variants --output-dir /workspace/ci/results/binary
```

`single_cases.sh` は encoder の FP32 について 5 backend × 2 mode をそれぞれ新しい Python プロセスで profile する。全10ケースで、記録された forward/backward kernel 数が期待値の10回に一致した。

Nsight Compute は次の対象を実行したが、ホストの performance counter 権限により `ERR_NVGPUCTRPERM` となった。counter を取得できた結果は含まない。

```bash
ncu --profile-from-start off --section LaunchStats --section Occupancy \
  --export /workspace/ci/results/ncu-case/encoder-native --force-overwrite \
  python diagnostics/one_case.py --backend native_control --case encoder \
  --mode forward --output-dir /workspace/ci/results/ncu-case --ncu-mode
```

**保存したデータ**

- [report-run2-corrected.json](report-run2-corrected.json): 32件の正確性検証、320 timing rows、修正済み profiler 集計。
- [report-run3.json](report-run3.json)、[summary-run3.json](summary-run3.json): 12件の正確性検証、168 timing rows、7反復の集計。
- [host-probe.json](host-probe.json): 微小 CUDA 入力での4経路の wall、graph、Python profile。
- [single-profiles.json](single-profiles.json): 個別プロセスで取得した完全な kernel profile。
- [binary_summary.json](binary_summary.json): 実測バイナリの hash、sm_89 の resource と SASS 命令数。
- [profiling-evidence.tar.gz](profiling-evidence.tar.gz): Chrome traces、pstats、元 report、修正版 report、SASS、variant source、build log、診断スクリプトを含む171ファイル。SHA-256 は `ba9e8ab68edca662473de1470e6a63f6b81c7e757cfb5d83789f3ff6350b369e`。

アーカイブの `results/evidence-manifest.json` に各170ファイルの hash とサイズを記録し、取得後に全件を検証した。CUDA extension のバイナリ本体と認証情報は含まない。`variants/manifest.json` と各 variant の `build_config.json`、`build_result.json`、ソース、ビルドログから再ビルドできる。

初期 run1 は autograd の stream mismatch 警告が出たため中断し、速度の根拠から除外した。アーカイブ内の `results/experiment.log` はこの初期実行の履歴で、run1 の測定 report は含めていない。

run2 の旧集計は GPU annotation と実 kernel を二重計上していた。保存済み trace の修正には以下を使った。現在の `probe.py` は既に同じ category filtering と kernel 数検証を実装している。欠落がある run2 encoder profile を全体時間の根拠に使わず、完全な `single-profiles.json` を参照する。

```bash
python diagnostics/reprocess_profiles.py /workspace/ci/results/run2/report.json
```

アーカイブを別ディレクトリに展開して再集計する場合は、`--path-map /workspace/ci/results=/absolute/path/to/extracted/results` を付ける。元 report は上書きされず、`report-corrected.json` が生成される。

計測に使った Pod と、転送失敗で終了した先行 Pod は両方とも削除確認済み。[計測 Pod](cleanup.json)、[先行 Pod](cleanup-attempt1.json)。
