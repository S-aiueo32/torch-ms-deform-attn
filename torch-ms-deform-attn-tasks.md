# torch-ms-deform-attn: 未実装・検証不足のタスク一覧

## 調査範囲と判定

- 対象: `S-aiueo32/torch-ms-deform-attn` の `main`。2026-09-14閲覧時のGitHub表示commitは `e7fe5a1`。
- 調査方法: 公開ソース、4つのオペレータテストファイル、ビルド設定、CI、ベンチマーク、ドキュメントの静的レビュー。今回ビルドやテストは実行していない。Git cloneとソースのダウンロードはこの実行環境では失敗したため、Webで取得した公開ソースを読んだ。
- 「検証不足」はリポジトリ内のテストまたは公開実行記録で確認できない範囲。開発者がローカルで一度も実行していない、という意味ではない。
- P1: 現在説明している対応範囲の信頼性・安全性を補強する。P2: 追加の統合検証・再現性・性能評価。任意: 現在の非対応範囲を拡張する場合のみ採用する。
- 今回の調査で、再現済みの重大不具合はない。以下は不具合を断定した一覧ではない。

CPU/CUDA forward・一次backward、AMPのfloat32昇格、基本的なtorch.compile、CPUの非連続入力・空テンソル、CUDAの端数batch・非default stream・決定性設定による拒否、CUDA index planning、sdistからのwheel作成とcheckout外でのテストは既に実装されている。[S02–S10]

L4 / Python 3.11 / PyTorch 2.5.1 / CUDA 12.4では、36テスト中2件skipとmemcheck 0 errorsの記録がある。公開Actionsの成功表示はcommit `36d5d60` の実行で、今回確認したmainと同一SHAではない。これだけでmainが未検証と断定はできないが、検証結果を対象revisionに紐付ける余地がある。[S11–S12]

## 既存仕様・品質保証を補うタスク

### T01 — 対応Python / PyTorch / CUDAの検証マトリクスを定義する

**優先度:** P1　**種別:** 検証不足　**環境:** CPU＋一部GPU

**根拠:** 依存条件はPython >=3.10、PyTorch >=2.5,<3だが、通常CIはPython 3.11 / PyTorch 2.5.1、CUDAビルドは12.4に固定されている。[S03–S04, S13]

**対象:** `pyproject.toml`, `.github/workflows/ci.yml`, `cuda-build.yml`, `scripts/run_cuda_checks.sh`, `docs/installation.md`

- [ ] サポート対象のPython/PyTorch組み合わせを明文化する。配布されていない組み合わせを機械的な直積で要求しない。
- [ ] 少なくとも宣言下限と、サポートする新しいPyTorch系列をCPU CIで検証する。
- [ ] CUDAについてもサポートするPyTorch/toolkitの組み合わせでビルド・実機テストを実施する。
- [ ] 対応すると記載する範囲を、検証結果または明示的なbest-effort方針と整合させる。

**完了条件:** 対応表の各行にbuild/test結果があり、未確認の行は未確認と区別できる。旧版から新版へのextension再ビルドも確認できる。

### T02 — CUDA検証をrevision単位のリリース条件にする

**優先度:** P1　**種別:** 運用上の検証不足　**環境:** CPU制御＋GPU

**根拠:** GPU実行は`workflow_dispatch`、push/PRのCUDA workflowはGPUなしのビルド確認。実機ログの保存は7日間である。[S04, S11–S14]

**対象:** `.github/workflows/cuda.yml`, `cuda-build.yml`, `docs/gpu-runner.md`

- [ ] リリース対象SHAに対するGPUテスト結果を必須の確認項目にする。課金GPUの毎PR自動起動は前提にしない。
- [ ] 実行結果にsource SHA、GPU、driver、Python、PyTorch、toolkit、ビルド設定、テスト件数、skip理由を記録する。
- [ ] 期待していないCUDAテストskipを失敗扱いにする。
- [ ] リリースの検証サマリと重要ログを長期参照できる場所に残す。

**完了条件:** あるリリースのソースと検証結果を一意に照合でき、CPU-onlyの成功をGPU検証成功と取り違えない。

### T03 — CUDA reductionの全特殊化と分岐境界を回帰テストする

**優先度:** P1　**種別:** 検証不足　**環境:** GPU

**根拠:** CUDA correctnessの主要ケースはD=2,3,32,64,71,1024,1025,2048。カーネルにはそれ以外のD専用分岐もある。D=16はCUDA benchmarkでも使われるが、全特殊化をfloat32/64で一貫して検証する表にはなっていない。[S06, S15–S16]

**対象:** `tests/test_cuda.py::test_reference_forward_backward`, `csrc/cuda/ms_deform_im2col_cuda.cuh::ms_deformable_col2im_cuda`

- [ ] D=1,2,4,8,16,32,64,128,256,512,1024の特殊化をfloat32/64で検証する。
- [ ] D=31,33,63,65,127,129,1023,1025,2048,2049などのgeneric/multi-block境界を追加する。
- [ ] batch=1、端数chunk、複数head、複数queryを組み合わせる。全組み合わせの直積は不要。
- [ ] forwardとvalue/locations/weightsの3勾配をreferenceと比較する。

**完了条件:** dispatcherの各分岐と主要境界を少なくとも1ケースが通り、精度別許容誤差の根拠を説明できる。

### T04 — CPUの境界・offset・衝突テストをCUDAへ展開する

**優先度:** P1　**種別:** 検証不足　**環境:** CPU＋GPU

**根拠:** CPUにはpadding境界、任意のlevel_start_index、重複levelと衝突samplingの専用テストがある。CUDA側はランダムな範囲外座標を検証するが、これらと同等の専用ケースはない。[S05–S06]

**対象:** `tests/test_cpu.py`, `tests/test_cuda.py`、共有fixture用の新規テストヘルパー

- [ ] CPUのpadding/support-boundaryケースを共有化しCUDAでも実行する。
- [ ] H=1/W=1、座標0/1、片側・四隅のpadding、完全範囲外を追加する。
- [ ] 非連続・逆順・重複するlevel offsets、同一点へ大量に集まるsamplingを検証する。
- [ ] `shapes`と`starts`を含むnoncontiguous入力もGPUで検証する。
- [ ] offset付きreferenceはvalueを対応する順序に再構成して作る。現行referenceにない引数を仮定しない。

**完了条件:** forwardと3勾配がreferenceと一致する。微分不能な境界はforward/採用する勾配規約の比較とし、有限差分gradcheckは境界を避ける。CUDAの反復結果にはatomic由来の誤差を許容し、bitwise一致を要求しない。

### T05 — racecheck / synccheckの実行証跡とinitcheck経路を追加する

**優先度:** P1　**種別:** 検証記録不足＋実行経路の未実装　**環境:** GPU

**根拠:** 公開記録はmemcheck。racecheck/synccheckの選択肢は既にあるが、成功記録は確認できない。initcheckは選択肢にない。sanitizer対象は現在2つのテストメソッド。[S11, S14, S17]

**対象:** `scripts/run_cuda_checks.sh`, `.github/workflows/cuda.yml`, `scripts/runpod_ci.py`のsanitizer引数、`docs/gpu-runner.md`

- [ ] 既存のracecheckとsynccheckを実行し、結果をrevision付きで保存する。
- [ ] initcheckを引数・workflow・テストに追加する。
- [ ] T03/T04の主要ケースをsanitizer対象に含める。
- [ ] device assertionを意図的に起こすnegative testsを、正常ケースのsanitizer判定と分離する。

**完了条件:** 採用したsanitizerの正常ケースがエラー0件。toolが対象外とするメモリ種別・検査範囲を越えて「raceなし」と保証しない。

### T06 — FakeTensor実装に構造的な入力検証を加える

**優先度:** P1　**種別:** 未実装　**環境:** CPUで実装・検証可能

**根拠:** `_forward_fake`は出力shapeを作るだけ、`_backward_fake`は勾配用Tensorを作るだけ。native側にあるrank/dtype/shape整合性の検査がない。[S02, S18–S19]

**対象:** `src/torch_ms_deform_attn/_ops.py::_forward_fake`, `_backward_fake`, `tests/test_integration.py`

- [ ] rank、metadataのint64、浮動小数dtype、device、共有次元の一致、step、grad shapeを、メタ情報から判断できる範囲で検査する。
- [ ] symbolic shapeを保持できる検査を使い、SymIntのPython整数化やTensorデータ読み出しを避ける。
- [ ] CPUとCUDAで異なるempty-input規約を保持する。
- [ ] fake/eagerの不正入力テストを追加する。

**完了条件:** 構造的に不正な入力はfakeでも拒否され、正常なdynamic-shape compile/opcheckが回帰しない。spatial_shapes/startsの実データ検証はnativeに残す。

### T07 — 入力契約のnegative testsをCPU/CUDAで体系化する

**優先度:** P1　**種別:** 検証不足　**環境:** CPU＋GPU

**根拠:** nativeの検査項目に比べ、CPU/CUDAの通常negative testsは少数。CUDA metadataのdevice assertionとindex planningのoverflow検証は既に存在するので、そこを未実装として扱わない。[S05–S09, S18–S19]

**対象:** `tests/test_cpu.py`, `tests/test_cuda.py`, `tests/test_integration.py`

- [ ] value/shapes/starts/locations/weightsのrankと共有次元不整合を表形式で生成する。
- [ ] metadata int32、floating dtype混在、autocast外fp16/bf16、step異常を検証する。
- [ ] device混在、backwardのgrad rank/dtype/device不整合を検証する。
- [ ] CUDAの空次元拒否を各次元で確認する。CPUに許されている空入力を誤って禁止しない。
- [ ] 公開関数とlegacy `.apply`で入力契約が一致することを確認する。

**完了条件:** documentedな各構造制約に少なくとも1つの拒否テストがあり、crash/hangではなく期待したエラーになる。CUDA contextを壊すケースは子プロセスに隔離する。

### T08 — 常時skipされるGPU構成の既存テストを実行する

**優先度:** P2　**種別:** テスト実装済み・実行記録不足　**環境:** 1 GPU＋別途2 GPUs

**根拠:** `test_noncurrent_device`は2 GPUが必要。`CPUOnlyBuildTest.test_cuda_input_error`はGPU上のCPU-only wheelが必要。現在のRunpod経路は1 GPUかつCUDA buildを強制するため、この2条件を満たさない。[S06, S11, S17, S20]

**対象:** `tests/test_cuda.py`, `scripts/run_cuda_checks.sh`、GPU実行構成

- [x] 1 GPU環境でCPU-only wheelを新規構築し、CUDA入力への説明的なエラーを確認する。
- [x] 2 GPU環境でnoncurrent-deviceテストをskipせず実行する。
- [x] noncurrent-deviceの3勾配もreference比較し、current deviceが復元されることを確認する。

**完了条件:** それぞれのテストにskipでない成功記録がある。必ずしもRunpod controllerを複数GPU対応へ改造する必要はなく、別の手動検証環境でもよい。

### T09 — AMPを実際の学習stepで検証する

**優先度:** P1　**種別:** 検証不足　**環境:** CPU＋GPU

**根拠:** 低精度入力からのfloat32計算と勾配比較は存在するが、GradScaler/optimizerを含む学習stepのテストはない。float64のautocast維持テストはCPUのみ。[S07, S10, S21]

**対象:** `tests/test_integration.py`, `tests/test_cuda.py`, `src/torch_ms_deform_attn/functional.py`

- [ ] FP32 parameterからAMPのoperator入力を作る小型moduleで、GradScaler→backward→unscale→optimizer.stepをテストする。
- [ ] CPU/CUDA、fp16/bf16、eager/compileについて実際に対応する組み合わせを検証する。
- [ ] `.apply`経由のAMP、CUDA float64保持、浮動入力の混在パターンを確認する。
- [ ] requires_gradが一部だけ有効、勾配蓄積、retain_graph、no_grad/inference_modeのケースを追加する。

**完了条件:** reference版とloss・勾配・更新後parameterが規定誤差内で一致する。意図したfp32出力と元の入力への勾配dtypeも確認する。

### T10 — CUDAのopcheckとdynamic-shape検証を拡張する

**優先度:** P2　**種別:** 検証不足　**環境:** GPU

**根拠:** `opcheck`はCPU入力のみ。CPUのdynamic compileはN/Qを変更するが、CUDA専用テストは主にNの変更を検証する。[S06–S07]

**対象:** `tests/test_integration.py::test_opcheck`, `tests/test_cuda.py::test_compile_dynamic`

- [x] forward/backwardのCUDA opcheckをfloat32/64、noncontiguous、複数requires_grad条件で実施する。
- [x] N/Qに加え、S/M/D/L/Pとfeature resolutionが変わる入力を検証する。
- [x] 同一shapeでもmetadataの値が変わる呼び出しで結果が正しく更新されることを確認する。
- [x] fullgraph、AMP併用、3勾配をreference比較する。

**完了条件:** 対応するshape変化でgraph breakや誤結果がない。再compile回数を観測し、unit dimension等の正当なspecializationと想定外の再compileを区別する。

### T11 — ビルド設定の自動判定・失敗経路を検証する

**優先度:** P2　**種別:** 検証不足　**環境:** CPU中心

**根拠:** CPU CIはFORCE_OPENMP=0/1を指定する。setup.pyには自動判定、fallback、native backend、compiler/probe失敗、macOS architecture処理など、それ以外の分岐がある。[S03, S22–S23]

**対象:** `setup.py`, `.github/workflows/ci.yml`、新規build tests

- [x] FORCE_OPENMP未設定の成功・serial fallbackと、強制ON時の失敗を検証する。
- [x] FORCE_CPU/FORCE_CUDA競合、toolkitなしのFORCE_CUDA、OMP_PREFIX不正、ARCHFLAGS不一致をテストする。
- [x] ninja有無、build option変更後の再ビルドで古いobjectが残らないことを確認する。
- [ ] native thread-pool backendを対応範囲に残すなら、そのPyTorch buildで実行結果を記録する。

**対応メモ（2026-09-15）:** native thread-pool の実PyTorchビルドは未検証であり、対応保証の対象にせずbest effortと明記。その他の実ビルド・自動fallback・失敗経路の記録は `docs/validation/2026-09-15-p2/` を参照。

**完了条件:** 分岐ごとに期待backendまたは説明的エラーが確認できる。コンパイラをmockしたunit testだけで実バイナリのリンク・ロード成功を保証したことにしない。

### T12 — 性能・メモリ計測の未カバー範囲と失敗判定を補う

**優先度:** P2　**種別:** 計測機能未実装＋検証不足　**環境:** CPU＋GPU

**根拠:** GPU benchmarkはeager float32、3種類のsynthetic workload、forward/forward+backward。compile benchmarkはCPUのみ。またcompile harnessは例外をJSONへ記録して続行し、失敗終了を要求するモードがない。[S15–S16, S24]

**対象:** `benchmarks/benchmark_cuda.py`, `benchmark_compile.py`, `docs/benchmarks.md`

- [x] CUDA compile、AMP、encoder相当の大きいQ、複数D・batch・im2col_stepを測定する。
- [x] backward単独とピークメモリを追加し、Python dispatchを含む指標とGPU kernel時間を区別する。
- [x] `benchmark_compile.py`に`--strict`等を設け、correctness/compile失敗を終了コードに反映する。
- [x] CPU型番、GPU/driver、ソースSHA、build flags、seed、warmup、IQRを結果へ残す。
- [x] 同一環境で繰り返し測定し、ノイズを含む回帰判定基準を定義する。

**完了条件:** correctnessを満たしたケースだけが性能比較に入り、計測エラーが成功扱いにならない。operator単体のspeedupをモデル全体のspeedupとして示さない。

### T13 — 上流実装との置換互換性をmodule単位で検証する

**優先度:** P2　**種別:** 統合検証不足　**環境:** CPU＋GPU

**根拠:** `.apply`互換入口はあるが、確認したtestsはoperator/reference単位。実際のupstream moduleで差し替えるfixtureはない。projectionや検出モデル自体はパッケージのスコープ外であり、追加実装する必要はない。[S01, S05–S07, S10]

**対象:** 新規`tests/integration/`、互換性のドキュメント

- [x] 導入対象とする上流moduleのrevisionを固定して、operatorだけを置換するfixtureを用意する。
- [x] 同じparameter/inputでforward、3入力勾配、module parameter勾配を比較する。
- [x] CPU/CUDA/AMP、短いoptimizer stepで互換性を確認する。
- [x] 必要なimport変更と、戻り値dtype・空入力などの差異を明文化する。

**完了条件:** 指定した上流revisionとのmodule-level同等性が示される。モデル本体をこのパッケージに取り込むことや、フル学習の精度再現を必須条件にはしない。

## 任意拡張: 現在の非対応・非保証を拡張する場合のみ

以下を既存仕様の不具合や必須実装として扱わない。

### E01 — 必要な勾配だけ計算する経路

**状態:** 最適化未実装。autogradは常に3勾配を要求し、native backwardも3バッファを確保する。[S02, S18–S19]

**タスク:** `ctx.needs_input_grad`等から勾配maskを受け渡し、不要な計算・allocationを省く。

**完了条件:** 全requires_gradパターンで結果一致、compile/opcheck非回帰、対象ケースで時間・メモリ改善。複雑化に見合う改善がなければ採用しない。

### E02 — native float16 / bfloat16 kernel

**状態:** 明示的に非対応。現行AMPはfloat32へ昇格するもので、native低精度kernelではない。[S10, S21]

**タスク:** CPU/CUDAのどちらを対象とするか定義し、入出力・accumulation dtypeの仕様を決めて実装する。

**完了条件:** forward・3勾配・AMP・sanitizer検証、float32基準の誤差評価、実測による速度/メモリ上の利益を示す。現在のAMP出力dtypeを無断で変更しない。

### E03 — CUDAの空テンソル対応

**状態:** CUDAでは明示的に拒否、CPUには対応とテストがある。[S05, S10, S19]

**タスク:** N/Q/M/D/L/P等が0の意味を定義し、kernelを起動せず適切な出力・zero gradientを返す。

**完了条件:** CPUとの意味上の一致と、eager/compile/AMPのshape・勾配検証。S=0の可否はmetadataとの整合条件も定義する。

### E04 — 決定的CUDA backward

**状態:** 未実装。現在は非決定的なatomic実装で、deterministic modeではエラーまたは警告になる仕様。[S06, S10, S19]

**タスク:** 決定的reduction経路を設計し、速度・メモリ制約と選択条件を定義する。

**完了条件:** 保証する同一環境で反復結果が一致し、reference精度を満たす。通常経路の性能を回帰させない。

### E05 — 高階微分・torch.func対応範囲の追加

**状態:** 高階微分は明示的に非対応。vmap/JVPについては専用登録・検証が確認できないため、既に動くとも全面非対応とも断定しない。[S02, S07, S10]

**タスク:** double backward、vmap、JVPを別々のサブタスクとして仕様化し、採用するものだけ実装・検証する。

**完了条件:** 高階微分はgradgradcheck、vmapはloop同等性、JVPは独立referenceとの比較を通す。

### E06 — torch.export / ONNX対応範囲の定義と検証

**状態:** 対応はclaimされていない。FakeTensor登録があるため、torch.exportまで必ず未実装とは言えない。ONNX用変換経路は確認できない。[S02, S10]

**タスク:** (a) torch.exportとsave/loadの実動作を先に評価する。(b) ONNXはdecompositionかcustom opかを選び、対応runtime・opset・dynamic shapeを定義する。

**完了条件:** export成功だけでなく、指定runtimeでの実行結果がreferenceと一致する。offset/重複level等、現行operatorの入力契約をどこまで維持するか明記する。

### E07 — 未保証platformの対応判断

**状態:** MPSは明示的非対応。WindowsはCI対象外。ROCmの実行保証は確認できない。3者を同じ「kernel未実装」と断定しない。[S01, S22–S23]

**タスク:** MPS、Windows/MSVC、ROCmを独立した採否判断に分ける。採用する対象だけbuild・forward/backward・AMPの実機検証を用意し、必要な実装を追加する。

**完了条件:** 対応表にOS/compiler/deviceごとの結果があり、対象外は明記される。

### E08 — 公開済みバイナリwheelの配布

**状態:** source distributionを配布する方針。ローカル/CIでwheelを作る機能は既にあり、wheel buildそのものが未実装ではない。[S03–S04, S13]

**タスク:** バイナリ配布する場合だけ、Python/PyTorch/CUDA/platform/architectureの互換性区分と配布先を設計し、build・install CIを追加する。

**完了条件:** サポートするクリーン環境にコンパイラなしで導入でき、非対応組み合わせへ誤導入させない。sdist-onlyを維持する判断でもよい。

## 根拠ソース

コードの参照は閲覧時の`main`を指す。`e7fe5a1`を使った同じパスでrevisionを固定できる。未公開ログや非公開設定は参照していない。

- [S01] commit/file tree: `https://github.com/S-aiueo32/torch-ms-deform-attn/commit/main`
- [S02] dispatcher/fake/autograd: `https://raw.githubusercontent.com/S-aiueo32/torch-ms-deform-attn/main/src/torch_ms_deform_attn/_ops.py`
- [S03] CPU CI: `https://raw.githubusercontent.com/S-aiueo32/torch-ms-deform-attn/main/.github/workflows/ci.yml`
- [S04] CUDA build CI: `https://raw.githubusercontent.com/S-aiueo32/torch-ms-deform-attn/main/.github/workflows/cuda-build.yml`
- [S05] CPU tests: `https://raw.githubusercontent.com/S-aiueo32/torch-ms-deform-attn/main/tests/test_cpu.py`
- [S06] CUDA tests: `https://raw.githubusercontent.com/S-aiueo32/torch-ms-deform-attn/main/tests/test_cuda.py`
- [S07] Integration tests: `https://raw.githubusercontent.com/S-aiueo32/torch-ms-deform-attn/main/tests/test_integration.py`
- [S08] Index tests: `https://raw.githubusercontent.com/S-aiueo32/torch-ms-deform-attn/main/tests/test_indexing.py`
- [S09] Index planning: `https://raw.githubusercontent.com/S-aiueo32/torch-ms-deform-attn/main/csrc/cuda/index_utils.h`
- [S10] API: `https://github.com/S-aiueo32/torch-ms-deform-attn/blob/main/docs/api.md`
- [S11] GPU validation record: `https://raw.githubusercontent.com/S-aiueo32/torch-ms-deform-attn/main/docs/gpu-runner.md`
- [S12] Successful CUDA run: `https://github.com/S-aiueo32/torch-ms-deform-attn/actions/runs/34704425778`
- [S13] Metadata: `https://raw.githubusercontent.com/S-aiueo32/torch-ms-deform-attn/main/pyproject.toml`; development/distribution policy: `https://raw.githubusercontent.com/S-aiueo32/torch-ms-deform-attn/main/docs/development.md`
- [S14] GPU workflow: `https://raw.githubusercontent.com/S-aiueo32/torch-ms-deform-attn/main/.github/workflows/cuda.yml`
- [S15] CUDA kernel: `https://raw.githubusercontent.com/S-aiueo32/torch-ms-deform-attn/main/csrc/cuda/ms_deform_im2col_cuda.cuh`
- [S16] CUDA benchmark: `https://raw.githubusercontent.com/S-aiueo32/torch-ms-deform-attn/main/benchmarks/benchmark_cuda.py`
- [S17] GPU test runner: `https://raw.githubusercontent.com/S-aiueo32/torch-ms-deform-attn/main/scripts/run_cuda_checks.sh`
- [S18] CPU kernel: `https://raw.githubusercontent.com/S-aiueo32/torch-ms-deform-attn/main/csrc/ms_deform_attn_cpu.cpp`
- [S19] CUDA entry points: `https://raw.githubusercontent.com/S-aiueo32/torch-ms-deform-attn/main/csrc/cuda/ms_deform_attn_cuda.cu`
- [S20] GPU controller: `https://raw.githubusercontent.com/S-aiueo32/torch-ms-deform-attn/main/scripts/runpod_ci.py`
- [S21] Functional API: `https://raw.githubusercontent.com/S-aiueo32/torch-ms-deform-attn/main/src/torch_ms_deform_attn/functional.py`
- [S22] Build: `https://raw.githubusercontent.com/S-aiueo32/torch-ms-deform-attn/main/setup.py`
- [S23] Installation: `https://raw.githubusercontent.com/S-aiueo32/torch-ms-deform-attn/main/docs/installation.md`
- [S24] Compile benchmark: `https://raw.githubusercontent.com/S-aiueo32/torch-ms-deform-attn/main/benchmarks/benchmark_compile.py`; benchmark limitations: `https://raw.githubusercontent.com/S-aiueo32/torch-ms-deform-attn/main/docs/benchmarks.md`


## P2 対応記録（2026-09-15）

T08・T10・T12・T13 は実装とCPU/GPU実行を確認済み。T11はOpenMP/serial、
Ninja有無、設定切替による再ビルドと失敗経路を確認済み。native thread-poolの
実PyTorchビルドのみ未検証で、best effortと明記した。

詳細、検証対象ソース、GPU削除記録、計測JSONは
[検証サマリ](docs/validation/2026-09-15-p2/README.md)を参照。
