# 初期プロファイリング：HFとの差を生む処理

PR #16 の `2d604d2` をRunpod L4で調べ、残存する64-bit添字計算、各thread／levelのメタデータ検証、Python autograd経路の追加コストを特定した。このレポートは診断用variantの比較であり、現在の実装の性能は[後続の変更と検証](../README.md)を参照する。

## 比較条件

2026-09-17、L4（sm_89）、PyTorch 2.10.0+cu126、CUDA 12.6、EPYC 9254。HF revisionは `abfd4042216fa4f84c9c5c4e3e844a3143c70ad5` に固定した。decoderはN=2、Q=300、encoderはN=2、Q=5440。共通条件は8 heads、32 channels、4 levels、4 sampling points。FP16もFP32計算とcastに統一した。

元のCUDAソースから、添字計算のint32化とメタデータ検証除去を独立に変更した2×2のvariantを同じ条件でビルドした。診断入力の範囲を事前確認し、全variantの出力と全3勾配を独立したFP64 oracleで検証した。検証除去を任意の入力へ適用できるという意味ではない。

## Encoder：添字計算とメタデータ検証

下表は新規プロセスで再測定した run3。profiler を無効にし、backend 順を反復ごとに変更した 7 反復、各 0.4 秒で比較した。数値は反復ごとの中央値の中央値、単位は µs。graph は Python dispatch をほぼ除いた GPU graph replay の時間で、kernel・cast・zero fill・GPU 内の実行間隔を含む。kernel 単体の時間ではない。[run3 記録](report-run3.json)

| encoder forward | FP32 wall | FP32 graph | FP16 wall | FP16 graph |
|---|---:|---:|---:|---:|
| native_control | 666.4 | 664.7 | 824.6 | 822.9 |
| baseline | 663.4 | 655.3 | 825.0 | 798.8 |
| int32 | 627.3 | 623.7 | 791.6 | 776.3 |
| no_metadata | 641.6 | 639.6 | 805.1 | 786.0 |
| int32_no_metadata | 606.9 | 606.1 | 773.0 | 750.4 |
| HF | 597.8 | 587.2 | 751.0 | 733.3 |

FP32 graph では int32 化だけで 31.6 µs（4.8%）、metadata 検証除去だけで 15.7 µs（2.4%）、両変更で 49.2 µs（7.5%）短縮した。int32 化の効果は metadata 検証があってもなくても現れ、metadata 検証除去の効果も両方の添字型で現れる。FP16 graph でもそれぞれ 2.8%、1.6%、6.1% 短縮した。通常の wall 時間でも、両変更で FP32 は 663.4 → 606.9 µs、FP16 は 825.0 → 773.0 µs と改善した。

先行する run2 の 5 反復でも、FP32 graph は baseline 664.0 → 両変更 613.4 µs、FP16 は 804.8 → 741.5 µs で、二要因の効果の方向は新規プロセスでも再現した。run3 の FP32 graph の反復中央値の範囲は baseline 651.2–675.2 µs、両変更 590.3–611.5 µs。一方、両変更後も HF との差は forward で FP32 約 18.9 µs、FP16 約 17.1 µs 残る。**二要因は主要因だが、すべての差を解消したわけではない**。[run2 記録](report-run2-corrected.json)

run2 の encoder FP32 wall は公開 API 655.0、native_control 656.3 µs で、公開 API を迂回しても差が残った。この結果と同一ビルド条件の二要因実験から、**encoder の主因は CUDA 内にある**。とくに graph では添字計算の変更の効果が大きい。相互作用や測定変動があるため、上の差をそのまま「HF に負ける原因の何割」という厳密な寄与率にはしない。

run3 の forward＋backward の FP32 graph も baseline 3059.8 → int32 2949.1 → 両変更 2911.0 µs、HF 2907.1 µs と差が縮まった。FP16 は baseline 3227.6 → 両変更 3116.3 µs、HF 3061.1 µs で、ここには約 55.2 µs の差が残った。

## Decoder：Python autograd経路

微小CUDA入力で同じカーネルを呼ぶ経路だけを比較すると、公開API／`fill_defaults`迂回／直接native呼び出しのwall中央値は219.4／186.8／131.5 µs、CUDA Graphは8.56／8.63／8.62 µsだった。Python profileでも `fill_defaults` と `redispatch` の追加処理を確認した。[host probe](host-probe.json)

これはhost側の追加コストの存在を示すが、微小入力の差をそのままdecoder全体の遅延率には換算できない。decoderのwall時間は反復間の変動が大きく、固定の改善率は結論にしていない。

## 根拠と限界

実測バイナリのFP32 forwardは、元の実装／int32化／検証除去／両変更／HFの順に、静的整数命令数が328／209／268／151／148だった。添字・stride・bilinear offsetをnarrowすると、生成コードと実測時間の両方が改善した。[バイナリ解析](binary_summary.json)

静的命令数は動的実行回数やstall時間ではない。local load/storeはなく、register spillを原因とする証拠はない。Nsight Computeは `ERR_NVGPUCTRPERM` でcounter取得に失敗したため、achieved occupancyの改善も断定しない。

run2のencoder profileにはGPU activityの欠落があり、時間の根拠から除外した。条件ごとに新規プロセスで再採取した全10 profileは数値検証と期待launch数の一致を確認済み。通常実行の速度比較にはrun3を使い、profile時間との直接比較・減算はしない。[完全なprofile](single-profiles.json)

[再現手順と保存データ](REPRODUCE.md)、[計測Podの削除確認](cleanup.json)。
