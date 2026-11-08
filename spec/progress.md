# Phase 0/1 実装記録

仕様 phase-01.md → tests/phase1.rs（未実装moduleで失敗を確認）→ 最小実装 → 公式クラスfixture照合。
traceも仕様 → tests/trace.rs（未実装moduleで失敗）→ 実装の順。
最初の実行では依存未取得も検出したが、その後依存を準備し、未実装による失敗を別途確認した。

実装: 形状付きMatrix・matmul/linear、要素積・残差加算、安定SiLU/softmax、
RMSNormの直接係数/zero-centered係数、FFN全中間値、単位無効化・書込み列表示、
opt-inローカルNDJSON（選択値・統計・全配列・バイト上限）。

参照: 固定公式ソースのMLP/RMSNorm本文をCPU PyTorch 2.6.0で実行。
fixture比較の最大絶対誤差7.450581e-9、最大相対誤差4.835257e-7。
許容誤差は事前定義atol=2e-6、rtol=2e-5のまま。手計算ケースも合格。

検証: fmt、clippy（警告をエラー化）、unit/fixture/traceテスト。
依存をvendorへ固定、実モデル・Python・ネットワークなしでテスト実行。
GitHub Actionsのcheckoutとツールチェーン準備にはネットワークが必要だが、
それ以降のRustのビルド・検証はoffline。実モデルテストは今回存在しない。

未対応: Attention/KV、Gated DeltaNet、層結線、logits、生成、BF16変換、
safetensors、tokenizer、実checkpointのtensor完全名・dtype/形状照合、方向除去overlay。
教育用f32単体演算の照合であり、実モデル全体の互換性確認ではない。
次フェーズ: 1ヘッド因果Attentionと独立KVキャッシュの仕様・fixture・全列対逐次比較。
