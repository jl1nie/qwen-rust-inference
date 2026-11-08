# 固定参照の対応表

Revisionとファイルhashは references.lock.json。層番号 i は0..23。
重みはモデルconfigのBF16、教育用と今回の参照計算はf32。BF16変換・丸めは未検証。
以下は固定configと参照クラスの定義から導出した形状。実checkpointのロード検証はPhase 5。
名称はdecoder相対名（checkpointの完全prefix照合はPhase 5で行う）。線形重みはすべて[out,in]。

| 重み／演算 | 実モデル形状 | 参照クラス／現在の対応 |
|---|---|---|
| embed_tokens.weight | [248320,1024] | Qwen3_5TextModel／未対応 |
| layers.i.mlp.gate_proj.weight | [3584,1024] | Qwen3_5MLP／ffn::FfnWeights gate |
| layers.i.mlp.up_proj.weight | [3584,1024] | Qwen3_5MLP／ffn::FfnWeights up |
| layers.i.mlp.down_proj.weight | [1024,3584] | Qwen3_5MLP／ffn::FfnWeights down |
| layers.i.input_layernorm.weight | [1024] | Qwen3_5RMSNorm／ops::rms_norm QwenZeroCentered |
| layers.i.post_attention_layernorm.weight | [1024] | 同上 |
| norm.weight | [1024] | 同上（結線は未対応） |
| lm_head.weight（共有埋め込み） | [248320,1024] | 未対応。別tensorの存在を前提にしない |
| layers.i.self_attn.q_proj.weight | [4096,1024] | Qと出力ゲートを一緒に射影／Phase 2 |
| layers.i.self_attn.k_proj.weight, v_proj.weight | 各[512,1024] | Phase 2 |
| layers.i.self_attn.o_proj.weight | [1024,2048] | Phase 2 |
| layers.i.self_attn.q_norm.weight, k_norm.weight | 各[256] | Qwen3_5RMSNorm／単体演算のみ対応 |
| layers.i.linear_attn.in_proj_qkv.weight | [6144,1024] | Qwen3_5GatedDeltaNet／Phase 3 |
| layers.i.linear_attn.in_proj_z.weight | [2048,1024] | 同上 |
| layers.i.linear_attn.in_proj_a.weight, in_proj_b.weight | 各[16,1024] | 同上 |
| layers.i.linear_attn.conv1d.weight | [6144,1,4] | causal depthwise conv／未対応 |
| layers.i.linear_attn.dt_bias, A_log | 各[16] | 未対応 |
| layers.i.linear_attn.norm.weight | [128] | RMSNormGatedの直接係数＋SiLU出力ゲート／未対応 |
| layers.i.linear_attn.out_proj.weight | [1024,2048] | Phase 3 |

Full Attentionはi%4==3のみ、DeltaNetはそれ以外。configの明示layer_typesをテストで確認。
FFN: gate/up射影 → gateのSiLU → 要素積 → down射影。biasなし。
RMSNorm: f32へ → 平方平均 → eps加算 → rsqrt → xに乗算 → (1+weight)乗算 → 入力dtypeへ。
層の順序: input norm → token mixer → residual add → post-attention norm → FFN → residual add。
この層結線自体、RoPE、GQA、ゲートや再帰の順序は今後の受入テスト対象で、今回の互換性保証外。

教育用幅8/中間幅16の対応する配列はtests/fixtures/phase1.jsonにすべて保存。
生成器は公式クラスの実際の本文を実行し、中間のgate/up/SiLU/productも個別保存する。
実モデルのvision/MTP等の未使用tensor許可リストは、ロード実装時に完全名を照合して作成する。
