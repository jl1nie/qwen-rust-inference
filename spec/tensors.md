# 固定参照の対応表

revisionとhashはreferences.lock.json。checkpoint完全prefixは `model.language_model.`。
以下の相対名へprefixを付けてロードする。層iは0..23、Attentionはi%4==3のみ。
実checkpointの全言語tensor名・形状・BF16変換と、f32演算を統合テストで照合済み。
線形重みはすべてrow-major [out,in]。各位置のベクトルを1つずつ処理する。

| 相対名 | 実形状 | Rust対応 |
|---|---|---|
| embed_tokens.weight | [248320,1024] | model.embedding（入力と出力で共有） |
| layers.i.mlp.gate_proj.weight | [3584,1024] | ffn.gate |
| layers.i.mlp.up_proj.weight | [3584,1024] | ffn.up |
| layers.i.mlp.down_proj.weight | [1024,3584] | ffn.down、writing_vector(j)=列j |
| layers.i.input_layernorm.weight | [1024] | block.input_norm（zero-centered） |
| layers.i.post_attention_layernorm.weight | [1024] | block.post_norm（zero-centered） |
| norm.weight | [1024] | model.norm（zero-centered） |
| layers.i.self_attn.q_proj.weight | [4096,1024] | attention.q、ヘッドごとの[Q,gate] |
| layers.i.self_attn.k_proj.weight / v_proj.weight | 各[512,1024] | attention.k/v |
| layers.i.self_attn.o_proj.weight | [1024,2048] | attention.o |
| layers.i.self_attn.q_norm.weight / k_norm.weight | 各[256] | attention.q_norm/k_norm（zero-centered） |
| layers.i.linear_attn.in_proj_qkv.weight | [6144,1024] | delta.qkv、全Q→全K→全V |
| layers.i.linear_attn.in_proj_z.weight | [2048,1024] | delta.z |
| layers.i.linear_attn.in_proj_a.weight / in_proj_b.weight | 各[16,1024] | delta.a/b |
| layers.i.linear_attn.conv1d.weight | [6144,1,4] | delta.conv、内部は[6144,4] |
| layers.i.linear_attn.dt_bias / A_log | 各[16] | delta.dt/a_log |
| layers.i.linear_attn.norm.weight | [128] | delta.norm（直接係数＋SiLU(z)） |
| layers.i.linear_attn.out_proj.weight | [1024,2048] | delta.out |

言語出力lm_headの独立tensorはこのcheckpointには存在しない（tie_word_embeddings=true）。
vision/MTPはignored-tensors.jsonの完全名リストでのみ除外。未知の未使用tensorはエラー。
MTP（`mtp.*`）は `load_model` では読まず、採用率の計測用に `load_mtp` が同じファイルを別に読む（[ADR 0022](adr-0022.md)、`src/mtp.rs`）。テキストモデルとidentityは変わらない。
checkpoint自体のSHA-256も固定し、別revisionの名前の偶然の一致では読み込まない。

## 状態と中間値

| 対象 | 格納 |
|---|---|
| Attention KV | 層ごと、位置列の各要素に[KV head,head_dim]をflatten |
| Delta再帰 | [value_heads,key_dim,value_dim]、最後の軸が連続 |
| Delta conv履歴 | [channels,kernel-1]、各チャンネルで古い順 |
| Q/K/V | ヘッド番号→各head成分の順にflatten |
| Attention scores/probabilities | [Q heads,過去を含む現在までの位置数] |
| FFN gate/up/product | [intermediate_size] |
| 残差・norm | [hidden_size] |
| logits | [vocab_size] |

小型fixtureは幅8、中間幅16、4層。全重みと中間値・各step状態を
`tests/fixtures/hybrid.json`に保存。公式生成器と元ソースのhashも検証する。

演算順序はphase-02-06.md、手計算と意味はlearning.mdを参照。
