# 検証拡大とトレース演習の仕様

実装前に固定した受入条件。既存の誤差基準は変更しない。

## 実モデル

- 固定モデル/参照revision、BF16重み→f32計算は従来と同じ。
- 異なる公開用の合成テキストから正確に32/128/512個のtoken IDを作る。
- 公式のchunk prefillで全位置の24層出力と全語彙logitsを保存する。
- 各prefillの後に8個の固定IDをteacher forcingで継続decodeし、同じ境界を保存。
- Rustは1/31/32/63/64/65/127/128/129/255/256/511/512の有効な位置で
  prefillを区切り、同じ状態を継続する。空prefillは状態を変えずNoneを返す。
- 境界後とdecode後の全DeltaNet再帰/畳み込み状態、全Attention KVも比較する。
- 全要素についてatol=3e-3、rtol=3e-3。絶対/相対誤差、cosine、top候補を報告。
  最初の逸脱位置・層・要素をエラーにする。失敗時に基準を広げない。
- 自然なgreedy生成を2プロンプトで行い、最大64トークン内に本物のEOSへ到達する。
  logitsを人為変更せず、予測分布も参照と照合する。`</think>`はEOSではない。
- 大きなfixtureはmodels/だけに保存。通常CIは生成/取得せず、opt-inで実行する。
- manifestは全ID、版、生成条件、誤差基準、ファイル長・SHA-256、生成器と参照ソースのhashを持つ。
- バイナリはlittle-endian f32、各位置の全層出力→logitsの順。
  1レコードずつ読み、全位置の活性値を同時保持しない。
- streaming prefill APIは最後のlogitsのみ返し、任意observerへ各stepを渡す。
  既存の全活性値保持APIは小規模観測用として残す。

## 学習教材

- 既存の小型4層fixtureのID [1,4,2] を使い、固定run IDの完全NDJSONを保存する。
- embedding→各層入力→norm→Mixer→残差→FFN→残差→最終norm→logitsを追跡できる。
- KVの増加、DeltaNet状態の固定サイズ、FFN unit 3の書込み列と除去差分を数値で示す。
- 予測してから実行する問題、jqでの抽出手順、数式付き解答、コードへの対応を提供。
- 教材の配列/解答と実行結果が整合することを通常のオフラインテストで確認。
- デモは未学習の合成重み。入力ID列を固定した観測と自己回帰生成を明確に区別する。

## 再生成・実行手順

既存の `tools/download-model.sh models/qwen` を明示実行済みの環境で行う。
新規依存は追加しない。比較用Python環境は従来のrequirements-reference.lockを使う。

```sh
.venv/bin/python tools/generate_extended.py --model-dir models/qwen
cargo test --release --offline --locked --test extended_real -- --ignored --nocapture --test-threads 1
```

参照生成・実モデルテスト・実モデルCLIを同時に起動しない。この環境のRAM上限は8GiB。
生成物は約0.9GB。通常CIには入れず、`models/qwen/extended/` に置く。
`validation-report.json` は小さい集約結果のみで、参照配列の全量は含まない。

各contextのバイナリは位置ごとに次の順で並ぶ（ヘッダなし）。

| 項目 | f32要素数 |
|---|---:|
| layer 0 output | 1024 |
| layer 1..23 output | 各1024 |
| vocabulary logits | 248320 |
| 合計 | 272896（1,091,584 bytes） |

全prefill位置のあと、8位置の継続decodeが続く。
生成ケースでは最後のprompt位置の予測から始め、生成ID数と同じレコード数を持つ。
EOSを予測した分布も比較するが、終了後にEOSを再入力するstepは作らない。

状態スナップショットは層0..23順。DeltaNetはconv→recurrent、AttentionはK→V。
convは[channel,kernel-1]、recurrentは[value_head,key_dim,value_dim]、
KVは[position,kv_head,head_dim]で最後の軸が連続する。
公式cacheが畳み込み履歴をkernel個保存していても、次stepに必要な最後のkernel-1個だけを保存してRustと比べる。

テキストは公開可能な固定の日本語・英語・混合文をtokenizeして指定長へ繰り返したもの。
ID列の全量と継続ID列はmanifestへ保存し、参照生成器とcheckpointのhashも保存する。
ランダムに切り出した利用者データや会話履歴は使わない。

この検証で比較する中間境界は全位置の層出力。既存の短い入力の統合テストは
正規化・Mixer・FFN・残差など7種類の演算境界を個別比較する役割を引き続き持つ。
長い入力で層が逸脱した場合は、その最初の位置・層をトレース対象にして内部演算を調べる。

## 実行結果

[集約した測定結果](validation-extended.json) に全層の最大誤差・cosine・候補順位を保存した。

| prefill長 | 継続decode | 比較した位置数 | logits最大絶対誤差 | 全層出力の最大絶対誤差 | top1一致 |
|---:|---:|---:|---:|---:|---:|
| 32 | 8 | 40 | 3.242493e-5 | 1.144409e-5 | 40/40 |
| 128 | 8 | 136 | 4.196167e-5 | 1.573563e-5 | 136/136 |
| 512 | 8 | 520 | 6.484985e-5 | 1.955032e-5 | 520/520 |

prefill/decode後の全層状態も合格。状態の最大絶対誤差は約1.086e-4。
自然なgreedy生成2例では6/9トークン目で `<|im_end|>` へ到達し、
その15個の予測分布も全層出力・全語彙logitsを比較した。
合計711位置、176,555,520個のlogits要素を照合した。
生成文字列の指示追従や一般的な品質を評価したものではない。

相対誤差は `abs(actual-reference)/max(abs(reference),1e-12)` をそのまま報告する。
参照値がゼロ付近では、絶対誤差が約1.97e-6でも相対値が約1.97e6になる例がある。
合否は事前定義した各要素の `abs(error) <= atol + rtol*abs(reference)`。
相対誤差だけを切り出して合否を決めず、絶対誤差と併せて読む。
許容値を変更したり、ゼロ付近の要素を比較から除外したりしていない。

Rust側tokenizerで3種類の長文コーパス、8個の継続ID、2種類のchat promptを再生成し、
参照manifestとID列が一致することも別のopt-inテストで確認した。

今回使った[参照manifest](extended-reference-manifest.json)とSHA-256も保存した。
全入力ID、各バイナリのバイト長/hash、生成条件をモデル本体なしでレビューできる。
通常CIはこの小さいmanifestの由来とレコード長を検査し、大きな配列は読み込まない。
