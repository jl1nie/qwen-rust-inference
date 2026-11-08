# 実装・検証記録

## Phase 0・1

仕様→演算テスト（未実装moduleで失敗）→実装→手計算/固定公式ソース比較を実施。
Matrix・残差・SiLU・softmax・RMSNorm・FFN・単位無効化・NDJSONを実装。
Phase 1 fixtureの最大絶対誤差7.450581e-9、許容値atol=2e-6/rtol=2e-5を維持。

## Phase 2・3

phase-02-06.mdで数式・形状・誤差を先に固定。
未実装moduleによるhybridテストの失敗を確認してから各モジュールを追加した。
Attention: 位置回転、zero-centered Q/K norm、GQA、出力ゲート、因果dense/逐次KV。
DeltaNet: causal depthwise conv、L2 QK、減衰/更新ゲート、逐次rank-1更新、gated RMSNorm。
公式本文から生成した小型fixtureの全stepについて、各層のconv状態と再帰状態を照合。
単一step、境界ゲート、非正方状態、reset、固定状態サイズ、因果性を確認。

## Phase 4

埋め込み→DeltaNet×3＋Attention×1→最終norm→共有埋め込みの語彙射影を実装。
小型モデルの全層/全logitsと公式chunk prefill、全prefill分割位置を比較。
Backend::Reference/Optimizedを比較し、別runへ状態が漏れないことを確認。
生成制御テストはdecode_withの未実装で失敗を確認してから実装。
制御したlogitsで</think>相当IDを通過し、EOS・最大数・固定seedを確認。

## Phase 5

固定checkpoint約1.75GBを明示スクリプトで取得し、hashを検証。
BF16→f32、言語tensorの完全名・形状、直接係数/zero-centered係数、共有埋め込みを照合。
vision/MTPの未使用完全名をspec/ignored-tensors.jsonに保存し、予期しない未使用を拒否。
3入力・9位置、全24層の7境界とlogits、計1,521件を数値比較した。
全比較の最大絶対誤差6.29425048828125e-5、logits最大3.528594970703125e-5。
演算境界の最小cosine約0.9999999999787、全位置のgreedy最上位候補が一致。
事前基準atol=3e-3/rtol=3e-3は変更していない。
詳細はvalidation-real.json。参照fixture本体はmodels/に置き、Gitには保存しない。

実モデルのchunk prefill対逐次decode、split prefillも合格。
日本語・英語・空白/改行/絵文字のtokenizerとtext-only JinjaテンプレートID列が一致。
初回の終了検証でconfig EOS=248044とchat EOS=248046の違いを検出し修正。
</think>=248069は終了扱いにしない。
Rust CLIで「1+1は？」から8トークンの出力「1+1 は **2** です」を確認。

参照fixture初回生成は8GiB制限でOOM。module重みの二重コピーをassign=Trueで避け、
実モデルの各step再帰状態全量のPython list化をやめて解消した。
小型fixtureの再帰状態検証は省略していない。

## Phase 6

ユニット無効化・down列の方向成分除去を別manifest overlayで実装。
モデルrevision/層/完全名/列/有限方向/係数を検証。ゼロ方向や不一致を拒否。
FFN差分の手計算、residual/logits差、無編集への復帰、manifest往復をテスト。
CLIは編集前後の各層FFN・残差・logits差と生成列の反復指標を表示する。
複数方向の修正は元列に対する加算規約。ユニット無効化を先に適用する。
研究記事のループ抑制率や性能向上は再現していない。別の評価予算・プロトコルが必要。

## 通常CIと制限

fmt・clippy -D warnings・unit/fixture/traceテストをオフラインで実行。
実モデル統合は#[ignore]で明示opt-in。Rustテスト/CLIからPythonは呼ばない。
Actionsのcheckout/ツールチェーン準備以外にネットワークは不要。

対応外: GPU/量子化/バッチ/画像動画/MTP実行/学習/分散/投機的decode、
最大文脈長検証、別revision/別dtype/別RoPE、複数turnのchat CLI、研究性能再現。
Python参照は固定公式ソース本文と小さいCacheアダプタであり、Transformers全体の
from_pretrained実行・任意高速カーネル・ネイティブBF16との一致を主張しない。

最終CLI観察: 実重みのユニット無効化と第0座標方向除去を実行し、
FFN/残差/logitsの非ゼロ差を記録（validation-edits.txt）。生成列はこの短い
例ではbaselineと同じだった。これを品質改善の根拠にはしない。
通常オフラインテストは20件。実モデル統合1件は別途実行。
この環境は8GiB制限なので、実モデルCLIと実モデル統合テストは直列で実行する。

## 検証拡大と学習教材（追加作業）

仕様をextended-validation.mdに固定し、未実装のprefill_observed/lessonによる
テスト失敗を確認してから実装した。新たな依存や自動ダウンロードは追加していない。

実モデルの32/128/512位置を公式chunk prefillと比較し、各8位置の逐次継続も検証。
計696位置の全24層出力・全語彙logitsと、prefill/decode後の全KV/conv/recurrent状態が合格。
2例の自然なgreedy生成では、6/9トークン目にEOSへ到達。15予測位置も数値比較し、
全体で711位置、176,555,520 logits要素を照合した。
logits最大絶対誤差6.4849853515625e-5、状態最大絶対誤差約1.086e-4。
既存のatol=3e-3/rtol=3e-3を維持。結果はvalidation-extended.json。
Rust tokenizerの新コーパス/継続/生成promptの一致も検証済み。

参照値はlittle-endian f32へ書き出し、manifestの版・hash・バイト長を検査して
1位置ずつ読む。全量は約0.9GBでmodels/にのみ保存。
Model::prefill_observedはobserver後に活性値を解放し最後のlogitsだけ返す。
空区間、分割、直接stepとの一致、observer失敗後の状態/resetをテスト。
比較器は成功した全要素のエラー文字列を作らないよう変更したが、判定式と数値計算順は不変。
個別要素の閾値・境界値・最初の失敗位置を追加テストで確認した。

教材はdemo lesson、330行の完全トレース、数値解答、8問の演習と別冊解答を追加。
保存配列から残差・Attention・FFN・DeltaNet・最終語彙射影をf64で独立再計算する。
入力ID固定の観測と自己回帰生成の違い、状態のサイズ、形状/列/ゲートの取り違えを扱う。
通常オフラインテストは30件。長い実モデル検証とtokenizer検証は別途opt-inで実行し合格。
