# 実用性改善の受入仕様

## 計測

load/hash/変換時間、prefill、decode、生成選択を分ける。固定ID/固定回数、release、
バックエンド、CPU/quota、論理重み/状態bytesとLinux RSS/peak RSSをJSONへ記録。
ウォームアップと測定を分離。ピークRSSはプロセス生涯値で、フェーズ固有と呼ばない。

行列×単一ベクトル専用の8累積器ループを候補として追加し、既存のスカラー/sgemmを残す。
参照fixture・全モデル/状態の比較が既存基準(atol=3e-3,rtol=3e-3)を満たし、
同じ入力の実測で改善した場合だけ通常推論へ採用する。教育用Full traceは維持する。
compact stepは層ごとに活性値を解放し、Delta状態の観測用コピーを省く。演算自体は同じ。

## 連続推論

tokensは出力済みを含むID列、state.positionはモデルへ入力済みの個数。
選ばれた最後のトークンはpendingとして最大1個保持する。
`tokens.len == state.position + pending.is_some`。
最後のlogitsとsamplingの現在のu64乱数状態を保存し、再開時にseedへ戻さない。
EOS/final tokenを無駄にもう一度計算せず、次の継続時にpendingをちょうど一度消費する。

## 複数ターン

text-only system(先頭任意)/user/assistant交互。公式固定Jinjaの分岐を再現する。
過去assistantの</think>以前は次のuser queryのテンプレートから省かれる。
従って文字列/IDの接頭辞が一致する時だけcacheを使う。
各user末尾のanchorを1つ保持し、次ターンではanchorへ戻って旧assistantの正規化済み内容と
新しいuserをprefillする。anchorより前の履歴は再計算しない。一致しなければ全prefill。
MaxTokens後は/continue、または明示的な次user入力でターンを閉じる。EOS後の/continueは拒否。
新ターン失敗時は元の会話を保持する。RNGはcache replayのために巻き戻さない。

## 保存/復元

明示opt-in、ローカルファイルのみ。version/model identity/revision/backend/sampling、
live+anchorの状態、tokens、pending、logits、RNG、会話と直近応答IDを保存。
モデルidentityはconfigと実際の言語重みのf32 bit列に結び付ける。
独自little-endian f32ストリーム + サイズ制限付きJSON header + SHA-256。
shape、有限値、位置、ID、payload長、version/model不一致、改変、切断、余剰bytesを検査。
既存ファイルを上書きせず、一時ファイルを閉じてからatomic publishする。
デフォルト上限256MiB、header上限8MiB。会話を含むため保存はデフォルト無効。

## 検証

小型モデル: compact/Full/各backend同値、pending有無・seeded samplingのsave/resume、
EOS/最大数/empty/境界、chat cache対全履歴prefill、snapshotの不正入力/別モデル拒否。
公式Jinja fixture: system・2/3turn・thinking・過去reasoningを含むレンダリング/ID列。
実モデル: short全演算、32/128/512の既存参照を変更せず比較、2turn・途中保存/再開、
実測baselineと候補を記録。大きいモデルを複数同時にロードしない。

## CLIとAPI

```sh
cargo run --release --offline --locked -- chat --model-dir models/qwen
# you> の入力。/continue /reset /save PATH /load PATH /quit を使用可能。

cargo run --release --offline --locked -- chat --model-dir models/qwen --prompt '数字だけで答えてください。1+1は？' --max-new-tokens 1 --save-state /tmp/partial.qrs
cargo run --release --offline --locked -- chat --model-dir models/qwen --load-state /tmp/partial.qrs --continue --max-new-tokens 16 --save-state /tmp/complete.qrs
cargo run --release --offline --locked -- chat --model-dir models/qwen --load-state /tmp/complete.qrs --prompt 'その数に1を足すと？数字だけ答えてください。' --max-new-tokens 16

cargo run --release --offline --locked -- bench --model-dir models/qwen --backend vector --ids 16,10,16,28 --decode-steps 8 --warmup 1 --runs 2 > /tmp/bench.json
```

chatは初回にモデルを1回だけロードし、複数userターンを同じプロセスで扱う。
`--system TEXT`、`--thinking`、`--seed N --temperature 0.8 --top-k 40`、
`--backend scalar|gemm|vector` は新規会話で指定する。ロード時は保存した設定を使い、
これらの上書きを拒否する。`--json` は1応答1JSON、`--stats` はcache/prefill数と時間、
`--no-cache` は全履歴replayでの比較用。保存した設定は/resetでも引き継ぐ（履歴とRNGは新runへ戻す）。

上記例では、最初の出力は「1」、再開で「1+1=2」、次ターンで「2+1=3」になった。
数値とcacheの詳細は[CLI観察](measurements/chat-cli.json)、
[実モデル保存/復元・replay比較](measurements/chat-resume.json) を参照。
実モデルの応答途中snapshotは約43.35MB。次ターンは16位置を再利用し、
prefillを52→36位置に減らした。cache/replayの生成IDと全状態が一致した。

### 保存の扱い

保存は `--save-state` または `/save` を実行した時だけ行う。会話内容・ID・状態を含む。
ローカルファイルのみで暗号化はしない。SHA-256は破損検出であり、認証ではない。
Unixでは作成ファイルを0600にする。既存ファイルを上書きせず、同じfilesystem上の
一時ファイルをflush/fsyncしてからhard linkで公開する。失敗時の一時ファイルは除去する。

デフォルトの総ファイル上限は256MiB、JSON headerは8MiB。
`--state-max-bytes N` で総上限を明示変更できるが、最大文脈での実用性を保証するものではない。
weights/tokenizerはsnapshotに含まない。別プロセスでの再開には同じmodel-dirを指定し、
重みの読込み/検証時間も再び必要。会話のprefixを再計算する時間を減らす機能である。

pendingは出力済みだがまだモデルに入れていない最後のID。
新しい生成/入力の前に一度だけ消費する。RNGを初期seedへ戻してはいけない。
`InferenceSession` と `snapshot::save_session/load_session` が生のtoken ID経路、
`ChatSession` と `save_chat/load_chat` が会話とanchorを含む経路。
observer失敗は完成したstepを保持するが、演算途中の失敗はsessionを無効化し、resetまで保存/継続を拒否する。
Modelはconfig()/embedding()/blocks()/norm()/revision()/identity()/unused()の読み取り専用APIを使う。

### 制限

system(先頭任意)・user/assistant交互のtext-only会話を対象とする。
ツール・マルチモーダル・自動履歴切詰め・モデル重みoverlay付き会話snapshotは未対応。
履歴が文脈上限を超えた新turnは失敗し、元の会話を保持する。/resetで新runにできる。
EOS後は/continueを拒否。MaxTokens後は/continueで同じ回答を伸ばすか、
次のuser入力でその回答を区切って会話を進める。
`--json`/通常表示の/continueは累積した応答全文を返し、UTF-8 token境界で文字を落とさない。

保存状態の再現性は同じモデル・backend・設定・実装で検証している。
別CPU/別カーネルでのbit一致や、将来の算術変更との互換性は仮定しない。
