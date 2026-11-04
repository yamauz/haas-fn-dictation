# haas-fn-dictation

日本語音声認識モデル [Haas-ja-v1.0](https://huggingface.co/third-intelligence/Haas-ja-v1.0) を Mac のローカルで動かす、プッシュトゥトーク入力のサンプル。

**fn キーを押している間だけ録音し、離すと文字起こしして、カーソルのあるフィールドに貼り付ける。** 音声は外部に送信しない。

## Mac で動かすための工夫

公式には Haas は「Linux + NVIDIA GPU のみ対応、CPU と macOS は非対応」となっている。予測ネットワークの Kimi Delta Attention (KDA) 層が flash-linear-attention の Triton/CUDA カーネルに依存しているため。

ただ KDA はモデル全体 616M パラメータのうち 3.9M の小さな部分で、エンコーダ本体は普通の FastConformer。そこで [`kda_torch.py`](kda_torch.py) で KDA 層を純 PyTorch で書き直し、`haas_asr` が使うクラスを差し替えている。

- パラメータ名・形状は fla 0.5.2 と同一なので、公開チェックポイントをそのまま読み込める
- 計算は fla の `fused_recurrent_kda` カーネルと同じ式(qk の L2 正規化、ゲート、beta の sigmoid、value-first の状態)
- `flash-linear-attention` は `pyproject.toml` の `override-dependencies` でインストール対象から外している

M2 Pro での実測(7 秒の音声):MPS で約 0.17 秒、CPU で約 1.3 秒。

## セットアップ

[uv](https://docs.astral.sh/uv/) が必要。

```bash
uv sync
```

初回起動時にモデル(約 2.4GB)を Hugging Face からダウンロードする。

### macOS の設定

1. **fn キーの既定動作を止める**:システム設定 > キーボード >「🌐 キーを押して」を「何もしない」にする。絵文字ピッカーや標準の音声入力が開くのを防ぐため
2. **権限**:スクリプトを実行するターミナルアプリ(Terminal / iTerm / VS Code など)に以下を許可する。初回起動時にダイアログが出るので、許可したらターミナルアプリを再起動する
   - プライバシーとセキュリティ > **入力監視**(fn キーの検知)
   - プライバシーとセキュリティ > **アクセシビリティ**(⌘V での貼り付け)
   - プライバシーとセキュリティ > **マイク**(録音)

## 使い方

```bash
uv run python dictate.py
```

「準備完了」と出たら、テキストを入力したい場所にカーソルを置き、fn を押しながら話して離す。

- 0.3 秒未満の押下は無視する
- fn を押したまま他のキーを押した場合(fn+矢印など)は取り消す
- 30 秒を超える録音は無音区間で分割して認識する
- 貼り付けはクリップボード経由で、貼り付け後に元のクリップボードの内容を戻す
- `--device cpu` で CPU 実行に切り替えられる

## 押している間ずっと入力する(shift + fn)

**shift + fn** を押している間は、話の区切り(0.6 秒の無音)ごとに文字起こしして、その場で貼り付けていく。離すと、最後の区切りの分を貼り付けて終わる。長めに話すときに、離すまで待たずに文字が出てくる。

- 無音の判定は音量(RMS)が `SPEECH_RMS = 0.01` 未満かどうか。周りがうるさくて区切れない、または小声で途切れる場合は `dictate.py` の定数を調整する
- 無音がなくても 20 秒で区切る
- fn を先に押してから shift を足しても切り替わる。control と同時に押した場合はコマンドモードが優先

## 音声でアプリを開く(fn + control)

**fn + control** を押しながら「Slack」「クローム開いて」などと話すと、そのアプリを開く。今のところ対応しているのは Slack と Google Chrome。

どのアプリを開くかは [Laya](https://huggingface.co/convaiinnovations/laya) で判定している。Laya は TypeSafe Jev 互換のローカル判定モデルで、文章を生成せず、決められた選択肢それぞれの確率だけを返す。読み込み方は fx-chart の `tools/system_one.py` と同じで、multilingual チェックポイント(英語版は日本語を読めない)を `device="mps"` で読む。初回は約 650MB をダウンロードする。

- 選択肢は `slack` / `chrome` / `none`。`none` が選ばれたら何もしない
- コマンドモードの文字起こしでは Haas のカスタム語彙(`launcher.VOCABULARY`)を有効にする。これがないと短い「Slack」が「スナック」になることがある
- アプリを増やすときは `launcher.py` の `APPS` と `VOCABULARY` に足す
- `--no-launcher` を付けると Laya を読み込まない(起動が速くなる)

### なぜ専用のキーにしているか

最初は、普段の音声入力をすべて Laya に通して「アプリ起動の命令か」も判定させようとした。でもゼロショットの multilingual 版はこの判定が苦手だった(M2 Pro で実測、1 判定あたり約 20ms)。

| 質問の作り方 | 命令の検出 | 普通の文の誤検出 |
|---|---|---|
| choice 1 問(slack / chrome / none) | 12/12 | 「Slackで連絡しておきます」→ slack **0.99**、「このバグを修正しました」→ chrome 0.81 |
| noul「今アプリを開けと言っているか」 | 「Slackを開いて」**0.10**、「スラック」0.01 で取りこぼし | ほぼなし |

fx-chart の `docs/laya-local.md` にある通り multilingual 版は温度較正がされておらず、間違えていても確信度が 0.99 になる。なので「命令かどうか」はユーザーにキーで明示してもらい、Laya には「どのアプリか」だけを聞いている。普段の音声入力にも混ぜたいなら、ラベル付きデータでのファインチューニングと温度較正が要る。

## ファイル

| ファイル | 内容 |
|---|---|
| [`dictate.py`](dictate.py) | fn キー監視・録音・文字起こし・貼り付け・垂れ流しモード・コマンドモード |
| [`launcher.py`](launcher.py) | Laya の読み込み、アプリ判定、アプリ起動 |
| [`kda_torch.py`](kda_torch.py) | KDA 層の純 PyTorch 実装と差し替え処理 |

## 補足

- マイクの入力ストリームは起動中ずっと開いている(話し始めの取りこぼしを防ぐため)。録音するのは fn を押している間だけだが、メニューバーのマイクインジケーターは点灯したままになる
- 終了時に `multiprocess` の `AttributeError: '_thread.RLock' ...` が出ることがあるが、NeMo の依存パッケージの終了処理によるもので動作には影響しない
- 純 PyTorch 版 KDA は推論専用。学習には fla の chunk カーネルを使う公式環境を使うこと
