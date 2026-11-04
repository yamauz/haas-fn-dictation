# haas-fn-dictation

A push-to-talk dictation sample that runs the Japanese speech recognition model [Haas-ja-v1.0](https://huggingface.co/third-intelligence/Haas-ja-v1.0) locally on a Mac.

**Hold fn to record; release it and the transcript is pasted into whatever field has the focus.** No audio leaves your machine.

## Making it run on macOS

Officially, Haas supports only Linux with an NVIDIA GPU; CPU and macOS are not supported. The reason is the Kimi Delta Attention (KDA) layer in its prediction network, which depends on flash-linear-attention's Triton/CUDA kernels.

KDA, however, is a small part of the model (3.9M of 616M parameters), and the encoder is a standard FastConformer. [`kda_torch.py`](kda_torch.py) reimplements the KDA layer in pure PyTorch and swaps it into the class `haas_asr` uses.

- Parameter names and shapes are identical to fla 0.5.2, so the published checkpoint loads as is
- The math follows fla's `fused_recurrent_kda` kernel (qk L2 normalization, gating, sigmoid beta, value-first state)
- `flash-linear-attention` is excluded from installation through `override-dependencies` in `pyproject.toml`

Measured on an M2 Pro (7 seconds of audio): about 0.17 s on MPS, about 1.3 s on CPU.

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

The model (about 2.4 GB) is downloaded from Hugging Face on the first run.

### macOS settings

1. **Disable the default fn key action**: in System Settings > Keyboard, set "Press 🌐 key to" to "Do Nothing", so the emoji picker or the built-in dictation doesn't open
2. **Permissions**: grant the following to the terminal app that runs the script (Terminal, iTerm, VS Code, ...). A prompt appears on the first run; after granting, restart the terminal app
   - Privacy & Security > **Input Monitoring** (detecting the fn key)
   - Privacy & Security > **Accessibility** (pasting with ⌘V)
   - Privacy & Security > **Microphone** (recording)

## Usage

```bash
uv run python dictate.py
```

When it says it is ready (「準備完了」), put the cursor where you want the text, hold fn while you speak, then release.

- Presses shorter than 0.3 s are ignored
- Pressing another key while holding fn (fn + arrows, etc.) cancels the recording
- Recordings longer than 30 s are split at silences before recognition
- Pasting goes through the clipboard, and the previous clipboard contents are restored afterwards
- `--device cpu` switches to CPU inference

## Streaming while held (shift + fn)

While **shift + fn** is held, each phrase is transcribed and pasted as soon as you pause (0.6 s of silence). When you release, the last phrase is pasted and the mode ends. For longer dictation, text appears without waiting for the release.

- Silence means the volume (RMS) is below `SPEECH_RMS = 0.01`. If phrases don't split in a noisy room, or quiet speech gets cut off, adjust the constants in `dictate.py`
- A phrase is cut at 20 s even without a pause
- Adding shift after pressing fn also switches to this mode. If control is held too, command mode takes priority

## Opening apps by voice (fn + control)

Hold **fn + control** and say something like 「Slack」 or 「クローム開いて」 ("open Chrome") to open that app. Slack and Google Chrome are supported for now.

Which app to open is decided by [Laya](https://huggingface.co/convaiinnovations/laya), a local decision model compatible with TypeSafe Jev. It never generates text; it returns only a probability for each option of a fixed-choice question. The multilingual checkpoint is used (the English one cannot read Japanese), loaded with `device="mps"` set explicitly. About 650 MB is downloaded on the first run.

- The options are `slack` / `chrome` / `none`; nothing happens when `none` is chosen
- In command mode, transcription uses Haas's custom vocabulary (`launcher.VOCABULARY`). Without it, a short 「Slack」 can come out as 「スナック」 ("snack")
- To add apps, extend `APPS` and `VOCABULARY` in `launcher.py`
- `--no-launcher` skips loading Laya (faster startup)

### Why a dedicated key

The first attempt ran every dictation through Laya and also asked it whether the utterance was a command to open an app. The zero-shot multilingual model turned out to be poor at that (measured on an M2 Pro, about 20 ms per decision).

| Question design | Detecting commands | False positives on ordinary sentences |
|---|---|---|
| One choice question (slack / chrome / none) | 12/12 | 「Slackで連絡しておきます」 ("I'll reach out on Slack") → slack **0.99**, 「このバグを修正しました」 ("I fixed this bug") → chrome 0.81 |
| A yes/no question "is the user asking to open an app now?" | Missed: 「Slackを開いて」 ("open Slack") **0.10**, 「スラック」 ("Slack") 0.01 | Almost none |

The multilingual checkpoint isn't temperature-calibrated, so it reports 0.99 confidence even when wrong. So the user states "this is a command" with the key, and Laya is asked only which app. Mixing commands into normal dictation would need fine-tuning on labeled data plus calibration.

## Files

| File | Contents |
|---|---|
| [`dictate.py`](dictate.py) | fn key monitoring, recording, transcription, pasting, streaming mode, command mode |
| [`launcher.py`](launcher.py) | Loading Laya, deciding the app, launching it |
| [`kda_torch.py`](kda_torch.py) | Pure-PyTorch KDA layer and the code that swaps it in |

## Notes

- The microphone input stream stays open while the script runs (so the first syllable isn't lost). Audio is recorded only while fn is held, but the menu bar microphone indicator stays on
- On exit, `multiprocess` may print `AttributeError: '_thread.RLock' ...`. It comes from the shutdown of a NeMo dependency and is harmless
- The pure-PyTorch KDA is for inference only. For training, use the official environment with fla's chunk kernels
- Console messages are in Japanese
