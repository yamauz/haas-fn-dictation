"""Push-to-talk dictation for macOS with Haas-ja-v1.0.

Hold fn to record, release it to transcribe, and the text is pasted into
whatever field has the focus. Runs fully locally.

    uv run python dictate.py
"""

from __future__ import annotations

import argparse
import logging
import queue
import signal
import threading
import time
import warnings

import numpy as np
import Quartz
import sounddevice as sd
from AppKit import NSPasteboard, NSPasteboardItem, NSSound

SAMPLE_RATE = 16_000
MODEL_ID = "third-intelligence/Haas-ja-v1.0"
FN_KEYCODE = 63  # kVK_Function
V_KEYCODE = 9  # kVK_ANSI_V
MIN_SECONDS = 0.3  # shorter presses are treated as accidental
LONGFORM_SECONDS = 30  # the model handles up to about 30 s in one pass
# Streaming mode (shift + fn) cuts the audio at pauses and pastes each phrase.
SPEECH_RMS = 0.01  # blocks louder than this count as speech
PAUSE_SECONDS = 0.6  # this much silence after speech ends a phrase
LEAD_SECONDS = 0.3  # silence kept before a phrase so its first syllable isn't clipped
MAX_PHRASE_SECONDS = 20  # cut even without a pause, well under LONGFORM_SECONDS

log = logging.getLogger("dictate")


# --- model -----------------------------------------------------------------


def load_model(device: str):
    # NeMo is chatty at import and load time.
    warnings.filterwarnings("ignore")
    import haas_asr
    import torch
    from nemo.utils import logging as nemo_logging

    import kda_torch

    nemo_logging.setLevel(logging.ERROR)
    kda_torch.install()
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    log.info("モデルを読み込み中 (%s)... 初回はダウンロードに時間がかかります", device)
    model = haas_asr.load_model(MODEL_ID, device=device)
    # The first pass on MPS compiles kernels; do it now, not on the first dictation.
    model.transcribe([np.zeros(SAMPLE_RATE, dtype=np.float32)], verbose=False)
    return model


def transcribe(model, audio: np.ndarray, vocabulary=None) -> str:
    import haas_asr

    if len(audio) > LONGFORM_SECONDS * SAMPLE_RATE:
        return haas_asr.transcribe_longform(model, audio, vocabulary=vocabulary).strip()
    return haas_asr.transcribe(model, [audio], vocabulary=vocabulary)[0]


# --- output ----------------------------------------------------------------


def paste_text(text: str) -> None:
    """Paste `text` into the focused field via the clipboard, then restore the clipboard.

    Typing it with synthetic key events would go through the active input method
    and mangle Japanese, so the clipboard is the reliable route.
    """
    pasteboard = NSPasteboard.generalPasteboard()
    saved = [
        {t: item.dataForType_(t) for t in item.types()}
        for item in (pasteboard.pasteboardItems() or [])
    ]

    pasteboard.clearContents()
    pasteboard.setString_forType_(text, "public.utf8-plain-text")
    for down in (True, False):
        event = Quartz.CGEventCreateKeyboardEvent(None, V_KEYCODE, down)
        Quartz.CGEventSetFlags(event, Quartz.kCGEventFlagMaskCommand)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)

    # Give the target app time to read the clipboard before putting the old one back.
    time.sleep(0.3)
    pasteboard.clearContents()
    items = []
    for types in saved:
        item = NSPasteboardItem.alloc().init()
        for t, data in types.items():
            if data is not None:
                item.setData_forType_(data, t)
        items.append(item)
    if items:
        pasteboard.writeObjects_(items)


def play(name: str) -> None:
    sound = NSSound.soundNamed_(name)
    if sound is not None:
        sound.play()


# --- recording + fn key ----------------------------------------------------


class Segmenter:
    """Cuts live audio at pauses, so each phrase is transcribed while recording goes on."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.chunks: list[np.ndarray] = []
        self.length = 0  # samples in self.chunks
        self.silence = 0  # trailing silent samples
        self.voiced = False

    def feed(self, block: np.ndarray) -> np.ndarray | None:
        """Add a block; returns a finished phrase when this block completes one."""
        self.chunks.append(block)
        self.length += len(block)
        if np.sqrt(np.mean(block**2)) >= SPEECH_RMS:
            self.voiced = True
            self.silence = 0
        else:
            self.silence += len(block)
        if not self.voiced:
            # Only keep a short lead-in of silence before speech starts.
            while self.length - len(self.chunks[0]) >= LEAD_SECONDS * SAMPLE_RATE:
                self.length -= len(self.chunks.pop(0))
            return None
        if self.silence >= PAUSE_SECONDS * SAMPLE_RATE or self.length >= MAX_PHRASE_SECONDS * SAMPLE_RATE:
            return self.flush()
        return None

    def flush(self) -> np.ndarray | None:
        """The audio collected so far if it contains speech; starts a new phrase either way."""
        phrase = np.concatenate(self.chunks) if self.voiced else None
        self.reset()
        return phrase


class Dictation:
    def __init__(self, model, laya=None):
        self.model = model
        self.laya = laya
        self.vocabulary = None
        if laya is not None:
            import haas_asr

            import launcher

            self.vocabulary = haas_asr.Vocabulary(launcher.VOCABULARY)
        self.recording = False
        self.cancelled = False
        self.command = False  # control held at some point while recording
        self.streaming = False  # shift + fn: paste phrase by phrase while held
        self.segmenter = Segmenter()
        self.started_at = 0.0
        self.chunks: list[np.ndarray] = []
        self.lock = threading.Lock()
        self.jobs: queue.Queue[tuple[np.ndarray, bool]] = queue.Queue()
        # The stream stays open so the first syllable isn't lost while the device starts.
        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="float32", callback=self._on_audio
        )

    def _on_audio(self, indata, frames, time_info, status):
        with self.lock:
            if not self.recording:
                return
            block = indata[:, 0].copy()
            if not self.streaming:
                self.chunks.append(block)
            elif (phrase := self.segmenter.feed(block)) is not None:
                self.jobs.put((phrase, False))

    def start(self, command: bool, streaming: bool) -> None:
        with self.lock:
            self.recording = True
            self.cancelled = False
            self.command = command
            self.streaming = streaming and not command
            self.chunks = []
            self.segmenter.reset()
            self.started_at = time.monotonic()
        play("Tink")
        mode = " (コマンドモード)" if self.command else " (垂れ流しモード)" if self.streaming else ""
        log.info("🎙  録音中%s...", mode)

    def start_streaming(self) -> None:
        """Switch a recording that has already started to streaming mode."""
        with self.lock:
            self.streaming = True
            for block in self.chunks:
                if (phrase := self.segmenter.feed(block)) is not None:
                    self.jobs.put((phrase, False))
            self.chunks = []
        log.info("   → 垂れ流しモードに切り替えました")

    def stop(self) -> None:
        with self.lock:
            self.recording = False
            chunks, self.chunks = self.chunks, []
            held = time.monotonic() - self.started_at
            rest = self.segmenter.flush() if self.streaming else None
        if self.streaming:
            # Earlier phrases are already pasted; only the unfinished one is left.
            if rest is not None and held >= MIN_SECONDS:
                self.jobs.put((rest, False))
            play("Pop")
            return
        if self.cancelled:
            log.info("fn と他のキーが同時に押されたので取り消しました")
            return
        if held < MIN_SECONDS or not chunks:
            log.info("短すぎるので無視しました")
            return
        play("Pop")
        self.jobs.put((np.concatenate(chunks), self.command))

    def worker(self) -> None:
        """Transcribes off the event-tap thread, which macOS disables if it blocks."""
        while True:
            audio, command = self.jobs.get()
            try:
                self.handle(audio, command and self.laya is not None)
            except Exception:
                # Keep the worker alive: one failed utterance must not stop the rest.
                log.exception("処理中にエラーが起きました")

    def handle(self, audio: np.ndarray, command: bool) -> None:
        started = time.monotonic()
        text = transcribe(self.model, audio, self.vocabulary if command else None)
        elapsed = time.monotonic() - started
        if not text:
            log.info("(認識結果なし)")
            return
        log.info("📝 %s  (%.1f 秒の音声 / %.2f 秒で認識)", text, len(audio) / SAMPLE_RATE, elapsed)
        if command:
            self.run_command(text)
        else:
            paste_text(text)

    def run_command(self, text: str) -> None:
        # Only in command mode: zero-shot Laya is confidently wrong on ordinary
        # sentences that mention an app ("Slackで連絡します" -> slack 0.99), so it
        # never decides whether something is a command, only which app.
        import launcher

        choice, _, probs = launcher.decide(self.laya, text)
        detail = ", ".join(f"{k} {v:.2f}" for k, v in probs.items())
        if choice == launcher.NONE:
            log.info("🤷 開くアプリが分かりませんでした (%s)", detail)
            play("Basso")
            return
        log.info("🚀 %s を開きます (%s)", launcher.APPS[choice][0], detail)
        launcher.open_app(choice)

    def on_event(self, proxy, event_type, event, refcon):
        if event_type in (
            Quartz.kCGEventTapDisabledByTimeout,
            Quartz.kCGEventTapDisabledByUserInput,
        ):
            Quartz.CGEventTapEnable(self.tap, True)
            return event

        if event_type == Quartz.kCGEventKeyDown:
            # fn + another key (fn+F5, fn+arrows, ...) is a shortcut, not dictation.
            # Streaming mode posts its own ⌘V while fn is held, so it never cancels.
            if self.recording and not self.streaming:
                self.cancelled = True
            return event

        flags = Quartz.CGEventGetFlags(event)
        control = bool(flags & Quartz.kCGEventFlagMaskControl)
        shift = bool(flags & Quartz.kCGEventFlagMaskShift)
        if self.recording and control and not self.command and not self.streaming:
            self.command = True
            log.info("   → コマンドモードに切り替えました")
        if self.recording and shift and not self.command and not self.streaming:
            self.start_streaming()
        keycode = Quartz.CGEventGetIntegerValueField(event, Quartz.kCGKeyboardEventKeycode)
        if keycode != FN_KEYCODE:
            return event
        fn_down = bool(flags & Quartz.kCGEventFlagMaskSecondaryFn)
        if fn_down and not self.recording:
            self.start(command=control, streaming=shift)
        elif not fn_down and self.recording:
            self.stop()
        return event

    def run(self) -> None:
        mask = Quartz.CGEventMaskBit(Quartz.kCGEventFlagsChanged) | Quartz.CGEventMaskBit(
            Quartz.kCGEventKeyDown
        )
        self.tap = Quartz.CGEventTapCreate(
            Quartz.kCGSessionEventTap,
            Quartz.kCGHeadInsertEventTap,
            Quartz.kCGEventTapOptionListenOnly,
            mask,
            self.on_event,
            None,
        )
        if self.tap is None:
            raise SystemExit("キー入力を監視できません。入力監視の権限を確認してください (README 参照)")
        source = Quartz.CFMachPortCreateRunLoopSource(None, self.tap, 0)
        Quartz.CFRunLoopAddSource(Quartz.CFRunLoopGetCurrent(), source, Quartz.kCFRunLoopCommonModes)
        Quartz.CGEventTapEnable(self.tap, True)

        threading.Thread(target=self.worker, daemon=True).start()
        with self.stream:
            log.info("準備完了。fn キーを押している間だけ録音します (終了は Ctrl+C)")
            log.info("shift + fn なら、押している間ずっと区切りごとに貼り付けます")
            if self.laya is not None:
                log.info("fn + control を押しながら話すとアプリを開きます (Slack / Chrome)")
            Quartz.CFRunLoopRun()


def check_permissions() -> None:
    # Each call shows the system prompt the first time; the grant applies to the
    # app that runs this script (Terminal, iTerm, VS Code, ...).
    missing = []
    if not Quartz.CGPreflightListenEventAccess():
        Quartz.CGRequestListenEventAccess()
        missing.append("入力監視 (fn キーの検知)")
    if not Quartz.CGPreflightPostEventAccess():
        Quartz.CGRequestPostEventAccess()
        missing.append("アクセシビリティ (⌘V での貼り付け)")
    if missing:
        raise SystemExit(
            "次の権限が必要です: "
            + "、".join(missing)
            + "\nシステム設定 > プライバシーとセキュリティ で、このスクリプトを実行している"
            "ターミナルアプリを許可してから、そのアプリを再起動してください。"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", default="auto", help="auto (既定: MPS があれば MPS) / mps / cpu")
    parser.add_argument(
        "--no-launcher", action="store_true", help="Laya を読み込まない (fn+control のアプリ起動を無効化)"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    # CFRunLoopRun blocks in C, where Python's KeyboardInterrupt never fires.
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    check_permissions()
    model = load_model(args.device)
    laya = None
    if not args.no_launcher:
        import launcher

        log.info("Laya を読み込み中...")
        laya = launcher.load_laya("mps" if args.device in ("auto", "mps") else "cpu")
    Dictation(model, laya).run()


if __name__ == "__main__":
    main()
