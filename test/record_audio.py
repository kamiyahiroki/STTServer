"""
音声認識テスト

- メインウィンドウ: パス入力、レコードボタン、測定ボタン、測定エリア（表示/非表示切替）
- 測定ボタンで測定エリア（IP/ポート/結果/実行）の表示・非表示を切り替え
- プロトコル: [4バイト 長さ Big-Endian][ペイロード]
"""

import os
import queue
import struct
import socket
import threading
import time
import wave
import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog
from datetime import datetime
from pathlib import Path

try:
    import sounddevice as sd
    import numpy as np
except ImportError:
    sd = None
    np = None

SAMPLE_RATE = 16000
CHANNELS = 1
BYTES_PER_SAMPLE = 2  # 16bit
HEADER_SIZE = 4

def pcm_path_dir(base_path: str) -> str:
    """入力パスを保存用ディレクトリとして正規化（空ならカレント）"""
    if not base_path or not base_path.strip():
        return os.path.abspath(".")
    p = base_path.strip().rstrip("/\\")
    if not p:
        return os.path.abspath(".")
    if os.path.isfile(p):
        return os.path.dirname(p)
    return os.path.abspath(p)


def record_while_pressed_callback(
    queue,
    stop_event: threading.Event,
):
    """録音スレッド: 16bit PCM 16kHz モノラルでキューに溜める"""
    if sd is None or np is None:
        queue.put(None)
        return
    recorded = []

    def audio_callback(indata, frames, time_info, status):
        if status:
            print("Audio status:", status)
        recorded.append(indata.copy())

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="int16",
        callback=audio_callback,
    ):
        while not stop_event.is_set():
            stop_event.wait(0.05)
    if recorded:
        data = np.concatenate(recorded, axis=0)
        if data.ndim > 1:
            data = np.mean(data, axis=1).astype(np.int16)
        queue.put(data.tobytes())
    else:
        queue.put(b"")


class MainWindow:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("音声認識テスト")
        self.root.resizable(False, False)
        self.recording = False
        self.record_stop_event = None
        self.record_thread = None
        self.record_queue = None
        self._build_ui()

    def _build_ui(self):
        f = ttk.Frame(self.root, padding=10)
        f.pack(fill=tk.BOTH, expand=True)

        ttk.Label(f, text="保存先ディレクトリ").pack(anchor=tk.W)
        path_row = ttk.Frame(f)
        path_row.pack(fill=tk.X, pady=(0, 4))
        self.path_var = tk.StringVar(value=os.path.abspath("./data") if os.path.exists("./data") else os.path.abspath("."))
        self.path_entry = ttk.Entry(path_row, textvariable=self.path_var, width=60)
        self.path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
        self.folder_btn = ttk.Button(path_row, text="参照…", command=self._choose_folder)
        self.folder_btn.pack(side=tk.LEFT, padx=2)

        ttk.Label(f, text="名前（ローマ字）").pack(anchor=tk.W)
        filename_row = ttk.Frame(f)
        filename_row.pack(fill=tk.X, pady=(0, 8))
        self.filename_var = tk.StringVar()
        self.filename_entry = ttk.Entry(filename_row, textvariable=self.filename_var, width=60)
        self.filename_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
        self._setup_filename_validation()
        self.record_btn = ttk.Button(filename_row, text="録音")
        self.record_btn.pack(side=tk.LEFT, padx=2)
        self.record_btn.bind("<ButtonPress-1>", self._on_record_press)
        self.record_btn.bind("<ButtonRelease-1>", self._on_record_release)

        self.measure_btn = ttk.Button(f, text="測定", command=self._toggle_measure_area)
        self.measure_btn.pack(anchor=tk.W, pady=(0, 8))

        # 測定エリア
        self.measure_frame = ttk.Frame(f, padding=0)
        self._measure_visible = False

        ttk.Label(self.measure_frame, text="IPアドレス").pack(anchor=tk.W)
        self.ip_var = tk.StringVar(value="192.168.0.80")
        ttk.Entry(self.measure_frame, textvariable=self.ip_var, width=20).pack(fill=tk.X, pady=(0, 8))

        ttk.Label(self.measure_frame, text="ポート").pack(anchor=tk.W)
        self.port_var = tk.StringVar(value="50008")
        ttk.Entry(self.measure_frame, textvariable=self.port_var, width=10).pack(fill=tk.X, pady=(0, 8))

        ttk.Label(self.measure_frame, text="結果").pack(anchor=tk.W)
        self.result_text = scrolledtext.ScrolledText(
            self.measure_frame, height=8, width=50
        )
        self.result_text.pack(fill=tk.BOTH, expand=True, pady=(0, 8))

        ttk.Button(self.measure_frame, text="実行", command=self._execute_measure).pack(pady=4)

        # ステータスバー
        status_frame = ttk.Frame(self.root, padding=(8, 4))
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        self.status_var = tk.StringVar(value="")
        self.status_label = tk.Label(
            status_frame,
            textvariable=self.status_var,
            relief=tk.SUNKEN,
            anchor=tk.W,
            bd=0,
        )
        self.status_label.pack(fill=tk.X)

    def _set_status(self, message: str):
        """ステータスバーにメッセージを表示"""
        self.status_var.set(message)

    def _on_record_press(self, ev):
        if self.recording:
            return
        if sd is None or np is None:
            self._set_status("エラー: sounddevice と numpy をインストールしてください。")
            return
        if not self.filename_var.get():
            self._set_status("エラー: 名前（ローマ字）を入力してください。")
            return
        self.recording = True
        self.record_stop_event = threading.Event()
        self.record_queue = queue.Queue()
        self.record_thread = threading.Thread(
            target=record_while_pressed_callback,
            args=(self.record_queue, self.record_stop_event),
            daemon=True,
        )
        self.record_thread.start()

    def _on_record_release(self, ev):
        if not self.recording:
            return
        if not self.filename_var.get():
            return
        self.recording = False
        if self.record_stop_event:
            self.record_stop_event.set()
        if self.record_thread:
            self.record_thread.join(timeout=2.0)

        try:
            pcm_bytes = self.record_queue.get(timeout=0.5)
        except Exception:
            pcm_bytes = b""
        if not pcm_bytes:
            return
        out_dir = pcm_path_dir(self.path_var.get())
        os.makedirs(out_dir, exist_ok=True)
        base_name = (self.filename_var.get()).strip()
        path = self._get_unique_wav_path(out_dir, base_name)
        with wave.open(path, "wb") as wav_f:
            wav_f.setnchannels(CHANNELS)
            wav_f.setsampwidth(BYTES_PER_SAMPLE)
            wav_f.setframerate(SAMPLE_RATE)
            wav_f.writeframes(pcm_bytes)
        self._set_status(f"保存しました: {path}")

    def _choose_folder(self):
        """フォルダ選択ダイアログを表示し、選択パスを path_var に設定"""
        initial = self.path_var.get().strip()
        if not initial or not os.path.isdir(initial):
            initial = os.path.abspath(".")
        path = filedialog.askdirectory(title="保存先ディレクトリを選択", initialdir=initial)
        if path:
            self.path_var.set(path)

    def _setup_filename_validation(self):
        """ファイル名エントリを英字のみに制限"""
        vcmd = (self.root.register(self._validate_filename), "%P")
        self.filename_entry.configure(validate="key", validatecommand=vcmd)

    def _validate_filename(self, new_value: str) -> bool:
        """英字（a-zA-Z）のみ許可"""
        return all(c.isalpha() and c.isascii() for c in new_value)

    def _get_unique_wav_path(self, out_dir: str, base_name: str) -> str:
        """同名が存在する場合はナンバリングしたパスを返す（例: name.wav, name_1.wav）"""
        if not base_name:
            base_name = "record"
        base_name = "".join(c for c in base_name if c.isalpha() and c.isascii()) or "record"
        path = os.path.join(out_dir, base_name + ".wav")
        if not os.path.exists(path):
            return path
        n = 1
        while True:
            path = os.path.join(out_dir, f"{base_name}_{n}.wav")
            if not os.path.exists(path):
                return path
            n += 1

    def _toggle_measure_area(self):
        """測定ボタンで測定エリアの表示/非表示を切り替え"""
        if self._measure_visible:
            self.measure_frame.pack_forget()
            self._measure_visible = False
        else:
            self.measure_frame.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
            self._measure_visible = True

    def _execute_measure(self):
        ip = self.ip_var.get().strip()
        try:
            port = int(self.port_var.get().strip())
        except ValueError:
            self._set_status("エラー: ポートは数値で指定してください。")
            return
        base_dir = pcm_path_dir(self.path_var.get())
        if not os.path.isdir(base_dir):
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"ディレクトリがありません: {base_dir}")
            return

        wav_files = sorted(Path(base_dir).glob("*.wav"))
        if not wav_files:
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"音声ファイル(.wav)がありません: {base_dir}")
            return

        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, f"送信先: {ip}:{port} ファイル数: {len(wav_files)}\n")

        count = 0
        for path in wav_files:
            stem = "".join([i for i in path.stem if i.isalpha()])
            try:
                received, elapsed_ms = self._send_and_receive(ip, port, path)
                match = received.strip().lower() == stem.lower()
                if match:
                    count += 1
                status = "OK" if match else "NG"
                self.result_text.insert(
                    tk.END, f"{path.name},{received},[{status}],{elapsed_ms}ms\n"
                )
            except Exception as e:
                self.result_text.insert(tk.END, f"{path.name},,[ERROR],{e}\n")
            self.root.update()

        total = len(wav_files)
        ratio = (count / total * 100) if total else 0
        self.result_text.insert(tk.END, f"\n正解率: {count} / {total} = {ratio:.1f}%\n")

    def _send_and_receive(self, ip: str, port: int, path: Path) -> str:
        with wave.open(str(path), "rb") as wav_f:
            payload = wav_f.readframes(wav_f.getnframes())
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(30)
            sock.connect((ip, port))
            header = struct.pack(">I", len(payload))
            sock.sendall(header + payload)
            begin_ms = time.perf_counter()
            h = sock.recv(HEADER_SIZE)
            elapsed_ms = int((time.perf_counter() - begin_ms) * 1000)
            if len(h) < HEADER_SIZE:
                raise ConnectionError("応答ヘッダ不足")
            text_len = struct.unpack(">I", h)[0]
            received = b""
            while len(received) < text_len:
                got = sock.recv(min(4096, text_len - len(received)))
                if not got:
                    break
                received += got
        return received.decode("utf-8", errors="replace"), elapsed_ms


def main():
    app = MainWindow()
    app.root.mainloop()


if __name__ == "__main__":
    main()
