"""
音声認識サーバー - Hailo Whisper Pipeline を用いた 16bit PCM 16kHz 音声のテキスト変換

クライアントからTCPで受信した16bit PCM（16kHz, モノラル）の音声データを
HailoWhisperPipelineでテキストに変換し、結果を返します。

プロトコル:
  受信: [4バイト 長さ Big-Endian][長さバイト分の 16bit PCM]
  送信: [4バイト 長さ Big-Endian][UTF-8 テキスト]

実行: python server.py [オプション]
"""

import sys
import os
import argparse
import struct
import socket
import threading
import numpy as np
from app.hailo_whisper_pipeline import HailoWhisperPipeline
from app.whisper_hef_registry import HEF_REGISTRY
from common.preprocessing import preprocess
from common.postprocessing import clean_transcription
import pykakasi

# 音声仕様（PCI 16bit, 16kHz）
SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2  # 16bit

# モデル別の推奨チャンク長（秒）
BASE_CHUNK_DURATION = 5.0
TINY_CHUNK_DURATION = 10.0

def get_hef_paths(variant: str, hw_arch: str):
    """HEFファイルパスを取得"""
    try:
        entry = HEF_REGISTRY[variant][hw_arch]
        enc = entry["encoder"]
        dec = entry["decoder"]
        if not os.path.exists(enc) or not os.path.exists(dec):
            raise FileNotFoundError(f"HEF not found: {enc!r} or {dec!r}")
        return enc, dec
    except KeyError as e:
        raise ValueError(
            f"Unknown variant/hw_arch: {variant!r} / {hw_arch!r}"
        ) from e

def pcm_bytes_to_float(pcm_bytes: bytes) -> np.ndarray:
    """16bit PCM バイト列を float32 [-1, 1] の 1D 配列に変換"""
    samples = np.frombuffer(pcm_bytes, dtype=np.int16)
    return samples.astype(np.float32) / 32768.0

def handle_client(
    conn: socket.socket,
    addr,
    pipeline: HailoWhisperPipeline,
    chunk_duration: float,
    inference_lock: threading.Lock,
):
    """
    1クライアントのループ: 長さ付きPCMを受信 → テキストに変換 → 長さ付きUTF-8で返送
    """
    required_bytes = int(chunk_duration * SAMPLE_RATE * BYTES_PER_SAMPLE)
    header_size = 4  # 4 byte length prefix (big-endian)

    try:
        # [4 byte length]
        header = conn.recv(header_size)
        if len(header) < header_size:
            print(f"[{addr}] Incomplete length header, closing")
            return
        payload_len = struct.unpack(">I", header)[0]

        # 簡易ガード（例: 10秒まで）
        if payload_len > 10 * SAMPLE_RATE * BYTES_PER_SAMPLE:
            print(f"[{addr}] Payload too large: {payload_len} bytes")
            return
        if payload_len < required_bytes:
            # 短い場合はパディングで補う（またはエラー返却）
            print(f"[{addr}] Payload short: {payload_len} < {required_bytes}, padding")
        # ペイロード受信
        received = 0
        chunks = []
        while received < payload_len:
            want = min(payload_len - received, 65536)
            data = conn.recv(want)
            if not data:
                break
            chunks.append(data)
            received += len(data)
        pcm_bytes = b"".join(chunks)
        if len(pcm_bytes) < required_bytes:
            # 足りない分はゼロパディング
            pcm_bytes = pcm_bytes + b"\x00" * (required_bytes - len(pcm_bytes))

        # 音声 → float
        audio = pcm_bytes_to_float(pcm_bytes)

        # メルスペクトログラムに変換
        mel_list = preprocess(
            audio,
            is_nhwc=True,
            chunk_length=chunk_duration,
            chunk_offset=0,
            max_duration=int(chunk_duration) + 1,
            overlap=0.0,
        )
        text = ""
        if not mel_list:
            return
        else:
            for mel in mel_list:
                with inference_lock:
                    pipeline.send_data(mel)
                    raw_text = pipeline.get_transcription()
                #text = clean_transcription(raw_text) if raw_text else ""
                text += raw_text

        # 日本語文字列では認証が難しいのでローマ字に変換
        roma_text = ""
        for item in pykakasi.kakasi().convert(text):
            roma_text += item['hepburn']

        # 応答: [4 byte length][UTF-8 text]
        text_bytes = roma_text.strip().encode("utf-8")
        conn.sendall(struct.pack(">I", len(text_bytes)) + text_bytes)
        print(f"result: {text_bytes}")
    except (ConnectionResetError, BrokenPipeError, OSError) as e:
        print(f"[{addr}] Connection error: {e}")
    except Exception as e:
        print(f"[{addr}] Error: {e}")
    finally:
        try:
            conn.close()
        except OSError:
            pass

def run_server(
    host: str = "0.0.0.0",
    port: int = 50008,
    variant: str = "base",
    hw_arch: str = "hailo8l",
    multi_process_service: bool = False,
):
    # tiny / tiny.en は 10秒、それ以外は 5秒
    chunk_duration = TINY_CHUNK_DURATION if variant in ("tiny", "tiny.en") else BASE_CHUNK_DURATION
    encoder_path, decoder_path = get_hef_paths(variant, hw_arch)

    pipeline = HailoWhisperPipeline(
        encoder_path,
        decoder_path,
        variant=variant,
        multi_process_service=multi_process_service,
    )
    print(f"Pipeline ready: variant={variant}, hw_arch={hw_arch}, chunk={chunk_duration}s")

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((host, port))
    server.listen(5)
    print(f"TCP server listening on {host}:{port} (16bit PCM 16kHz)")

    inference_lock = threading.Lock()
    try:
        while True:
            conn, addr = server.accept()
            print(f"Client connected: {addr}")
            t = threading.Thread(
                target=handle_client,
                args=(conn, addr, pipeline, chunk_duration, inference_lock),
                daemon=True,
            )
            t.start()
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        server.close()
        pipeline.stop()

def main():
    parser = argparse.ArgumentParser(
        description="STTServer: receive 16bit PCM 16kHz audio, return transcription"
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind address")
    parser.add_argument("--port", type=int, default=50008, help="TCP port")
    parser.add_argument(
        "--variant",
        default="base",
        choices=["tiny", "tiny.en", "base"],
        help="Whisper variant",
    )
    parser.add_argument(
        "--hw-arch",
        default="hailo8l",
        choices=["hailo8", "hailo8l", "hailo10h"],
        help="Hailo hardware",
    )
    parser.add_argument(
        "--multi-process-service",
        action="store_true",
        help="Enable multi-process service",
    )
    args = parser.parse_args()

    run_server(
        host=args.host,
        port=args.port,
        variant=args.variant,
        hw_arch=args.hw_arch,
        multi_process_service=args.multi_process_service,
    )

if __name__ == "__main__":
    main()
