# STTServer — Whisper TCP 音声認識サーバー

`HailoWhisperPipeline` を用い、TCP で受信した **16bit PCM・16kHz・モノラル** の音声をテキストに変換するサーバーです。  
デフォルトで**日本語の書き起こし**（transcribe）を行い、英語への翻訳はしません。

## 前提

- プロジェクトルート（`STTServer`）に Hailo 実行環境と HEF が用意されていること
- 音声は **16bit PCM、サンプリングレート 16kHz、モノラル** であること

## プロトコル

- **受信**: `[4バイト 長さ Big-Endian][長さバイト分の 16bit PCM]`
- **送信**: `[4バイト 長さ Big-Endian][UTF-8 テキスト]`

1回の送信で「1チャンク」として処理します。モデルによって推奨長が異なります。

- **base**: 5秒分（160,000 バイト）推奨
- **tiny / tiny.en**: 10秒分（320,000 バイト）推奨

不足分はゼロパディング、超過分は先頭から使用されます。ペイロードは最大 10 秒分までです。

## 起動方法

プロジェクトルートで実行してください。

```bash
# デフォルト: 0.0.0.0:50008, variant=base, hw-arch=hailo8l
python stt_server.py

# オプション例
python stt_server.py --host 0.0.0.0 --port 50008 --variant base --hw-arch hailo8l
python stt_server.py --variant tiny --port 9000
python stt_server.py --multi-process-service
```

## オプション

| オプション | 説明 | デフォルト |
|-----------|------|------------|
| `--host` | バインドアドレス | `0.0.0.0` |
| `--port` | TCP ポート | `50008` |
| `--variant` | モデル: `tiny`, `tiny.en`, `base` | `base` |
| `--hw-arch` | Hailo: `hailo8`, `hailo8l`, `hailo10h` | `hailo8l` |
| `--multi-process-service` | マルチプロセスサービス有効化 | オフ |

**注意**: モデルと Hailo の組み合わせは HEF の有無に依存します。  
`base` は hailo8 / hailo8l、`tiny` は hailo8 / hailo8l / hailo10h、`tiny.en` は hailo10h のみ対応です。

## クライアント実装の目安

プロトコルに従い、次の手順でクライアントを実装できます。

1. TCP でサーバーに接続する。
2. **送信**: 4 バイト（Big-Endian の長さ）＋ その長さバイト分の 16bit PCM（16kHz モノラル）。
3. **受信**: 4 バイト（Big-Endian の長さ）＋ その長さバイト分の UTF-8 テキスト。

WAV を送る場合は、16bit・16kHz・モノラルに変換してから PCM 部分だけを送信してください。

## フォルダ構成

```
STTServer/
  stt_server.py       # TCP サーバー本体（エントリポイント）
  app/
    hailo_whisper_pipeline.py  # Hailo Whisper 推論パイプライン
    whisper_hef_registry.py    # モデル別 HEF パス
    download_resources.py      # リソース取得
    hefs/                      # Hailo HEF ファイル（要配置）
  common/
    preprocessing.py   # メルスペクトログラム前処理
    postprocessing.py  # 書き起こし後処理
    audio_utils.py
    record_utils.py
  README.md
```

## 言語・タスク

パイプラインはデフォルトで **言語=日本語（ja）**、**タスク=書き起こし（transcribe）** で初期化されています。  
日本語音声はそのまま日本語テキストで返り、英語へは翻訳しません。  
他言語や「英語へ翻訳」が必要な場合は、`HailoWhisperPipeline` の `language` / `task` を変更してください（現状は `stt_server.py` からは未指定でパイプラインのデフォルトが使われます）。
