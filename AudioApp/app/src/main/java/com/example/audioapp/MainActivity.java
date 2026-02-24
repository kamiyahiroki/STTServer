package com.example.audioapp;

import android.Manifest;
import android.content.pm.PackageManager;
import android.media.AudioAttributes;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.AudioTrack;
import android.media.MediaRecorder;
import android.os.Bundle;
import android.text.InputFilter;
import android.view.Gravity;
import android.view.LayoutInflater;
import android.view.MotionEvent;
import android.view.View;
import android.widget.ArrayAdapter;
import android.widget.AutoCompleteTextView;
import android.widget.Button;
import android.widget.PopupWindow;
import android.widget.TextView;
import android.widget.Toast;

import androidx.activity.EdgeToEdge;
import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.Socket;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;

public class MainActivity extends AppCompatActivity {

    private static final int PORT = 50008;
    private static final String[] IP_SUGGESTIONS = {"10.0.2.2", "192.168.0.80"};
    private static final int SAMPLE_RATE = 16000;
    private static final int CHANNEL_CONFIG = AudioFormat.CHANNEL_IN_MONO;
    private static final int AUDIO_FORMAT = AudioFormat.ENCODING_PCM_16BIT;
    private static final int REQUEST_RECORD_AUDIO = 1;

    private Button btnSendAudio;
    private AutoCompleteTextView ipAddressInput;
    private volatile boolean isRecording = false;
    private Thread recordThread;
    private AudioRecord audioRecord;
    private ByteArrayOutputStream audioBuffer;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_main);
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main), (v, insets) -> {
            Insets systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars());
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom);
            return insets;
        });

        btnSendAudio = findViewById(R.id.btn_send_audio);
        ipAddressInput = findViewById(R.id.ip_address_input);

        ArrayAdapter<String> ipAdapter = new ArrayAdapter<>(this,
                android.R.layout.simple_dropdown_item_1line, IP_SUGGESTIONS);
        ipAddressInput.setAdapter(ipAdapter);
        ipAddressInput.setThreshold(0);
        ipAddressInput.setOnFocusChangeListener((v, hasFocus) -> {
            if (hasFocus) {
                ipAddressInput.post(() -> ipAddressInput.showDropDown());
            }
        });
        ipAddressInput.setOnClickListener(v -> ipAddressInput.showDropDown());

        InputFilter digitsAndDotOnly = (source, start, end, dest, dstart, dend) -> {
            for (int i = start; i < end; i++) {
                char c = source.charAt(i);
                if (c != '.' && (c < '0' || c > '9')) {
                    return "";
                }
            }
            return null;
        };
        ipAddressInput.setFilters(new InputFilter[]{digitsAndDotOnly});

        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
                != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this,
                    new String[]{Manifest.permission.RECORD_AUDIO}, REQUEST_RECORD_AUDIO);
        }

        btnSendAudio.setOnTouchListener((v, event) -> {
            if (ActivityCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
                    != PackageManager.PERMISSION_GRANTED) {
                Toast.makeText(this, "マイクの権限を許可してください", Toast.LENGTH_SHORT).show();
                return false;
            }
            switch (event.getAction()) {
                case MotionEvent.ACTION_DOWN:
                    TextView textView = findViewById(R.id.textView);
                    textView.setText("");
                    v.setPressed(true);
                    startBuffering();
                    return true;
                case MotionEvent.ACTION_UP:
                case MotionEvent.ACTION_CANCEL:
                    v.setPressed(false);
                    stopBufferingAndSend();
                    return true;
            }
            return false;
        });
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions,
                                           @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == REQUEST_RECORD_AUDIO) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                Toast.makeText(this, "マイクの権限が許可されました", Toast.LENGTH_SHORT).show();
            }
        }
    }

    private void startBuffering() {
        if (isRecording) return;
        isRecording = true;
        audioBuffer = new ByteArrayOutputStream();
        recordThread = new Thread(this::recordAudioLoop);
        recordThread.start();
    }

    private void stopBufferingAndSend() {
        if (!isRecording) return;
        isRecording = false;
        if (recordThread != null) {
            try {
                recordThread.join(2000);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
            recordThread = null;
        }
        releaseAudioRecord();

        byte[] dataToSend = audioBuffer != null ? audioBuffer.toByteArray() : new byte[0];
        audioBuffer = null;

        if (dataToSend.length > 0) {
            String host = ipAddressInput.getText().toString().trim();
            if (host.isEmpty() || !isValidIPv4(host)) {
                Toast.makeText(this, "IPアドレスを入力してください", Toast.LENGTH_SHORT).show();
                return;
            }
            new Thread(() -> sendBufferedData(dataToSend, host)).start();
            //playAudioData(dataToSend);        // デバッグ用に再生する
        }
    }

    private void playAudioData(byte[] pcmData) {
        if (pcmData == null || pcmData.length == 0) return;
        new Thread(() -> {
            int bufferSize = AudioTrack.getMinBufferSize(SAMPLE_RATE, AudioFormat.CHANNEL_OUT_MONO, AUDIO_FORMAT);
            if (bufferSize <= 0) return;
            try {
                AudioTrack audioTrack = new AudioTrack.Builder()
                        .setAudioAttributes(new AudioAttributes.Builder()
                                .setUsage(AudioAttributes.USAGE_MEDIA)
                                .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                                .build())
                        .setAudioFormat(new AudioFormat.Builder()
                                .setEncoding(AUDIO_FORMAT)
                                .setSampleRate(SAMPLE_RATE)
                                .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                                .build())
                        .setBufferSizeInBytes(bufferSize * 2)
                        .setTransferMode(AudioTrack.MODE_STREAM)
                        .build();
                if (audioTrack.getState() != AudioTrack.STATE_INITIALIZED) return;
                audioTrack.play();
                int offset = 0;
                while (offset < pcmData.length) {
                    int toWrite = Math.min(bufferSize, pcmData.length - offset);
                    int written = audioTrack.write(pcmData, offset, toWrite);
                    if (written <= 0) break;
                    offset += written;
                }
                long durationMs = (pcmData.length / 2L * 1000L) / SAMPLE_RATE;
                Thread.sleep(Math.max(0, durationMs));
                audioTrack.stop();
                audioTrack.release();
            } catch (Exception ignored) {
            }
        }).start();
    }

    private void recordAudioLoop() {
        int bufferSize = AudioRecord.getMinBufferSize(SAMPLE_RATE, CHANNEL_CONFIG, AUDIO_FORMAT);
        if (bufferSize <= 0) {
            runOnUiThread(() -> Toast.makeText(this, "AudioRecordの初期化に失敗しました", Toast.LENGTH_SHORT).show());
            return;
        }

        try {
            audioRecord = new AudioRecord(MediaRecorder.AudioSource.MIC, SAMPLE_RATE,
                    CHANNEL_CONFIG, AUDIO_FORMAT, bufferSize * 2);
            if (audioRecord.getState() != AudioRecord.STATE_INITIALIZED) {
                runOnUiThread(() -> Toast.makeText(this, "マイクを開始できませんでした", Toast.LENGTH_SHORT).show());
                return;
            }

            audioRecord.startRecording();
            byte[] buffer = new byte[bufferSize];
            while (isRecording) {
                int read = audioRecord.read(buffer, 0, buffer.length);
                if (read > 0 && audioBuffer != null) {
                    audioBuffer.write(buffer, 0, read);
                }
            }
        } finally {
            releaseAudioRecord();
        }
    }

    private static boolean isValidIPv4(String host) {
        if (host == null || host.isEmpty()) return false;
        String[] parts = host.split("\\.");
        if (parts.length != 4) return false;
        for (String part : parts) {
            if (part.isEmpty()) return false;
            try {
                int octet = Integer.parseInt(part);
                if (octet < 0 || octet > 255) return false;
            } catch (NumberFormatException e) {
                return false;
            }
        }
        return true;
    }

    private void sendBufferedData(byte[] data, String host) {
        ByteBuffer header = ByteBuffer.allocate(4);
        header.putInt(data.length);
        byte[] headerBytes = header.array();
        try (Socket sock = new Socket(host, PORT)) {
            OutputStream out = sock.getOutputStream();
            out.write(headerBytes);
            out.write(data);
            out.flush();

            InputStream in = sock.getInputStream();
            int headerLength = in.read(headerBytes);
            if (headerLength != 4) {
                throw new IOException("ヘッダーの読み込みに失敗しました");
            }
            int dataLength = ByteBuffer.wrap(headerBytes).getInt();
            byte[] textBuf = new byte[dataLength];
            int total = 0;
            while (total < dataLength) {
                int n = in.read(textBuf, total, dataLength - total);
                if (n <= 0) break;
                total += n;
            }
            sock.close();

            String receivedText = new String(textBuf, StandardCharsets.UTF_8);
            final String textToShow = receivedText.isEmpty() ? "（受信データがありません）" : receivedText;
            runOnUiThread(() -> {
                TextView textView = findViewById(R.id.textView);
                if (textView != null) {
                    textView.setText(textToShow);
                }
            });
        } catch (IOException e) {
            runOnUiThread(() -> Toast.makeText(this,
                    "接続できません: " + host + ":" + PORT + " - " + e.getMessage(), Toast.LENGTH_LONG).show());
        }
    }

    private void releaseAudioRecord() {
        if (audioRecord != null) {
            try {
                if (audioRecord.getRecordingState() == AudioRecord.RECORDSTATE_RECORDING) {
                    audioRecord.stop();
                }
                audioRecord.release();
            } catch (Exception ignored) {
            }
            audioRecord = null;
        }
    }

    @Override
    protected void onDestroy() {
        if (isRecording) {
            isRecording = false;
            if (recordThread != null) {
                try {
                    recordThread.join(1000);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
            }
            releaseAudioRecord();
            audioBuffer = null;
        }
        super.onDestroy();
    }
}
