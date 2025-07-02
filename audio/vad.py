import numpy as np
from config.settings import (
    VAD_ENERGY_THRESHOLD, VAD_SPEECH_PAD_MS, VAD_MIN_SPEECH_MS,
    VAD_SILENCE_TIMEOUT_MS, SOURCE_SAMPLE_RATE, BLOCKSIZE
)


class VoiceActivityDetector:
    def __init__(self):
        self.silence_blocks = int(VAD_SILENCE_TIMEOUT_MS / (BLOCKSIZE / SOURCE_SAMPLE_RATE * 1000))
        self.min_speech = int(VAD_MIN_SPEECH_MS / (BLOCKSIZE / SOURCE_SAMPLE_RATE * 1000))
        self.pad_blocks = int(VAD_SPEECH_PAD_MS / (BLOCKSIZE / SOURCE_SAMPLE_RATE * 1000))

        self.is_speaking = False
        self.silence_count = 0
        self.speech_buf = []
        self.pad_buf = []

        print(f"VAD инициализирован: silence_blocks={self.silence_blocks}, "
              f"min_speech={self.min_speech}, pad_blocks={self.pad_blocks}")

    def process_audio_block(self, audio_block):
        """
        Обрабатывает блок аудио и возвращает готовые речевые сегменты
        Returns: список готовых аудио буферов для распознавания или None
        """
        rms = np.sqrt(np.mean(audio_block ** 2))

        # Управляем буфером padding
        self.pad_buf.append(audio_block)
        if len(self.pad_buf) > self.pad_blocks * 2:
            self.pad_buf.pop(0)

        # Детекция активности
        if rms > VAD_ENERGY_THRESHOLD:
            if not self.is_speaking:
                # Начало речи
                self.is_speaking = True
                self.speech_buf = self.pad_buf[-self.pad_blocks:].copy()
                # print(f"[VAD] Начало речи detected (RMS: {rms:.6f})")

            self.speech_buf.append(audio_block)
            self.silence_count = 0
        elif self.is_speaking:
            # Продолжаем запись тишины после речи
            self.silence_count += 1
            self.speech_buf.append(audio_block)

            if self.silence_count >= self.silence_blocks:
                # Конец речи
                if len(self.speech_buf) > self.min_speech + self.pad_blocks:
                    # Возвращаем готовый буфер (без последних блоков тишины)
                    result_buffer = self.speech_buf[:-self.silence_blocks]
                    print(f"[VAD] Конец речи: {len(result_buffer)} блоков аудио готово для STT")

                    # Сброс состояния
                    self.is_speaking = False
                    self.speech_buf = []
                    self.silence_count = 0

                    return result_buffer
                else:
                    # Слишком короткий сегмент, игнорируем
                    print(f"[VAD] Сегмент слишком короткий ({len(self.speech_buf)} блоков), игнорируем")
                    self.is_speaking = False
                    self.speech_buf = []
                    self.silence_count = 0

        return None

    def force_reset(self):
        """Принудительный сброс состояния VAD"""
        self.is_speaking = False
        self.speech_buf = []
        self.pad_buf = []
        self.silence_count = 0
        print("[VAD] Принудительный сброс состояния")

    def get_current_status(self):
        """Возвращает текущее состояние детектора"""
        return {
            "is_speaking": self.is_speaking,
            "speech_buffer_length": len(self.speech_buf),
            "silence_count": self.silence_count,
            "pad_buffer_length": len(self.pad_buf)
        }