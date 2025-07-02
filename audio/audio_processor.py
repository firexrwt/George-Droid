import asyncio
import subprocess
import sys
import json
import os
import numpy as np
import scipy.signal
import sounddevice as sd

from config.settings import (
    PIPER_EXE_PATH, VOICE_MODEL_PATH, VOICE_CONFIG_PATH,
    STT_MODEL_SIZE, STT_DEVICE, STT_COMPUTE_TYPE,
    SOURCE_SAMPLE_RATE, TARGET_SAMPLE_RATE, TARGET_DTYPE
)

# Глобальные переменные
stt_model = None
piper_sample_rate = None
tts_lock = asyncio.Lock()
chosen_output_device_id = None
activity_time_callback = None


def set_activity_time_callback(callback_func):
    """Установка callback для обновления времени активности"""
    global activity_time_callback
    activity_time_callback = callback_func


def init_stt_model():
    """Инициализация модели STT"""
    global stt_model
    try:
        from faster_whisper import WhisperModel
        stt_model = WhisperModel(STT_MODEL_SIZE, device=STT_DEVICE, compute_type=STT_COMPUTE_TYPE)
        print(f"Модель STT ({STT_MODEL_SIZE}, {STT_DEVICE}, {STT_COMPUTE_TYPE}) загружена.")
        return True
    except ImportError:
        print("ОШИБКА: faster-whisper не установлен. STT не будет работать.")
        return False
    except Exception as e_stt_load:
        print(f"Критическая ошибка загрузки faster-whisper: {e_stt_load}")
        return False


def init_piper_tts():
    """Инициализация Piper TTS"""
    global piper_sample_rate
    try:
        if os.path.exists(VOICE_CONFIG_PATH):
            with open(VOICE_CONFIG_PATH, 'r', encoding='utf-8') as f:
                piper_config = json.load(f)
                piper_sample_rate = piper_config.get('audio', {}).get('sample_rate')
            if not piper_sample_rate:
                print(f"ОШИБКА: Не найден 'sample_rate' в {VOICE_CONFIG_PATH}")
                return False
            else:
                print(f"Piper TTS sample rate: {piper_sample_rate}")
        else:
            print(f"ОШИБКА: Не найден JSON конфиг голоса: {os.path.abspath(VOICE_CONFIG_PATH)}")
            return False

        if not all([os.path.exists(PIPER_EXE_PATH), os.path.exists(VOICE_MODEL_PATH), piper_sample_rate]):
            print("TTS Piper не будет работать из-за отсутствия файлов или sample_rate.")
            piper_sample_rate = None
            return False
        return True
    except Exception as e_piper_init:
        print(f"Критическая ошибка инициализации Piper TTS: {e_piper_init}")
        piper_sample_rate = None
        return False


def set_output_device(device_id):
    """Установка устройства вывода"""
    global chosen_output_device_id
    chosen_output_device_id = device_id


def resample_audio(audio_data: np.ndarray, input_rate: int, target_rate: int) -> np.ndarray:
    """Передискретизация аудио"""
    if input_rate == target_rate:
        return audio_data.astype(np.float32)
    try:
        duration = audio_data.shape[0] / input_rate
        new_num_samples = int(duration * target_rate)
        resampled_audio = scipy.signal.resample(audio_data, new_num_samples)
        return resampled_audio.astype(np.float32)
    except Exception as e:
        print(f"Ошибка передискретизации: {e}", file=sys.stderr)
        return np.array([], dtype=np.float32)


def transcribe_audio_faster_whisper(audio_np_array):
    """Распознавание речи через Faster-Whisper"""
    global stt_model
    if stt_model is None or not isinstance(audio_np_array, np.ndarray) or audio_np_array.size == 0:
        return None
    try:
        segments, _ = stt_model.transcribe(audio_np_array, language="ru", word_timestamps=False)
        return "".join(segment.text for segment in segments).strip()
    except Exception as e:
        print(f"Ошибка распознавания whisper: {e}", file=sys.stderr)
        return None


def play_raw_audio_sync(audio_bytes, samplerate, dtype='int16'):
    """Синхронное воспроизведение аудио"""
    global chosen_output_device_id
    if not audio_bytes or not samplerate:
        return
    try:
        sd.play(np.frombuffer(audio_bytes, dtype=dtype), samplerate=samplerate, blocking=True,
                device=chosen_output_device_id)
    except Exception as e:
        print(f"Ошибка sd.play: {e}", file=sys.stderr)


async def speak_text(text_to_speak):
    """Асинхронная озвучка текста через Piper TTS"""
    global piper_sample_rate, tts_lock
    if not piper_sample_rate or not os.path.exists(PIPER_EXE_PATH) or not os.path.exists(VOICE_MODEL_PATH):
        print("TTS недоступен.")
        return

    async with tts_lock:
        print(f"[TTS] Озвучка: \"{text_to_speak[:50]}...\"")
        cmd = [PIPER_EXE_PATH, '--model', VOICE_MODEL_PATH, '--output-raw']
        process = None
        audio_bytes = None
        stderr_bytes = b''
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            )
            audio_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(input=text_to_speak.encode('utf-8')), timeout=30
            )
            if process.returncode != 0:
                print(f"Ошибка piper: {process.returncode}, {stderr_bytes.decode(errors='ignore')}", file=sys.stderr)
                audio_bytes = None
            elif not audio_bytes:
                print(f"Ошибка piper: нет аудио. Stderr: {stderr_bytes.decode(errors='ignore')}", file=sys.stderr)

        except asyncio.TimeoutError:
            print("Ошибка TTS: Таймаут piper.exe", file=sys.stderr)
            if process and process.returncode is None:
                try:
                    process.kill()
                    await process.wait()
                except Exception as kill_e:
                    print(f"Ошибка убийства piper: {kill_e}", file=sys.stderr)
        except FileNotFoundError:
            print(f"КРИТИКА: Не найден piper.exe: {PIPER_EXE_PATH}", file=sys.stderr)
        except Exception as e:
            print(f"Ошибка вызова piper: {e}", file=sys.stderr)

        if audio_bytes:
            try:
                await asyncio.to_thread(play_raw_audio_sync, audio_bytes, piper_sample_rate)

                # Обновляем время активности после TTS
                if activity_time_callback:
                    import time
                    activity_time_callback(time.time())

            except Exception as e_play:
                print(f"Ошибка play audio: {e_play}", file=sys.stderr)
            print(f"[TTS] Озвучка завершена.")


def convert_to_mono(audio_data):
    """Конвертация в моно"""
    if audio_data.ndim > 1 and audio_data.shape[1] > 1:
        return audio_data.mean(axis=1)
    return audio_data