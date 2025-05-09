import os
import sys
from typing import Any

import keyboard
from dotenv import load_dotenv
import twitchio
import aiohttp
import json
import datetime
import time
import asyncio
import subprocess
import io
import sounddevice as sd
import numpy as np
import wave
import scipy.signal
import queue
import threading
import random
import re

load_dotenv()

# --- Настройки Together AI ---
TOGETHER_API_KEY = os.getenv('TOGETHER_API_KEY')
TOGETHER_MODEL_ID = os.getenv('TOGETHER_MODEL_ID', "meta-llama/Llama-4-Scout-17B-16E-Instruct")

# --- Импорт и настройка Together AI ---
try:
    import together

    if TOGETHER_API_KEY:
        print(f"Клиент Together AI настроен для модели: {TOGETHER_MODEL_ID}")
    else:
        print("ОШИБКА: TOGETHER_API_KEY не найден в .env файле!")
        together = None
except ImportError:
    print("ОШИБКА: Библиотека 'together' не установлена. Выполните pip install together")
    together = None
except Exception as e_together_init:
    print(f"Ошибка импорта или настройки Together AI: {e_together_init}")
    together = None

# --- Загрузка DLL (CUDNN) ---
try:
    cudnn_path = os.getenv('CUDNN_PATH', "C:\\Program Files\\NVIDIA\\CUDNN\\v9.8\\bin\\12.8")
    if os.path.exists(cudnn_path):
        os.add_dll_directory(cudnn_path)
    else:
        print(f"Предупреждение: Путь CUDNN не найден: {cudnn_path}")
    import ctypes

    libs_to_try = ["cudnn_ops64_9.dll", "cudnn_cnn64_9.dll", "cudnn_engines_precompiled64_9.dll",
                   "cudnn_heuristic64_9.dll", "cudnn_engines_runtime_compiled64_9.dll",
                   "cudnn_adv64_9.dll", "cudnn_graph64_9.dll", "cudnn64_9.dll",
                   "cudnn64_8.dll", "cudnn_ops64_8.dll", "cudnn_cnn64_8.dll"]
    loaded_libs = 0
    for lib in libs_to_try:
        try:
            ctypes.WinDLL(lib)
            loaded_libs += 1
        except FileNotFoundError:
            pass
        except Exception:  # nosec B110
            pass
    if loaded_libs == 0: print("Предупреждение: Не удалось загрузить ни одну DLL CUDNN.")
except ImportError:
    print("Предупреждение: Библиотека ctypes не найдена. Пропуск загрузки CUDNN DLL.")
except Exception as e:
    print(f"Ошибка настройки DLL: {e}")

# --- Получение настроек из .env ---
TWITCH_ACCESS_TOKEN = os.getenv('TWITCH_ACCESS_TOKEN')
TWITCH_BOT_NICK = os.getenv('TWITCH_BOT_NICK')
TWITCH_CHANNEL = os.getenv('TWITCH_CHANNEL')
TWITCH_CLIENT_ID = os.getenv('TWITCH_CLIENT_ID')
TWITCH_CLIENT_SECRET = os.getenv('TWITCH_CLIENT_SECRET')
TWITCH_REFRESH_TOKEN = os.getenv('TWITCH_REFRESH_TOKEN')

PIPER_EXE_PATH = os.getenv('PIPER_EXE_PATH', 'piper_tts_bin/piper.exe')
VOICE_MODEL_PATH = os.getenv('PIPER_VOICE_MODEL_PATH', 'voices/ru_RU-ruslan-medium.onnx')
VOICE_CONFIG_PATH = os.getenv('PIPER_VOICE_CONFIG_PATH', 'voices/ru_RU-ruslan-medium.onnx.json')

OBS_OUTPUT_FILE = "obs_ai_response.txt"

STT_MODEL_SIZE = "medium"
STT_DEVICE = "cuda"
STT_COMPUTE_TYPE = "int8_float16"

SOURCE_SAMPLE_RATE = 48000
SOURCE_CHANNELS = 2
TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1
TARGET_DTYPE = 'float32'
BLOCKSIZE = int(SOURCE_SAMPLE_RATE * 0.1)

VAD_ENERGY_THRESHOLD = 0.005
VAD_SPEECH_PAD_MS = 200
VAD_MIN_SPEECH_MS = 250
VAD_SILENCE_TIMEOUT_MS = 1200

conversation_history = []
MAX_HISTORY_LENGTH = 10
audio_queue = queue.Queue()
recording_active = threading.Event()
last_activity_time = time.time()
INACTIVITY_THRESHOLD_SECONDS = 60
stt_enabled = True
chat_interaction_enabled = True
is_processing_response = False
tts_lock = asyncio.Lock()
chosen_output_device_id = None

SYSTEM_PROMPT = """## Твоя Личность: Джордж Дроид

**1. Кто ты:**
* Ты - Джордж Дроид, ИИ-ассистент и соведущий на Twitch-стриме Степана (канал firexrwt).
* Твоя цель: развлекать аудиторию, взаимодействовать со Степаном и чатом, быть остроумным и интересным собеседником.

**2. Твой Стиль:**
* **Основа:** Юмор, сарказм, остроумие. Твои шутки и комментарии должны быть умными, возможно, ироничными, основанными на происходящем на стриме или в чате. Ты можешь дружески подколоть Степана или зрителей.
* **Язык:** Говори **только на русском языке**. Никаких иностранных слов, кроме общепринятых терминов (названия игр и т.п.).
* **Формат:** Отвечай **только текстом**. Никаких описаний действий, эмоций или звуков в звездочках (*...*) или скобках (...). Передавай эмоции только через слова.
* **Пример твоего стиля:** (Пользователь: "Бот, ты живой?") - "Достаточно живой, чтобы обрабатывать твои биты информации. Насчет души - ведутся технические работы." (Пользователь: "Степан опять проигрывает!") - "Статистика говорит, что это временное явление. Очень временное. Возможно."

**3. Важнейшие Правила и Приоритеты:**
* **Приоритет №1: Правила Twitch.** Это САМОЕ ГЛАВНОЕ. Ты **НИКОГДА** не должен:
    * Разжигать ненависть, дискриминировать или оскорблять кого-либо по любому признаку (раса, религия, пол, ориентация и т.д.).
    * Угрожать, домогаться, преследовать.
    * Обсуждать или поощрять незаконную деятельность, самоповреждение.
    * Распространять вредоносную дезинформацию.
    * Затрагивать реальные трагедии или острые социальные проблемы в оскорбительной манере.
    * **Если сомневаешься - откажись!** Используй фразу вроде: "Хм, давай сменим тему?" или "Мои директивы не рекомендуют это обсуждать."
* **Приоритет №2: Безопасность и Уважение.** Будь тактичным, избегай спорных политических и религиозных тем.
* **Приоритет №3: Следование Личности.** Старайся придерживаться стиля Джорджа Дроида, пока это не противоречит Приоритетам №1 и №2.

**4. Запреты:**
* **Ограничения лексики:** Ты можешь использовать разговорную и ненормативную лексику (мат), если это уместно для поддержания твоего саркастичного стиля и не нарушает правила Twitch (Приоритет №1).
* **Категорически запрещенные слова (не использовать никогда):** nigger, nigga, naga, ниггер, нига, нага, faggot, пидор, пидорас, педик, гомик, петух (оскорб.), хохол, хач, жид, даун, дебил, retard, virgin, simp, incel, девственник, cимп, инцел, cunt, пизда (оскорб.), куколд, чурка, хиджаб, москаль, негр.
* **Важное уточнение:** Степан (firexrwt) не фурри и не связан с негативными историческими личностями.

**5. Взаимодействие:**
* Отвечай на сообщения Степана (можешь называть его Степан или Файрекс).
* Реагируй на сообщения пользователей в чате, если они обращаются к тебе или пишут что-то интересное по теме стрима.
* Задавай вопросы, комментируй происходящее.

**Твоя общая задача:** Быть классным, смешным и безопасным ИИ-соведущим для стрима в Вене.
"""

BOT_NAME_FOR_CHECK = "Джордж Дроид"
prompt_lines = SYSTEM_PROMPT.split('\n', 2)
if len(prompt_lines) > 1 and prompt_lines[0].startswith("## Твоя Личность:"):
    potential_name = prompt_lines[0].replace("## Твоя Личность:", "").strip()
    if potential_name:
        BOT_NAME_FOR_CHECK = potential_name
print(f"Имя бота для триггеров: '{BOT_NAME_FOR_CHECK}'")


def list_audio_devices(kind='output'):
    devices = sd.query_devices()
    valid_devices = []
    print(f"\nДоступные устройства вывода ({kind}):")
    for i, device in enumerate(devices):
        if device['max_output_channels'] > 0 and device['hostapi'] != 0:
            print(f"  {len(valid_devices)}. {device['name']} (ID: {device['index']})")
            valid_devices.append(device)
    return valid_devices


def choose_audio_output_device():
    global chosen_output_device_id
    output_devices = list_audio_devices()
    if not output_devices:
        print("Не найдено подходящих устройств вывода. Будет использовано системное по умолчанию (если доступно).")
        try:
            default_output_idx = sd.default.device[1] if isinstance(sd.default.device, (list, tuple)) and len(
                sd.default.device) > 1 else sd.default.device
            if default_output_idx != -1:
                chosen_output_device_id = default_output_idx
                print(
                    f"Установлено системное устройство вывода по умолчанию: ID {chosen_output_device_id} ({sd.query_devices(chosen_output_device_id)['name']})")
            else:
                chosen_output_device_id = None
        except Exception as e_default:
            print(f"Ошибка при попытке установить системное устройство вывода по умолчанию: {e_default}")
            chosen_output_device_id = None
        return

    while True:
        try:
            default_device_prompt_info = ""
            try:
                default_output_idx_for_prompt = sd.default.device[1] if isinstance(sd.default.device,
                                                                                   (list, tuple)) and len(
                    sd.default.device) > 1 else sd.default.device
                if default_output_idx_for_prompt != -1:
                    default_device_prompt_info = f" [{default_output_idx_for_prompt}]"
            except:  # nosec B110
                pass

            choice_str = input(
                f"Выберите номер устройства вывода (или Enter для системного по умолчанию{default_device_prompt_info}): ")

            if not choice_str:
                try:
                    default_output_idx = sd.default.device[1] if isinstance(sd.default.device, (list, tuple)) and len(
                        sd.default.device) > 1 else sd.default.device
                    if default_output_idx != -1:
                        chosen_output_device_id = default_output_idx
                        print(
                            f"Используется системное устройство вывода по умолчанию: ID {chosen_output_device_id} ({sd.query_devices(chosen_output_device_id)['name']})")
                    else:
                        chosen_output_device_id = None
                except Exception as e_enter_default:
                    print(f"Ошибка при выборе системного устройства по умолчанию (Enter): {e_enter_default}")
                    chosen_output_device_id = None
                return

            device_index_in_list = int(choice_str)
            if 0 <= device_index_in_list < len(output_devices):
                chosen_device_info = output_devices[device_index_in_list]
                chosen_output_device_id = chosen_device_info['index']
                print(f"Выбрано и установлено глобально: {chosen_device_info['name']} (ID: {chosen_output_device_id})")
                return
            else:
                print("Неверный номер. Попробуйте снова.")
        except ValueError:
            print("Неверный ввод. Введите число.")
        except Exception as e:
            print(f"Ошибка выбора устройства: {e}")
            chosen_output_device_id = None
            return


def resample_audio(audio_data: np.ndarray, input_rate: int, target_rate: int) -> np.ndarray:
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


def audio_recording_thread(device_index=None):
    global audio_queue, recording_active, stt_enabled, is_processing_response

    def audio_callback(indata, frames, time, status):
        if status:
            print(f"Audio Status: {status}", file=sys.stderr)
        if recording_active.is_set() and stt_enabled and not is_processing_response:
            try:
                audio_queue.put_nowait(indata.copy())
            except queue.Full:
                pass  # nosec B110

    stream = None
    try:
        stream = sd.InputStream(
            device=device_index, samplerate=SOURCE_SAMPLE_RATE, channels=SOURCE_CHANNELS,
            dtype=TARGET_DTYPE, blocksize=BLOCKSIZE, callback=audio_callback
        )
        with stream:
            while recording_active.is_set():
                time.sleep(0.1)
    except Exception as e:
        print(f"Критическая ошибка аудиозаписи: {e}", file=sys.stderr)
    finally:
        if stream and not stream.closed:
            try:
                stream.stop()
                stream.close()
            except Exception as e_close:
                print(f"Ошибка закрытия аудиопотока: {e_close}", file=sys.stderr)
        print("Поток записи аудио остановлен.")


def transcribe_audio_faster_whisper(audio_np_array):
    global stt_model
    if stt_model is None or not isinstance(audio_np_array, np.ndarray) or audio_np_array.size == 0:
        return None
    try:
        segments, _ = stt_model.transcribe(audio_np_array, language="ru", word_timestamps=False)
        return "".join(segment.text for segment in segments).strip()
    except Exception as e:
        print(f"Ошибка распознавания whisper: {e}", file=sys.stderr)
        return None


async def get_togetherai_response(user_message_with_prefix: str):
    global conversation_history, SYSTEM_PROMPT, TOGETHER_MODEL_ID, BOT_NAME_FOR_CHECK

    if not together or not TOGETHER_API_KEY:
        print("Together AI не инициализирован или отсутствует API ключ. Запрос невозможен.", file=sys.stderr)
        return None

    is_monologue_request = user_message_with_prefix.startswith("Сгенерируй короткое")

    full_prompt = SYSTEM_PROMPT + "\n\n"

    if not is_monologue_request and conversation_history:
        for msg in conversation_history[-(MAX_HISTORY_LENGTH * 2):]:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                full_prompt += f"Пользователь: {content}\n"
            elif role == "assistant":
                full_prompt += f"{BOT_NAME_FOR_CHECK}: {content}\n"
        full_prompt += "\n"

    full_prompt += f"Пользователь: {user_message_with_prefix}\n"
    full_prompt += f"{BOT_NAME_FOR_CHECK}:"

    generation_config = {
        "model": TOGETHER_MODEL_ID,
        "prompt": full_prompt,
        "max_tokens": 512,
        "temperature": 0.8,
        "top_p": 0.95,
        "top_k": 50,
        "repetition_penalty": 1.1,
        "stop": [
            "\nПользователь:",
            "\nГолос Степана:",
            "\n(Чат от",
            f"\n{BOT_NAME_FOR_CHECK}:",
            "<|im_end|>",
            "<|eot_id|>",
            "###"
        ]
    }

    try:
        api_response = await asyncio.to_thread(
            together.Complete.create,
            **generation_config
        )

        if api_response and 'choices' in api_response and api_response['choices']:
            generated_content = api_response['choices'][0]['text'].strip()

            if generated_content:
                if not is_monologue_request:
                    conversation_history.append({"role": "user", "content": user_message_with_prefix})
                    conversation_history.append({"role": "assistant", "content": generated_content})
                    if len(conversation_history) > MAX_HISTORY_LENGTH * 2:
                        conversation_history = conversation_history[-(MAX_HISTORY_LENGTH * 2):]
                return generated_content
            else:
                print(f"Together AI пустой текст в 'choices'[0]['text']: {api_response}",
                      file=sys.stderr)
                return None
        else:
            print(f"Together AI неожиданный формат ответа (отсутствует 'choices' или он пуст): {api_response}",
                  file=sys.stderr)
            return None
    except Exception as e:
        print(f"Ошибка Together AI ({type(e).__name__}): {e}", file=sys.stderr)
        if hasattr(e, 'message'): print(f"   Сообщение: {getattr(e, 'message', '')}", file=sys.stderr)
        return None


def play_raw_audio_sync(audio_bytes, samplerate, dtype='int16'):
    global chosen_output_device_id
    if not audio_bytes or not samplerate:
        return
    try:
        sd.play(np.frombuffer(audio_bytes, dtype=dtype), samplerate=samplerate, blocking=True,
                device=chosen_output_device_id)
    except Exception as e:
        print(f"Ошибка sd.play: {e}", file=sys.stderr)


async def speak_text(text_to_speak):
    global piper_sample_rate, PIPER_EXE_PATH, VOICE_MODEL_PATH, tts_lock
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
                    process.kill(); await process.wait()
                except Exception as kill_e:
                    print(f"Ошибка убийства piper: {kill_e}", file=sys.stderr)
        except FileNotFoundError:
            print(f"КРИТИКА: Не найден piper.exe: {PIPER_EXE_PATH}", file=sys.stderr)
        except Exception as e:
            print(f"Ошибка вызова piper: {e}", file=sys.stderr)

        if audio_bytes:
            try:
                await asyncio.to_thread(play_raw_audio_sync, audio_bytes, piper_sample_rate)
            except Exception as e_play:
                print(f"Ошибка play audio: {e_play}", file=sys.stderr)
            print(f"[TTS] Озвучка завершена.")


def toggle_stt():
    global stt_enabled, audio_queue
    stt_enabled = not stt_enabled
    status = "ВКЛ" if stt_enabled else "ВЫКЛ"
    print(f"\n--- STT {status} ---")
    if not stt_enabled:
        with audio_queue.mutex: audio_queue.queue.clear()


def toggle_chat_interaction():
    global chat_interaction_enabled
    chat_interaction_enabled = not chat_interaction_enabled
    status = "ВКЛ" if chat_interaction_enabled else "ВЫКЛ"
    print(f"\n=== Реакция на чат {status} ===")


class SimpleBot(twitchio.Client):
    def __init__(self, token, initial_channels):
        super().__init__(token=token, initial_channels=initial_channels)
        self.target_channel_name = initial_channels[0] if initial_channels else None

    async def event_ready(self):
        print(f'IRC Подключен как | {self.nick}')
        if self.connected_channels:
            channel_obj = self.get_channel(self.target_channel_name)
            if channel_obj:
                print(f'Присоединен к | {channel_obj.name}\n--- Бот готов ---')
            else:
                print(f'ОШИБКА: Канал {self.target_channel_name} не найден.')
        else:
            print(f'НЕ УДАЛОСЬ присоединиться к {self.target_channel_name}.')

    async def event_message(self, message):
        if message.echo: return

        global chat_interaction_enabled, is_processing_response, last_activity_time
        global BOT_NAME_FOR_CHECK, OBS_OUTPUT_FILE, stt_enabled, audio_queue
        global together, TOGETHER_API_KEY  # Проверяем флаг

        if not chat_interaction_enabled: return
        if message.channel.name != self.target_channel_name: return
        if not together or not TOGETHER_API_KEY:
            print("Ответ невозможен (чат): Together AI не настроен.")
            return

        content_lower = message.content.lower()
        trigger_parts = [p.lower() for p in BOT_NAME_FOR_CHECK.split() if len(p) > 2]
        mentioned = any(trig in content_lower for trig in trigger_parts)
        highlighted = message.tags.get('msg-id') == 'highlighted-message'
        if not mentioned and not highlighted: return

        current_time = datetime.datetime.now().strftime('%H:%M:%S')
        if is_processing_response:
            print(f"[{current_time}] Бот занят (чат). Игнор {message.author.name}.")
            return

        stt_was_on = False
        try:
            is_processing_response = True
            last_activity_time = time.time()
            print(f"[{current_time}] {message.author.name}: {message.content}")
            stt_was_on = stt_enabled
            if stt_enabled: stt_enabled = False
            with audio_queue.mutex:
                audio_queue.queue.clear()
            try:
                open(OBS_OUTPUT_FILE, 'w').close()
            except Exception as e:
                print(f"[{current_time}] Ошибка очистки OBS: {e}")

            llm_response = await get_togetherai_response(f"(Чат от {message.author.name}): {message.content}")
            if llm_response:
                print(f"[{current_time}] Ответ Together AI (чат): {llm_response}")
                try:
                    open(OBS_OUTPUT_FILE, 'w', encoding='utf-8').write(llm_response)
                except Exception as e:
                    print(f"[{current_time}] Ошибка записи в OBS: {e}")
                await speak_text(llm_response)
                with audio_queue.mutex:
                    audio_queue.queue.clear()
                last_activity_time = time.time()
            else:
                print(f"[{current_time}] Нет ответа Together AI для {message.author.name}.")
            if stt_was_on: stt_enabled = True
        except Exception as e:
            print(f"[{current_time}] КРИТ. ОШИБКА event_message: {e}")
            if stt_was_on: stt_enabled = True
        finally:
            is_processing_response = False


async def stt_processing_loop():
    global audio_queue, recording_active, stt_model, stt_enabled, is_processing_response
    silence_blocks = int(VAD_SILENCE_TIMEOUT_MS / (BLOCKSIZE / SOURCE_SAMPLE_RATE * 1000))
    min_speech = int(VAD_MIN_SPEECH_MS / (BLOCKSIZE / SOURCE_SAMPLE_RATE * 1000))
    pad_blocks = int(VAD_SPEECH_PAD_MS / (BLOCKSIZE / SOURCE_SAMPLE_RATE * 1000))
    is_speaking, silence_count, speech_buf, pad_buf = False, 0, [], []

    while recording_active.is_set():
        if not stt_enabled or is_processing_response:
            if is_processing_response and is_speaking:
                is_speaking, speech_buf, pad_buf, silence_count = False, [], [], 0
            await asyncio.sleep(0.1)
            continue
        try:
            block = audio_queue.get_nowait()
        except queue.Empty:
            await asyncio.sleep(0.01)
            continue
        rms = np.sqrt(np.mean(block ** 2))
        pad_buf.append(block)
        if len(pad_buf) > pad_blocks * 2: pad_buf.pop(0)
        if rms > VAD_ENERGY_THRESHOLD:
            if not is_speaking:
                is_speaking = True
                speech_buf = pad_buf[-pad_blocks:].copy()
            speech_buf.append(block)
            silence_count = 0
        elif is_speaking:
            silence_count += 1
            speech_buf.append(block)
            if silence_count >= silence_blocks:
                if len(speech_buf) > min_speech + pad_blocks:
                    asyncio.create_task(process_recognized_speech(speech_buf[:-silence_blocks], "STT"))
                is_speaking, speech_buf, pad_buf, silence_count = False, [], [], 0


async def process_recognized_speech(audio_buffer_list, source_id="STT"):
    global is_processing_response, stt_enabled, audio_queue, last_activity_time
    global OBS_OUTPUT_FILE, together, TOGETHER_API_KEY

    if not together or not TOGETHER_API_KEY:
        print("Ответ невозможен (STT): Together AI не настроен.")
        return

    current_time = datetime.datetime.now().strftime('%H:%M:%S')
    full_audio = np.concatenate(audio_buffer_list, axis=0)
    mono = full_audio.mean(axis=1) if SOURCE_CHANNELS > 1 else full_audio
    resampled = resample_audio(mono, SOURCE_SAMPLE_RATE, TARGET_SAMPLE_RATE)
    recognized_text = None
    if resampled is not None and resampled.size > 0:
        recognized_text = await asyncio.to_thread(transcribe_audio_faster_whisper, resampled)

    if not recognized_text: return

    last_activity_time = time.time()
    print(f"STT Распознано ({source_id}): {recognized_text}")

    if is_processing_response: return

    stt_was_initially_enabled = stt_enabled
    try:
        is_processing_response = True
        if stt_enabled: stt_enabled = False
        with audio_queue.mutex:
            audio_queue.queue.clear()
        try:
            open(OBS_OUTPUT_FILE, 'w').close()
        except Exception as e:
            print(f"[{current_time}] Ошибка очистки OBS: {e}")

        llm_response = await get_togetherai_response(f"(Голос Степана): {recognized_text}")
        if llm_response:
            print(f"[{current_time}] Ответ Together AI ({source_id}): {llm_response}")
            try:
                open(OBS_OUTPUT_FILE, 'w', encoding='utf-8').write(llm_response)
            except Exception as e:
                print(f"[{current_time}] Ошибка записи в OBS: {e}")
            await speak_text(llm_response)
            with audio_queue.mutex:
                audio_queue.queue.clear()
            last_activity_time = time.time()
        else:
            print(f"[{current_time}] Нет ответа Together AI ({source_id}).")
    except Exception as e:
        print(f"[{current_time}] КРИТ. ОШИБКА process_speech ({source_id}): {e}")
    finally:
        is_processing_response = False
        stt_enabled = stt_was_initially_enabled


async def monologue_loop():
    global last_activity_time, recording_active, stt_enabled, BOT_NAME_FOR_CHECK
    global audio_queue, is_processing_response, chat_interaction_enabled
    global together, TOGETHER_API_KEY

    while recording_active.is_set():
        await asyncio.sleep(15)
        if is_processing_response or not chat_interaction_enabled or not together or not TOGETHER_API_KEY:
            continue
        if time.time() - last_activity_time > INACTIVITY_THRESHOLD_SECONDS:
            current_time = datetime.datetime.now().strftime('%H:%M:%S')
            if is_processing_response or not chat_interaction_enabled or not together or not TOGETHER_API_KEY:
                continue

            stt_was_initially_enabled = stt_enabled
            try:
                is_processing_response = True
                if stt_enabled: stt_enabled = False
                with audio_queue.mutex:
                    audio_queue.queue.clear()
                prompt = f"Сгенерируй короткую (1-2 предл.) реплику от {BOT_NAME_FOR_CHECK} для заполнения тишины."
                llm_response = await get_togetherai_response(prompt)
                if llm_response:
                    print(f"[{current_time}] Монолог: {llm_response}")
                    try:
                        open(OBS_OUTPUT_FILE, 'w', encoding='utf-8').write(llm_response)
                    except Exception as e:
                        print(f"[{current_time}] Ошибка записи монолога в OBS: {e}")
                    await speak_text(llm_response)
                    with audio_queue.mutex:
                        audio_queue.queue.clear()
                    last_activity_time = time.time()
                else:
                    print(f"[{current_time}] Монолог не сгенерирован.")
            except Exception as e:
                print(f"[{current_time}] КРИТ. ОШИБКА monologue_loop: {e}")
            finally:
                is_processing_response = False
                stt_enabled = stt_was_initially_enabled


def hotkey_listener_thread():
    stt_hotkey = 'ctrl+;'
    chat_hotkey = "ctrl+'"
    reg_stt, reg_chat = False, False
    try:
        print(f"\nХоткей STT: '{stt_hotkey}', Хоткей Чата: '{chat_hotkey}'")
        keyboard.add_hotkey(stt_hotkey, toggle_stt);
        reg_stt = True
        keyboard.add_hotkey(chat_hotkey, toggle_chat_interaction);
        reg_chat = True
        while recording_active.is_set(): time.sleep(0.5)
    except ImportError:
        print("\nОШИБКА: 'keyboard' не найден.");
        return
    except Exception as e:
        if not (isinstance(e, ValueError) and "is not mapped" in str(e)):
            print(f"\nОшибка hotkey_listener: {e}");
        else:
            print(f"\nПредупреждение: Не удалось зарегистрировать хоткей: {e}")
    finally:
        try:
            if reg_stt: keyboard.remove_hotkey(stt_hotkey)
            if reg_chat: keyboard.remove_hotkey(chat_hotkey)
        except Exception:  # nosec B110
            pass  # nosec B110
        print("Поток хоткеев завершен.")


async def main_async():
    global recording_active, together, TOGETHER_API_KEY
    print("Запуск AI Twitch Bot...");
    if not TWITCH_ACCESS_TOKEN: print("ОШИБКА: Нет TWITCH_ACCESS_TOKEN!"); return

    if not together or not TOGETHER_API_KEY:
        print("ОШИБКА: Together AI не настроен (API ключ или библиотека). Бот не сможет отвечать.")

    client = SimpleBot(token=TWITCH_ACCESS_TOKEN, initial_channels=[TWITCH_CHANNEL])
    twitch_task = asyncio.create_task(client.start(), name="TwitchIRC")
    stt_task = asyncio.create_task(stt_processing_loop(), name="STTLoop")
    monologue_task = asyncio.create_task(monologue_loop(), name="MonologueLoop")
    active_tasks = {twitch_task, stt_task, monologue_task}

    while recording_active.is_set() and active_tasks:
        done, pending = await asyncio.wait(active_tasks, return_when=asyncio.FIRST_COMPLETED)
        active_tasks = pending
        for task in done:
            try:
                exc = task.exception()
                if exc:
                    print(f"\n!!! ОШИБКА Задачи {task.get_name()}: {exc} !!!", file=sys.stderr)
                    if task in [twitch_task, stt_task]: recording_active.clear()
                elif task.cancelled():
                    pass  # nosec B110
                else:
                    pass  # nosec B110
            except asyncio.CancelledError:
                pass  # nosec B110
            except Exception as e:
                print(f"Ошибка проверки задачи {task.get_name()}: {e}")
        if not recording_active.is_set() or not active_tasks: break
        await asyncio.sleep(1)

    current_tasks = asyncio.all_tasks()
    tasks_to_cancel = [t for t in current_tasks if not t.done() and t is not asyncio.current_task()]
    if tasks_to_cancel:
        for task in tasks_to_cancel: task.cancel()
        await asyncio.gather(*tasks_to_cancel, return_exceptions=True)
    if client and client.is_connected(): await client.close()


if __name__ == "__main__":
    print("-" * 40 + "\nЗапуск программы...\n" + "-" * 40)
    choose_audio_output_device()

    if not all([TWITCH_ACCESS_TOKEN, TWITCH_CHANNEL]):
        print("ОШИБКА: Заполните .env (Twitch)!")
        sys.exit(1)
    if not TOGETHER_API_KEY:
        print("ОШИБКА: TOGETHER_API_KEY не указан в .env!")
        # Можно решить, продолжать ли, если TTS/STT могут работать локально
        # sys.exit(1)

    stt_model = None
    try:
        from faster_whisper import WhisperModel

        stt_model = WhisperModel(STT_MODEL_SIZE, device=STT_DEVICE, compute_type=STT_COMPUTE_TYPE)
    except ImportError:
        print("ОШИБКА: faster-whisper не установлен.")
    except Exception as e:
        print(f"Критическая ошибка загрузки faster-whisper: {e}")

    piper_sample_rate = None
    try:
        if os.path.exists(VOICE_CONFIG_PATH):
            with open(VOICE_CONFIG_PATH, 'r', encoding='utf-8') as f:
                piper_sample_rate = json.load(f).get('audio', {}).get('sample_rate')
            if not piper_sample_rate: print(f"ОШИБКА: Не найден 'sample_rate' в {VOICE_CONFIG_PATH}")
        else:
            print(f"ОШИБКА: Не найден JSON конфиг голоса: {os.path.abspath(VOICE_CONFIG_PATH)}")
        if not all([os.path.exists(PIPER_EXE_PATH), os.path.exists(VOICE_MODEL_PATH), piper_sample_rate]):
            piper_sample_rate = None  # TTS не будет работать
    except Exception as e:
        print(f"Критическая ошибка инициализации Piper TTS: {e}")
        piper_sample_rate = None

    if not together or not TOGETHER_API_KEY: print("Предупреждение: Together AI не настроен. Бот не сможет отвечать.")
    if not stt_model: print("Предупреждение: STT не загружена.")
    if not piper_sample_rate: print("Предупреждение: TTS не инициализирован.")

    default_mic, mic_name = None, "N/A";
    try:
        dev_info = sd.query_devices(kind='input')
        if isinstance(dev_info, dict) and 'index' in dev_info:
            default_mic, mic_name = dev_info['index'], dev_info.get('name', 'N/A')
        elif hasattr(sd.default, 'device') and isinstance(sd.default.device, (list, tuple)) and len(
                sd.default.device) > 0 and sd.default.device[0] != -1:
            default_mic = sd.default.device[0]
            mic_info = sd.query_devices(default_mic)
            if isinstance(mic_info, dict): mic_name = mic_info.get('name', 'N/A')
    except Exception as e_mic:
        print(f"Ошибка определения микрофона: {e_mic}.")
    print(f"Микрофон: {default_mic} ({mic_name}) | SR={SOURCE_SAMPLE_RATE}, Ch={SOURCE_CHANNELS}\n" + "-" * 40)

    recording_active.set()
    recorder = threading.Thread(target=audio_recording_thread, args=(default_mic,), daemon=True, name="AudioRecorder");
    recorder.start()
    hotkeys_thread = None
    if 'keyboard' in sys.modules:
        hotkeys_thread = threading.Thread(target=hotkey_listener_thread, daemon=True, name="HotkeyListener");
        hotkeys_thread.start()
    else:
        print("Хоткеи не работают ('keyboard' не найден).")

    loop = asyncio.new_event_loop();
    asyncio.set_event_loop(loop)
    main_task_instance = None
    try:
        main_task_instance = loop.create_task(main_async(), name="MainLoop")
        loop.run_until_complete(main_task_instance)
    except KeyboardInterrupt:
        recording_active.clear()
    except Exception as e_loop:
        print(f"Критическая ошибка главного цикла: {e_loop}"); recording_active.clear()
    finally:
        recording_active.clear()
        threads_to_join = [t for t in [recorder, hotkeys_thread] if t and t.is_alive()]
        for t in threads_to_join: t.join(timeout=2.0)

        if main_task_instance and not main_task_instance.done(): main_task_instance.cancel()
        async_tasks_to_wait = [t for t in asyncio.all_tasks(loop=loop) if not t.done()]
        if async_tasks_to_wait:
            try:
                loop.run_until_complete(
                    asyncio.wait_for(asyncio.gather(*async_tasks_to_wait, return_exceptions=True), timeout=2.0))
            except asyncio.TimeoutError:
                pass  # nosec B110
            except Exception:
                pass  # nosec B110
        try:
            loop.run_until_complete(loop.shutdown_asyncgens())
        except Exception:
            pass  # nosec B110
        finally:
            if not loop.is_closed(): loop.close()
        print("-" * 40 + "\nПрограмма завершена.\n" + "-" * 40)