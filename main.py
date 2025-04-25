import os
import sys

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

# --- Загрузка DLL ---
try:
    cudnn_path = os.getenv('CUDNN_PATH', "C:\\Program Files\\NVIDIA\\CUDNN\\v9.8\\bin\\12.8")
    if os.path.exists(cudnn_path):
        os.add_dll_directory(cudnn_path)
        print(f"Добавлен путь CUDNN: {cudnn_path}")
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
            print(f"Успешно загружена DLL: {lib}")
            loaded_libs += 1
        except FileNotFoundError:
            pass
        except Exception as e_dll:
            print(f"Предупреждение: Ошибка загрузки {lib}: {e_dll}")
    if loaded_libs == 0: print("Предупреждение: Не удалось загрузить ни одну DLL CUDNN.")
except ImportError:
    print("Предупреждение: Библиотека ctypes не найдена. Пропуск загрузки CUDNN DLL.")
except Exception as e:
    print(f"Ошибка настройки DLL: {e}")

# --- Загрузка переменных окружения ---
load_dotenv()

# --- Получение настроек из .env ---
TWITCH_ACCESS_TOKEN = os.getenv('TWITCH_ACCESS_TOKEN')
TWITCH_BOT_NICK = os.getenv('TWITCH_BOT_NICK')
TWITCH_CHANNEL = os.getenv('TWITCH_CHANNEL')
TWITCH_CLIENT_ID = os.getenv('TWITCH_CLIENT_ID')
TWITCH_CLIENT_SECRET = os.getenv('TWITCH_CLIENT_SECRET')
TWITCH_REFRESH_TOKEN = os.getenv('TWITCH_REFRESH_TOKEN')

# --- Настройки Ollama ---
OLLAMA_API_URL = os.getenv('OLLAMA_API_URL', "http://localhost:11434/api/chat")
OLLAMA_MODEL_NAME = os.getenv('OLLAMA_MODEL_NAME', "llama3:8b-instruct-q4_K_M")

# --- Настройки Piper TTS ---
PIPER_EXE_PATH = os.getenv('PIPER_EXE_PATH', 'piper_tts_bin/piper.exe')
VOICE_MODEL_PATH = os.getenv('PIPER_VOICE_MODEL_PATH', 'voices/ru_RU-ruslan-medium.onnx')
VOICE_CONFIG_PATH = os.getenv('PIPER_VOICE_CONFIG_PATH', 'voices/ru_RU-ruslan-medium.onnx.json')

# --- Настройка файла для вывода текста в OBS ---
OBS_OUTPUT_FILE = "obs_ai_response.txt"

# --- Настройки Faster Whisper ---
STT_MODEL_SIZE = "large-v3"
STT_DEVICE = "cuda"
STT_COMPUTE_TYPE = "int8"

# --- Настройки Аудио STT ---
SOURCE_SAMPLE_RATE = 48000
SOURCE_CHANNELS = 2
TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1
TARGET_DTYPE = 'float32'
BLOCKSIZE = int(SOURCE_SAMPLE_RATE * 0.1)

# --- Константы для VAD ---
VAD_ENERGY_THRESHOLD = 0.005
VAD_SPEECH_PAD_MS = 200
VAD_MIN_SPEECH_MS = 250
VAD_SILENCE_TIMEOUT_MS = 1200

# --- Глобальные переменные ---
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

# --- Загрузка модели Faster Whisper ---
stt_model = None
try:
    from faster_whisper import WhisperModel

    print(f"Загрузка/конвертация faster-whisper '{STT_MODEL_SIZE}' ({STT_DEVICE}, {STT_COMPUTE_TYPE})...")
    stt_model = WhisperModel(STT_MODEL_SIZE, device=STT_DEVICE, compute_type=STT_COMPUTE_TYPE)
    print("Модель faster-whisper успешно загружена.")
except ImportError:
    print("ОШИБКА: faster-whisper не установлен."); stt_model = None
except Exception as e:
    print(f"Критическая ошибка загрузки faster-whisper: {e}"); stt_model = None

# --- Чтение Sample Rate из конфига голоса Piper ---
piper_sample_rate = None
try:
    if os.path.exists(VOICE_CONFIG_PATH):
        with open(VOICE_CONFIG_PATH, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
            piper_sample_rate = config_data.get('audio', {}).get('sample_rate')
        if not piper_sample_rate:
            print(f"ОШИБКА: Не найден 'sample_rate' в {VOICE_CONFIG_PATH}")
        else:
            print(f"Голос Piper: '{VOICE_MODEL_PATH}', SR: {piper_sample_rate}")
    else:
        print(f"ОШИБКА: Не найден JSON конфиг голоса: {os.path.abspath(VOICE_CONFIG_PATH)}")
    if not all([os.path.exists(PIPER_EXE_PATH), os.path.exists(VOICE_MODEL_PATH), piper_sample_rate]):
        print("ОШИБКА: TTS не будет работать.");
        piper_sample_rate = None
except Exception as e:
    print(f"Критическая ошибка инициализации Piper TTS: {e}"); piper_sample_rate = None

# --- Системный промпт для ИИ ---
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

# --- Проверка имени бота при запуске ---
BOT_NAME_FOR_CHECK = "Джордж Дроид"
prompt_lines = SYSTEM_PROMPT.split('\n', 2)
if len(prompt_lines) > 1 and prompt_lines[0].startswith("## Твоя Личность:"):
    potential_name = prompt_lines[0].replace("## Твоя Личность:", "").strip()
    if potential_name:
        BOT_NAME_FOR_CHECK = potential_name
        print(f"Имя бота для триггеров в чате установлено: '{BOT_NAME_FOR_CHECK}'")
    else:
        print(f"Предупреждение: Не удалось извлечь имя бота из SYSTEM_PROMPT. Используется '{BOT_NAME_FOR_CHECK}'.")
else:
    print(
        f"Предупреждение: Первая строка SYSTEM_PROMPT не соответствует формату. Используется имя '{BOT_NAME_FOR_CHECK}'.")


# --- Вспомогательные функции ---

def resample_audio(audio_data: np.ndarray, input_rate: int, target_rate: int) -> np.ndarray:
    if input_rate == target_rate:
        return audio_data.astype(np.float32)
    try:
        duration = audio_data.shape[0] / input_rate
        new_num_samples = int(duration * target_rate)
        resampled_audio = scipy.signal.resample(audio_data, new_num_samples)
        return resampled_audio.astype(np.float32)
    except Exception as e:
        print(f"Ошибка при передискретизации: {e}", file=sys.stderr)
        return np.array([], dtype=np.float32)


def audio_recording_thread(device_index=None):
    global audio_queue, recording_active, stt_enabled, is_processing_response

    def audio_callback(indata, frames, time, status):
        if status: print(f"Статус аудиопотока: {status}", file=sys.stderr)
        if recording_active.is_set() and stt_enabled and not is_processing_response:
            try:
                audio_queue.put_nowait(indata.copy())
            except queue.Full:
                pass

    stream = None
    try:
        print(f"Поток записи: Запуск аудиопотока (устройство: {device_index or 'default'})...")
        stream = sd.InputStream(
            device=device_index, samplerate=SOURCE_SAMPLE_RATE, channels=SOURCE_CHANNELS,
            dtype=TARGET_DTYPE, blocksize=BLOCKSIZE, callback=audio_callback)
        with stream:
            while recording_active.is_set(): time.sleep(0.1)
    except sd.PortAudioError as e:
        print(f"КРИТИЧЕСКАЯ ОШИБКА PortAudio в потоке записи: {e}", file=sys.stderr)
        print("Возможные причины: Неверное устройство, занято другим приложением, проблемы с драйверами.",
              file=sys.stderr)
    except Exception as e:
        print(f"Критическая ошибка в потоке записи аудио: {e}", file=sys.stderr)
    finally:
        if stream is not None and not stream.closed:
            stream.stop()
            stream.close()
        print("Поток записи: Аудиопоток остановлен.")


def transcribe_audio_faster_whisper(audio_np_array):
    global stt_model
    if stt_model is None or not isinstance(audio_np_array, np.ndarray) or audio_np_array.size == 0: return None
    try:
        segments, info = stt_model.transcribe(audio_np_array, language="ru", word_timestamps=False)
        full_text = "".join(segment.text for segment in segments).strip()
        return full_text
    except Exception as e:
        print(f"Ошибка во время распознавания faster-whisper: {e}", file=sys.stderr)
        return None


async def get_ollama_response(user_message):
    global conversation_history, OLLAMA_API_URL, OLLAMA_MODEL_NAME, SYSTEM_PROMPT
    is_monologue_request = user_message.startswith("Сгенерируй короткое")

    if is_monologue_request:
        messages_payload = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_message}]
    else:
        current_user_message = {"role": "user", "content": user_message}
        temp_history = conversation_history + [current_user_message]
        if len(temp_history) > MAX_HISTORY_LENGTH:
            temp_history = temp_history[-MAX_HISTORY_LENGTH:]
        messages_payload = [{"role": "system", "content": SYSTEM_PROMPT}] + temp_history

    payload = {"model": OLLAMA_MODEL_NAME, "messages": messages_payload, "stream": False}

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(OLLAMA_API_URL, json=payload, timeout=60) as response:
                if response.status == 200:
                    response_data = await response.json()
                    llm_content = response_data.get('message', {}).get('content')
                    if llm_content:
                        if not is_monologue_request:
                            conversation_history.append({"role": "user", "content": user_message})
                            conversation_history.append({"role": "assistant", "content": llm_content})
                            if len(conversation_history) > MAX_HISTORY_LENGTH * 2:
                                conversation_history = conversation_history[-(MAX_HISTORY_LENGTH * 2):]
                        return llm_content.strip()
                    else:
                        print(f"Ollama вернула пустой ответ: {response_data}", file=sys.stderr)
                        return None
                else:
                    error_text = await response.text()
                    print(f"Ошибка Ollama: Статус {response.status}, Ответ: {error_text}", file=sys.stderr)
                    return None
    except asyncio.TimeoutError:
        print("Ошибка Ollama: Таймаут запроса.", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Ошибка при обращении к Ollama: {e}", file=sys.stderr)
        return None


def play_raw_audio_sync(audio_bytes, samplerate, dtype='int16'):
    if not audio_bytes or not samplerate: return
    try:
        audio_data = np.frombuffer(audio_bytes, dtype=dtype)
        sd.play(audio_data, samplerate=samplerate, blocking=True)
    except Exception as e:
        print(f"Ошибка при воспроизведении sd.play: {e}", file=sys.stderr)


async def speak_text(text_to_speak):
    global piper_sample_rate, PIPER_EXE_PATH, VOICE_MODEL_PATH, tts_lock
    if not piper_sample_rate or not os.path.exists(PIPER_EXE_PATH) or not os.path.exists(VOICE_MODEL_PATH):
        print("TTS недоступен, пропуск озвучки.")
        return

    async with tts_lock:
        print(f"[TTS LOCK] Захвачен: \"{text_to_speak[:30]}...\"")
        command = [PIPER_EXE_PATH, '--model', VOICE_MODEL_PATH, '--output-raw']
        process = None
        audio_bytes = None
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            )
            audio_bytes, stderr_bytes = await asyncio.wait_for(process.communicate(input=text_to_speak.encode('utf-8')),
                                                               timeout=30)

            if process.returncode != 0:
                print(
                    f"Ошибка piper.exe: Exit code {process.returncode}\nStderr: {stderr_bytes.decode('utf-8', errors='ignore')}",
                    file=sys.stderr)
                audio_bytes = None
            elif not audio_bytes:
                print(f"Ошибка: piper.exe не вернул аудио.\nStderr: {stderr_bytes.decode('utf-8', errors='ignore')}",
                      file=sys.stderr)

        except asyncio.TimeoutError:
            print("Ошибка TTS: Таймаут при ожидании piper.exe", file=sys.stderr)
            if process and process.returncode is None:
                try:
                    print("Попытка принудительно завершить piper.exe...")
                    process.kill();
                    await process.wait()
                    print("piper.exe завершен.")
                except ProcessLookupError:
                    pass
                except Exception as kill_e:
                    print(f"Ошибка при остановке piper.exe: {kill_e}", file=sys.stderr)
        except FileNotFoundError:
            print(f"Критическая ошибка TTS: Не найден piper.exe по пути {PIPER_EXE_PATH}", file=sys.stderr)
        except Exception as e:
            print(f"Ошибка при вызове piper.exe: {e}", file=sys.stderr)

        if audio_bytes:
            try:
                await asyncio.to_thread(play_raw_audio_sync, audio_bytes, piper_sample_rate)
            except Exception as e_play:
                print(f"Ошибка при воспроизведении аудио через asyncio.to_thread: {e_play}", file=sys.stderr)

        print(f"[TTS LOCK] Освобожден: \"{text_to_speak[:30]}...\"")


def toggle_stt():
    global stt_enabled, audio_queue, is_processing_response
    # Убрана проверка is_processing_response
    stt_enabled = not stt_enabled
    status_text = "ВКЛЮЧЕНО" if stt_enabled else "ВЫКЛЮЧЕНО"
    print("\n" + "-" * 30 + f"\n--- Распознавание голоса (STT) {status_text} ---\n" + "-" * 30)
    if not stt_enabled:
        with audio_queue.mutex: audio_queue.queue.clear()
        print("[INFO] Очередь аудио очищена при выключении STT хоткеем.")


# --- ВОЗВРАЩЕНА: Функция переключения чата ---
def toggle_chat_interaction():
    """Переключает состояние реакции на чат (вкл/выкл) - В ЛЮБОЙ МОМЕНТ."""
    global chat_interaction_enabled

    chat_interaction_enabled = not chat_interaction_enabled
    status_text = "ВКЛЮЧЕНО" if chat_interaction_enabled else "ВЫКЛЮЧЕНО"
    print("\n" + "=" * 30 + f"\n=== Реакция на чат Twitch {status_text} ===\n" + "=" * 30)


# --- КОНЕЦ ВОЗВРАЩЕННОЙ ФУНКЦИИ ---


# --- Twitch бот ---
class SimpleBot(twitchio.Client):
    """Класс Twitch бота."""

    def __init__(self, token, initial_channels):
        super().__init__(token=token, initial_channels=initial_channels)
        self.target_channel_name = initial_channels[0] if initial_channels else None

    async def event_ready(self):
        print(f'Подключен к Twitch IRC как | {self.nick}')
        if self.connected_channels:
            channel_obj = self.get_channel(self.target_channel_name)
            if channel_obj:
                print(
                    f'Присоединился к каналу | {channel_obj.name}\n' + '-' * 40 + '\n--------- Бот готов читать чат ---------\n' + '-' * 40)
            else:
                print(f'ОШИБКА: Не найден объект для канала {self.target_channel_name} среди подключенных.',
                      file=sys.stderr)
        else:
            print(f'НЕ УДАЛОСЬ присоединиться к каналу {self.target_channel_name}. Проверьте токен и имя канала.',
                  file=sys.stderr)

    # --- ИЗМЕНЕННАЯ event_message ---
    async def event_message(self, message):
        """Обрабатывает сообщения из чата."""
        if message.echo:
            return

        global chat_interaction_enabled  # <-- Добавлен global
        # --- Проверка флага чата ---
        if not chat_interaction_enabled:
            return  # Игнорируем, если чат выключен
        # --- Конец проверки флага чата ---

        global last_activity_time, BOT_NAME_FOR_CHECK, OBS_OUTPUT_FILE
        global stt_enabled, audio_queue, is_processing_response

        if message.channel.name != self.target_channel_name:
            return

        content_lower = message.content.lower()
        trigger_name_parts = [part.lower() for part in BOT_NAME_FOR_CHECK.split() if len(part) > 2]
        bot_name_mentioned = any(trigger in content_lower for trigger in trigger_name_parts)
        message_tags = message.tags or {}
        is_highlighted = message_tags.get('msg-id') == 'highlighted-message'

        if not bot_name_mentioned and not is_highlighted:
            return

        current_time_str = datetime.datetime.now().strftime('%H:%M:%S')

        if is_processing_response:
            print(f"[{current_time_str}] Бот занят. Игнорируется сообщение от {message.author.name}.")
            return

        original_stt_state = False  # Инициализация на случай ошибки
        stt_was_enabled = False  # Инициализация
        try:
            is_processing_response = True
            print(f"[{current_time_str}] НАЧАЛО обработки сообщения от {message.author.name}.")

            last_activity_time = time.time()
            print(f"[{current_time_str}] {message.author.name}: {message.content}")

            original_stt_state = stt_enabled  # Запоминаем здесь

            if stt_enabled:
                print("[INFO] Отключаю STT для ответа (чат).")
                stt_enabled = False
                stt_was_enabled = True
            with audio_queue.mutex:
                audio_queue.queue.clear()

            try:
                with open(OBS_OUTPUT_FILE, 'w', encoding='utf-8') as f:
                    f.write("")
            except Exception as e:
                print(f"[{current_time_str}] ОШИБКА очистки файла OBS: {e}", file=sys.stderr)

            # --- ИЗМЕНЕНО: Добавлен префикс имени пользователя ---
            llm_response_text = await get_ollama_response(
                f"(Сообщение от пользователя {message.author.name}): {message.content}")
            # --- КОНЕЦ ИЗМЕНЕНИЯ ---

            if llm_response_text:
                print(f"[{current_time_str}] Ответ Ollama: {llm_response_text}")
                try:
                    with open(OBS_OUTPUT_FILE, 'w', encoding='utf-8') as f:
                        f.write(llm_response_text)
                except Exception as e:
                    print(f"[{current_time_str}] ОШИБКА записи в файл OBS: {e}", file=sys.stderr)

                await speak_text(llm_response_text)

                with audio_queue.mutex:
                    audio_queue.queue.clear()
                last_activity_time = time.time()
                print(f"[INFO] Время активности обновлено после ответа на чат.")
            else:
                print(f"[{current_time_str}] Не удалось получить ответ от Ollama для {message.author.name}.")

            # Блок авто-восстановления STT удален (окончательно)

        except Exception as e:
            print(f"[{current_time_str}] КРИТИЧЕСКАЯ ОШИБКА в event_message: {e}", file=sys.stderr)
            try:
                # Восстанавливаем STT только если была ошибка ДО его возможного авто-отключения
                if stt_was_enabled:
                    stt_enabled = True
                    print("[INFO] STT восстановлен после ошибки в event_message.")
            except Exception as e_restore:
                print(f"Ошибка восстановления STT: {e_restore}", file=sys.stderr)
        finally:
            is_processing_response = False
            print(f"[{current_time_str}] КОНЕЦ обработки сообщения от {message.author.name}.")
    # --- КОНЕЦ ИЗМЕНЕННОЙ event_message ---


# --- Асинхронный цикл обработки аудио и STT ---
async def stt_processing_loop():
    global audio_queue, recording_active, stt_model, OBS_OUTPUT_FILE, last_activity_time
    global stt_enabled, is_processing_response

    silence_blocks_needed = int(VAD_SILENCE_TIMEOUT_MS / (BLOCKSIZE / SOURCE_SAMPLE_RATE * 1000))
    min_speech_blocks = int(VAD_MIN_SPEECH_MS / (BLOCKSIZE / SOURCE_SAMPLE_RATE * 1000))
    speech_pad_blocks = int(VAD_SPEECH_PAD_MS / (BLOCKSIZE / SOURCE_SAMPLE_RATE * 1000))
    print(
        f"VAD Настройки: Порог={VAD_ENERGY_THRESHOLD:.3f}, Мин. речь={min_speech_blocks} бл., Пауза={silence_blocks_needed} бл., Паддинг={speech_pad_blocks} бл.")

    is_speaking_vad = False
    silence_blocks_count = 0
    speech_audio_buffer = []
    buffer_for_padding = []

    print("Цикл обработки STT запущен...")
    while recording_active.is_set():
        if not stt_enabled or is_processing_response:
            if is_processing_response and is_speaking_vad:
                is_speaking_vad = False;
                speech_audio_buffer = [];
                buffer_for_padding = [];
                silence_blocks_count = 0
            await asyncio.sleep(0.1)
            continue

        try:
            block = audio_queue.get_nowait()
        except queue.Empty:
            await asyncio.sleep(0.01)
            continue

        rms = np.sqrt(np.mean(block ** 2))
        buffer_for_padding.append(block)
        if len(buffer_for_padding) > speech_pad_blocks * 2: buffer_for_padding.pop(0)

        if rms > VAD_ENERGY_THRESHOLD:
            if not is_speaking_vad: is_speaking_vad = True; speech_audio_buffer = buffer_for_padding[
                                                                                  -speech_pad_blocks:].copy()
            speech_audio_buffer.append(block);
            silence_blocks_count = 0
        else:
            if is_speaking_vad:
                silence_blocks_count += 1;
                speech_audio_buffer.append(block)
                if silence_blocks_count >= silence_blocks_needed:
                    if len(speech_audio_buffer) > min_speech_blocks:
                        final_buffer_copy = speech_audio_buffer.copy()
                        source_identifier = "STT (ветка 2: Silence Detected)"
                        is_speaking_vad = False;
                        speech_audio_buffer = [];
                        buffer_for_padding = [];
                        silence_blocks_count = 0
                        asyncio.create_task(process_recognized_speech(final_buffer_copy, source_identifier))
                    else:
                        is_speaking_vad = False;
                        speech_audio_buffer = [];
                        buffer_for_padding = [];
                        silence_blocks_count = 0

    print("Цикл обработки STT остановлен.")


# --- Функция обработки распознанной речи ---
async def process_recognized_speech(audio_buffer_list, source_id="STT"):
    global is_processing_response, stt_enabled, audio_queue, last_activity_time, OBS_OUTPUT_FILE

    current_time_str = datetime.datetime.now().strftime('%H:%M:%S')

    if is_processing_response:
        print(f"[{current_time_str}] Бот занят (проверка в process_recognized_speech). Игнорируется {source_id}.")
        return

    stt_was_enabled = False
    try:
        is_processing_response = True
        print(f"[{current_time_str}] НАЧАЛО обработки речи ({source_id}).")

        full_audio_raw = np.concatenate(audio_buffer_list, axis=0)
        mono_audio = full_audio_raw.mean(axis=1) if SOURCE_CHANNELS > 1 else full_audio_raw
        resampled_for_stt = resample_audio(mono_audio, SOURCE_SAMPLE_RATE, TARGET_SAMPLE_RATE)
        recognized_text = None
        if resampled_for_stt is not None and resampled_for_stt.size > 0:
            recognized_text = await asyncio.to_thread(transcribe_audio_faster_whisper, resampled_for_stt)

        if recognized_text:
            last_activity_time = time.time()
            print(f"STT Распознано ({source_id}): {recognized_text}")

            if stt_enabled:
                print(f"[INFO] Отключаю STT для ответа ({source_id}).")
                stt_enabled = False
                stt_was_enabled = True
            with audio_queue.mutex:
                audio_queue.queue.clear()

            try:
                with open(OBS_OUTPUT_FILE, 'w', encoding='utf-8') as f:
                    f.write("")
            except Exception as e:
                print(f"[{current_time_str}] ОШИБКА очистки файла OBS: {e}", file=sys.stderr)

            llm_response_text = await get_ollama_response(f"(Голосовое сообщение от Степана): {recognized_text}")

            if llm_response_text:
                print(f"[{current_time_str}] Ответ Ollama ({source_id}): {llm_response_text}")
                try:
                    with open(OBS_OUTPUT_FILE, 'w', encoding='utf-8') as f:
                        f.write(llm_response_text)
                except Exception as e:
                    print(f"[{current_time_str}] ОШИБКА записи в файл OBS: {e}", file=sys.stderr)

                await speak_text(llm_response_text)

                with audio_queue.mutex:
                    audio_queue.queue.clear()
                last_activity_time = time.time()
                print(f"[INFO] Время активности обновлено после ответа на чат.")
            else:
                print(f"[{current_time_str}] Не удалось получить ответ от Ollama ({source_id}).")

            # Блок авто-включения STT удален

        else:
            print(f"STT: Не удалось распознать речь или аудио было некорректным ({source_id}).")

    except Exception as e:
        print(f"[{current_time_str}] КРИТИЧЕСКАЯ ОШИБКА в process_recognized_speech ({source_id}): {e}",
              file=sys.stderr)
        try:
            if stt_was_enabled:
                stt_enabled = True
                print(f"[INFO] STT восстановлен после ошибки в process_recognized_speech ({source_id}).")
        except Exception as e_restore:
            print(f"Ошибка восстановления STT: {e_restore}", file=sys.stderr)
    finally:
        is_processing_response = False
        print(f"[{current_time_str}] КОНЕЦ обработки речи ({source_id}).")


# --- Цикл случайных монологов ---
async def monologue_loop():
    global last_activity_time, recording_active, OBS_OUTPUT_FILE, stt_enabled, BOT_NAME_FOR_CHECK
    global audio_queue, is_processing_response

    print("Цикл монологов запущен...")
    while recording_active.is_set():
        await asyncio.sleep(15)

        if is_processing_response:
            continue

        current_time_unix = time.time()
        time_since_last_activity = current_time_unix - last_activity_time

        if time_since_last_activity > INACTIVITY_THRESHOLD_SECONDS:
            current_time_str = datetime.datetime.now().strftime('%H:%M:%S')

            if is_processing_response:
                continue

            stt_was_enabled = False
            try:
                is_processing_response = True
                print(f"[{current_time_str}] НАЧАЛО обработки монолога.")
                print(
                    f"[{current_time_str}] Обнаружено бездействие ({time_since_last_activity:.0f} сек), запуск монолога...")

                if stt_enabled:
                    print("[INFO] Отключаю STT для монолога.")
                    stt_enabled = False
                    stt_was_enabled = True
                with audio_queue.mutex:
                    audio_queue.queue.clear()

                monologue_prompt = (f"Сгенерируй короткое (1-2 предложения) спонтанное размышление, интересный факт "
                                    f"или вопрос к чату от имени {BOT_NAME_FOR_CHECK}, чтобы заполнить тишину на стриме. "
                                    "Начни фразу естественно, например: 'Кстати, чат...', 'Задумался тут...', "
                                    "'А вы знали, что...', 'Степан, а ты когда-нибудь...', но НЕ как ответ на запрос "
                                    "('Хорошо, вот факт...'). Тема абсолютно случайна.")

                llm_response_text = await get_ollama_response(monologue_prompt)

                if llm_response_text:
                    print(f"[{current_time_str}] Монолог Ollama: {llm_response_text}")
                    try:
                        with open(OBS_OUTPUT_FILE, 'w', encoding='utf-8') as f:
                            f.write(llm_response_text)
                    except Exception as e:
                        print(f"[{current_time_str}] ОШИБКА записи монолога в файл OBS: {e}", file=sys.stderr)

                    await speak_text(llm_response_text)

                    with audio_queue.mutex:
                        audio_queue.queue.clear()

                    last_activity_time = time.time()
                else:
                    print(f"[{current_time_str}] Не удалось получить монолог от Ollama.")

                # Блок авто-включения STT удален

            except Exception as e:
                print(f"[{current_time_str}] КРИТИЧЕСКАЯ ОШИБКА в monologue_loop: {e}", file=sys.stderr)
                try:
                    if stt_was_enabled:
                        stt_enabled = True
                        print("[INFO] STT восстановлен после ошибки в monologue_loop.")
                except Exception as e_restore:
                    print(f"Ошибка восстановления STT: {e_restore}", file=sys.stderr)
            finally:
                is_processing_response = False
                print(f"[{current_time_str}] КОНЕЦ обработки монолога.")

    print("Цикл монологов остановлен.")


# --- Поток слушателя горячих клавиш (ИЗМЕНЕННЫЙ) ---
def hotkey_listener_thread():
    stt_hotkey = 'ctrl+;'
    chat_hotkey = "ctrl+apostrophe"  # <-- Новый хоткей

    registered_stt = False
    registered_chat = False
    try:
        print(f"\nНажмите '{stt_hotkey}' для вкл/выкл STT.")
        keyboard.add_hotkey(stt_hotkey, toggle_stt)
        registered_stt = True

        # --- Регистрация нового хоткея ---
        print(f"Нажмите '{chat_hotkey}' для вкл/выкл реакции на чат.")
        keyboard.add_hotkey(chat_hotkey,
                            toggle_chat_interaction)  # Убедись, что функция toggle_chat_interaction определена
        registered_chat = True
        # --- Конец регистрации ---

        while recording_active.is_set():
            time.sleep(0.5)

        print("Поток слушателя горячих клавиш: получен сигнал завершения (is_set() стал False).")
    except ImportError:
        print("\nОШИБКА: Библиотека 'keyboard' не найдена. Установите: pip install keyboard", file=sys.stderr)
        print("Горячие клавиши работать не будут.")
    except Exception as e_hk_thread:
        print(f"\nОшибка в потоке слушателя горячих клавиш: {e_hk_thread}", file=sys.stderr)
        if "root privileges" in str(e_hk_thread).lower() or "permission denied" in str(e_hk_thread).lower():
            print(
                "Подсказка: На Linux/MacOS для глобальных хоткеев могут требоваться права суперпользователя (sudo) или разрешения доступа.")
    finally:
        try:
            # --- Удаление обоих хоткеев ---
            if registered_stt:
                keyboard.remove_hotkey(stt_hotkey)
                print(f"Хоткей '{stt_hotkey}' удален.")
            if registered_chat:
                keyboard.remove_hotkey(chat_hotkey)  # <-- Удаляем новый хоткей
                print(f"Хоткей '{chat_hotkey}' удален.")
            # --- Конец удаления ---
        except Exception as e_remove:
            print(f"Предупреждение при удалении хоткеев: {e_remove}", file=sys.stderr)
            pass
        print("Поток слушателя горячих клавиш завершен.")


# --- КОНЕЦ ИЗМЕНЕННОГО hotkey_listener_thread ---


# --- Основная асинхронная функция запуска ---
async def main_async():
    print("Запуск основного Twitch бота, цикла STT и цикла монологов...")
    client = SimpleBot(token=TWITCH_ACCESS_TOKEN, initial_channels=[TWITCH_CHANNEL])

    twitch_task = asyncio.create_task(client.start(), name="TwitchTask")
    stt_task = asyncio.create_task(stt_processing_loop(), name="STTTask")
    monologue_task = asyncio.create_task(monologue_loop(), name="MonologueTask")
    all_main_tasks = [twitch_task, stt_task, monologue_task]

    done, pending = await asyncio.wait(all_main_tasks, return_when=asyncio.FIRST_COMPLETED)

    print("\nОдна из основных задач завершилась или получена ошибка.")
    for task in done:
        try:
            if task.exception():
                print(f"Задача {task.get_name()} завершилась с ошибкой: {task.exception()}", file=sys.stderr)
            elif task.cancelled():
                print(f"Задача {task.get_name()} была отменена.")
            else:
                print(f"Задача {task.get_name()} успешно завершилась.")
        except asyncio.InvalidStateError:
            print(f"Задача {task.get_name()} в неопределенном состоянии при проверке.")

    print("Отмена оставшихся асинхронных задач...")
    cancelled_count = 0
    for task in pending:
        if task and not task.done():
            task.cancel();
            cancelled_count += 1
    print(f"Отменено {cancelled_count} задач.")

    if pending:
        await asyncio.wait(pending, timeout=1.0)


# --- Точка входа в программу ---
if __name__ == "__main__":
    print("-" * 40)
    if not TWITCH_ACCESS_TOKEN or not TWITCH_CHANNEL:
        print("КРИТИЧЕСКАЯ ОШИБКА: Не заданы TWITCH_ACCESS_TOKEN или TWITCH_CHANNEL в .env файле!", file=sys.stderr)
        sys.exit(1)
    if not OLLAMA_API_URL: print("Предупреждение: Не задан OLLAMA_API_URL, используется значение по умолчанию.")
    if not piper_sample_rate: print("Предупреждение: TTS (Piper) не инициализирован.")
    if not stt_model: print("Предупреждение: STT модель (Whisper) не загружена.")
    print("-" * 40)

    default_mic_index = None
    mic_name = "N/A"
    try:
        print("Доступные устройства записи (Input):")
        print(sd.query_devices())
        default_mic_info = sd.query_devices(kind='input')
        if isinstance(default_mic_info, dict) and 'index' in default_mic_info:
            default_mic_index = default_mic_info['index']
            mic_name = default_mic_info.get('name', mic_name)
        else:
            if sd.default.device[0] != -1:
                default_mic_index = sd.default.device[0]
                mic_info = sd.query_devices(device=default_mic_index)
                mic_name = mic_info.get('name', mic_name) if isinstance(mic_info, dict) else mic_name
            else:
                print("Не удалось определить устройство записи по умолчанию.", file=sys.stderr)
        print("-" * 40)
        print(f"Выбрано устройство записи: индекс {default_mic_index} ({mic_name})")
        print(f"Параметры записи: SR={SOURCE_SAMPLE_RATE}, Каналы={SOURCE_CHANNELS}")
        print("-" * 40)
    except Exception as e:
        print(f"Ошибка определения микрофона: {e}. Используем системное по умолчанию (None).", file=sys.stderr)
        print("-" * 40)

    recording_active.set()

    recorder_thread = threading.Thread(target=audio_recording_thread, args=(default_mic_index,), daemon=True,
                                       name="AudioRecorder")
    recorder_thread.start()
    print("Поток записи аудио запущен.")

    hotkey_thread = None
    if 'keyboard' in sys.modules:
        hotkey_thread = threading.Thread(target=hotkey_listener_thread, daemon=True, name="HotkeyListener")
        hotkey_thread.start()
    else:
        print("Поток слушателя горячих клавиш не запущен (библиотека keyboard не найдена).")

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    main_task = None
    try:
        main_task = loop.create_task(main_async(), name="MainAsync")
        loop.run_until_complete(main_task)
    except KeyboardInterrupt:
        print("\nПолучен сигнал Ctrl+C...")
        if main_task and not main_task.done():
            main_task.cancel()
            try:
                loop.run_until_complete(main_task)
            except asyncio.CancelledError:
                pass
    except asyncio.CancelledError:
        print("Главный цикл был отменен...")
    finally:
        print("Начало процедуры graceful shutdown...")

        print("Остановка фоновых потоков...")
        recording_active.clear()

        active_threads = [t for t in [recorder_thread, hotkey_thread] if t and t.is_alive()]
        if active_threads:
            print(f"Ожидание завершения {len(active_threads)} потоков...")
            for t in active_threads:
                try:
                    t.join(timeout=3.0)
                except Exception as e_join:
                    print(f"Ошибка при ожидании потока {t.name}: {e_join}", file=sys.stderr)
                if t.is_alive(): print(f"Предупреждение: Поток {t.name} не завершился за таймаут.", file=sys.stderr)
            print("Ожидание потоков завершено (или таймаут).")

        print("Завершение asyncio...")
        all_asyncio_tasks = asyncio.all_tasks(loop=loop)
        if all_asyncio_tasks:
            tasks_to_cancel = [task for task in all_asyncio_tasks if not task.done()]
            if tasks_to_cancel:
                print(f"Отмена {len(tasks_to_cancel)} оставшихся asyncio задач...")
                for task in tasks_to_cancel: task.cancel()
                loop.run_until_complete(asyncio.gather(*tasks_to_cancel, return_exceptions=True))
                print("Сбор отмененных задач завершен.")
            else:
                print("Нет активных задач для отмены.")

        try:
            print("Завершение асинхронных генераторов...")
            loop.run_until_complete(loop.shutdown_asyncgens())
            print("Асинхронные генераторы завершены.")
        except Exception as e_gens:
            print(f"Ошибка при shutdown_asyncgens: {e_gens}", file=sys.stderr)
        finally:
            if not loop.is_closed():
                loop.close()
                print("Цикл asyncio закрыт.")

        print("-" * 40 + "\nПрограмма штатно завершена.\n" + "-" * 40)
        sys.exit(0)