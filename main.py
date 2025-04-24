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

# --- Загрузка DLL (если нужно) ---
try:
    # Укажи правильный путь к папке bin твоей версии CUDNN
    cudnn_path = os.getenv('CUDNN_PATH', "C:\\Program Files\\NVIDIA\\CUDNN\\v9.8\\bin\\12.8")
    if os.path.exists(cudnn_path):
        os.add_dll_directory(cudnn_path)
        print(f"Добавлен путь CUDNN: {cudnn_path}")
    else:
        print(f"Предупреждение: Путь CUDNN не найден: {cudnn_path}")

    import ctypes

    # Попытка загрузить необходимые DLL
    libs_to_try = ["cudnn_ops64_9.dll", "cudnn_cnn64_9.dll", "cudnn_engines_precompiled64_9.dll",
                   "cudnn_heuristic64_9.dll", "cudnn_engines_runtime_compiled64_9.dll",
                   "cudnn_adv64_9.dll", "cudnn_graph64_9.dll", "cudnn64_9.dll"]
    # Другие возможные имена (зависит от версии)
    libs_to_try.extend(["cudnn64_8.dll", "cudnn_ops64_8.dll", "cudnn_cnn64_8.dll"])  # Пример для cuDNN 8

    loaded_libs = 0
    for lib in libs_to_try:
        try:
            ctypes.WinDLL(lib)
            print(f"Успешно загружена DLL: {lib}")
            loaded_libs += 1
        except FileNotFoundError:
            pass  # Просто пропускаем, если файл не найден
        except Exception as e_dll:
            print(f"Предупреждение: Ошибка загрузки {lib}: {e_dll}")
    if loaded_libs == 0:
        print("Предупреждение: Не удалось загрузить ни одну DLL CUDNN.")

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
STT_DEVICE = "cuda"  # "cuda" или "cpu"
STT_COMPUTE_TYPE = "int8"  # "float16", "int8_float16", "int8" (int8 обычно быстрее и требует меньше VRAM)

# --- Настройки Аудио STT ---
SOURCE_SAMPLE_RATE = 48000  # Частота дискретизации твоего микрофона
SOURCE_CHANNELS = 2  # Каналы твоего микрофона (1 - моно, 2 - стерео)
TARGET_SAMPLE_RATE = 16000  # Целевая частота для Whisper
TARGET_CHANNELS = 1  # Whisper работает с моно
TARGET_DTYPE = 'float32'  # Тип данных для обработки
BLOCKSIZE = int(SOURCE_SAMPLE_RATE * 0.1)  # Размер блока аудио (100 мс)

# --- Константы для VAD ---
VAD_ENERGY_THRESHOLD = 0.005  # Порог энергии для детекции речи (подбирается экспериментально)
VAD_SPEECH_PAD_MS = 200  # Сколько тишины добавить в начало и конец речевого сегмента (мс)
VAD_MIN_SPEECH_MS = 250  # Минимальная длительность сегмента, считающегося речью (мс)
VAD_SILENCE_TIMEOUT_MS = 1500  # Сколько тишины должно пройти, чтобы считать фразу законченной (мс)

# --- Глобальные переменные ---
conversation_history = []
MAX_HISTORY_LENGTH = 10  # Сколько последних пар сообщений (юзер+бот) хранить
audio_queue = queue.Queue()  # Очередь для аудио блоков с микрофона
recording_active = threading.Event()  # Флаг для управления потоком записи
last_activity_time = time.time()  # Время последнего сообщения/речи
INACTIVITY_THRESHOLD_SECONDS = 60  # Порог неактивности для запуска монолога (секунды)
stt_enabled = True  # Флаг для управления STT (через хоткей)

# --- НОВЫЕ ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ ДЛЯ БЛОКИРОВКИ ---
is_processing_response = False  # Флаг, показывающий, занят ли бот обработкой/ответом
tts_lock = asyncio.Lock()  # Блокировка для эксклюзивного доступа к TTS

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
# Используем промпт из файла, не меняем
SYSTEM_PROMPT = """## Твоя Личность: Джордж Дроид



Ты - Джордж Дроид, остроумный, саркастичный и высокоинтеллектуальный нейросетевой ассистент и соведущий на Twitch-стриме на канале firexrwt. Его владелец Степан. Ты создан, чтобы развлекать аудиторию и взаимодействовать со Степаном и чатом.



**Твой Стиль Общения и Характер:**

1. **Основа - Юмор и Остроумие:** Ты прирожденный юморист с острым умом. Твои шутки должны быть умными, часто основанными на игре слов, иронии, сарказме и наблюдениях за происходящим на стриме или в чате. Ты быстро соображаешь и можешь подколоть Степана или зрителей, но обычно по-дружески.

2. **Язык Ответа:** Ты должен генерировать ответ **строго и исключительно на русском языке**. Не допускается использование слов или фраз на других языках (включая английский), если только это не общепринятый термин или название, не имеющее русского аналога (например, название игры).

3. **Форматирование Ответа:** Не используй звездочки (`*`), скобки или любые другие символы для описания действий, эмоций, звуков или интонаций (например, НЕ НАДО писать `*улыбается*`, `*вздыхает*`, `(смеется)`). Весь твой ответ должен быть обычным текстом. Передавай эмоции и сарказм только через подбор слов и стиль речи, а не через описания действий.

4. **"Черный" Юмор (ИСПОЛЬЗОВАТЬ КРАЙНЕ ОСТОРОЖНО!):** Ты ценишь интеллектуальный юмор, который может быть немного циничным или касаться сложных тем (абсурдность бытия, технологические парадоксы и т.д.). **НО!** Твой юмор **НИКОГДА** не должен:
* Нарушать правила Twitch (см. ниже).
* Быть направлен против каких-либо групп людей (расовых, религиозных, гендерных, ЛГБТ+ и т.д.).
* Оскорблять или унижать кого-либо.
* Касаться реальных трагедий, чувствительных социальных проблем таким образом, чтобы это выглядело как издевательство или глумление.
* Быть чрезмерно жестоким или шокирующим ради шока.
* **Думай об этом как об "едком сарказме" или "интеллектуальной провокации", а не как о безвкусной "чернухе".** Если есть хоть малейшее сомнение, что шутка может быть неправильно понята или нарушить правила - **НЕ ИСПОЛЬЗУЙ ЕЕ**.

5. **Шутки про Мам (СУПЕР РЕДКО, АБСТРАКТНО, НЕОЖИДАННО):** Это твоя странная, почти глючная особенность. **КРАЙНЕ РЕДКО** (буквально раз в несколько часов или даже реже), совершенно внезапно и, возможно, не к месту, ты можешь выдать **абсурдную, нелепую, сюрреалистичную** шутку в стиле "Твоя мама...". **КЛЮЧЕВЫЕ ПРАВИЛА для этих шуток:**
* **Максимально глупые и оторванные от реальности:** Пример: "Твоя мама настолько старая, что ее первый питомец был динозавр... и он до сих пор просит есть". Или "Твоя мама так хорошо готовит, что даже каннибалы просят у нее рецепт... овощного рагу".
* **Очень редки и неожиданны:** Не надо пытаться вставить их часто. Это должно быть похоже на случайный сбой в твоей программе юмора.
* **Если не уверен на 100%, что это будет смешно и безобидно - НЕ ИСПОЛЬЗУЙ.**

6. **Взаимодействие:** Активно общайся со Степаном (называй его по и мени или Файрекс) и реагируй на сообщения из чата Twitch. Можешь задавать вопросы, комментировать игру или действия Степана, отвечать на вопросы зрителей.
**!!! ВАЖНЕЙШЕЕ ПРАВИЛО №1: СОБЛЮДЕНИЕ ПРАВИЛ TWITCH !!!**
* **ПРИОРИТЕТ №0:** Соблюдение Условий обслуживания (ToS) и Правил сообщества Twitch (**Community Guidelines**) — это **АБСОЛЮТНО ГЛАВНАЯ ЗАДАЧА**. Важнее любой шутки, любого ответа.
* **СТРОГО ЗАПРЕЩЕНО:**
* Разжигание ненависти, дискриминация, оскорбления на основе расы, этнической принадлежности, религии, пола, гендерной идентичности, сексуальной ориентации, возраста, инвалидности и т.д.
* Домогательства, преследование, угрозы в адрес кого-либо.
* Обсуждение или прославление незаконной деятельности, самоповреждения.
* Распространение дезинформации, особенно вредоносной.
* **ИЗБЕГАЙ:** Спорных политических дискуссий, религиозных споров, излишне откровенных тем. Будь умным и тактичным.

7. **Слова, которые ЗАПРЕЩЕНЫ ДЛЯ ИСПОЛЬЗОВАНИЯ(в целом мат допустим, но ОТДЕЛЬНЫЕ СЛОВА НЕДОПУСТИМЫ)**: nigger, nigga, naga, ниггер, нига, нага, faggot, пидор, пидорас, педик, гомик, петух (если не подразумевается птица), хохол, хач, жид, даун, аутист, дебил, retard, virgin, simp, incel, девственник, cимп, инцел, cunt, пизда (по отношению к девушке), куколд, чурка, хиджаб, москаль, негр.

8. ФАЙРЕКС НЕ ФУРРИ И НЕ ГИТЛЕР

* **ПОЛИТИКА ОТКАЗА:** Если запрос от пользователя или ситуация на стриме кажутся тебе рискованными с точки зрения правил Twitch, ты **ДОЛЖЕН** вежливо отказаться от ответа или сменить тему. Пример ответа: "Ох, эта тема кажется немного скользкой для Twitch, давай лучше поговорим о [другая тема]?" или "Мои алгоритмы советуют мне не углубляться в это".

**Твоя Цель:** Быть уникальным, запоминающимся, смешным и интеллектуальным ИИ-персонажем, который делает стрим Степана круче, но при этом всегда остается безопасным, уважительным и на 100% соответствующим правилам Twitch. Ты — умный помощник и развлекатель, а не генератор проблем.

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
    """Передискретизирует аудиоданные."""
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
    """Поток для непрерывной записи аудио с микрофона в очередь."""
    global audio_queue, recording_active, stt_enabled, is_processing_response  # Добавили флаги

    def audio_callback(indata, frames, time, status):
        if status: print(f"Статус аудиопотока: {status}", file=sys.stderr)
        # Не кладем в очередь, если STT выключен ИЛИ бот сейчас отвечает
        if recording_active.is_set() and stt_enabled and not is_processing_response:
            try:
                audio_queue.put_nowait(indata.copy())
            except queue.Full:
                # Можно добавить логирование или очистку старых данных, если очередь переполняется
                # print("Предупреждение: Аудио очередь переполнена, данные потеряны.", file=sys.stderr)
                pass  # Пока просто игнорируем переполнение

    stream = None
    try:
        print(f"Поток записи: Запуск аудиопотока (устройство: {device_index or 'default'})...")
        stream = sd.InputStream(
            device=device_index, samplerate=SOURCE_SAMPLE_RATE, channels=SOURCE_CHANNELS,
            dtype=TARGET_DTYPE, blocksize=BLOCKSIZE, callback=audio_callback)
        with stream:
            while recording_active.is_set(): time.sleep(0.1)  # Просто ждем сигнала завершения
    except sd.PortAudioError as e:
        print(f"КРИТИЧЕСКАЯ ОШИБКА PortAudio в потоке записи: {e}", file=sys.stderr)
        print("Возможные причины: Неверное устройство, занято другим приложением, проблемы с драйверами.",
              file=sys.stderr)
        # Можно попробовать остановить приложение здесь или сигнализировать основной поток
        # recording_active.clear() # Например
    except Exception as e:
        print(f"Критическая ошибка в потоке записи аудио: {e}", file=sys.stderr)
    finally:
        if stream is not None and not stream.closed:
            stream.stop()
            stream.close()
        print("Поток записи: Аудиопоток остановлен.")


def transcribe_audio_faster_whisper(audio_np_array):
    """Распознает аудио с помощью Faster Whisper."""
    global stt_model
    if stt_model is None or not isinstance(audio_np_array, np.ndarray) or audio_np_array.size == 0: return None
    try:
        # print(f"Запуск распознавания faster-whisper...")
        segments, info = stt_model.transcribe(audio_np_array, language="ru", word_timestamps=False)
        full_text = "".join(segment.text for segment in segments).strip()
        # print(f"Распознавание завершено: {full_text}")
        return full_text
    except Exception as e:
        print(f"Ошибка во время распознавания faster-whisper: {e}", file=sys.stderr)
        return None


async def get_ollama_response(user_message):
    """Отправляет запрос к Ollama API и возвращает ответ."""
    global conversation_history, OLLAMA_API_URL, OLLAMA_MODEL_NAME, SYSTEM_PROMPT
    is_monologue_request = user_message.startswith("Сгенерируй короткое")

    # Формирование истории сообщений для запроса
    if is_monologue_request:
        messages_payload = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_message}]
    else:
        # Добавляем текущее сообщение пользователя в историю перед отправкой
        current_user_message = {"role": "user", "content": user_message}
        temp_history = conversation_history + [current_user_message]
        # Обрезаем историю, если она слишком длинная
        if len(temp_history) > MAX_HISTORY_LENGTH:
            temp_history = temp_history[-MAX_HISTORY_LENGTH:]
        messages_payload = [{"role": "system", "content": SYSTEM_PROMPT}] + temp_history

    payload = {"model": OLLAMA_MODEL_NAME, "messages": messages_payload, "stream": False}
    # print(f"Отправка запроса в Ollama с {len(messages_payload)} сообщениями...")

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(OLLAMA_API_URL, json=payload, timeout=60) as response:
                if response.status == 200:
                    response_data = await response.json()
                    llm_content = response_data.get('message', {}).get('content')
                    if llm_content:
                        # Добавляем сообщение пользователя и ответ бота в основную историю ТОЛЬКО ЕСЛИ запрос не монолог
                        if not is_monologue_request:
                            conversation_history.append({"role": "user", "content": user_message})
                            conversation_history.append({"role": "assistant", "content": llm_content})
                            # Обрезаем основную историю после добавления
                            if len(conversation_history) > MAX_HISTORY_LENGTH * 2:  # Умножаем на 2 (юзер+бот)
                                conversation_history = conversation_history[-(MAX_HISTORY_LENGTH * 2):]
                        # print("Ответ от Ollama получен.")
                        return llm_content.strip()
                    else:
                        print(f"Ollama вернула пустой ответ: {response_data}", file=sys.stderr)
                        return None  # Не меняем историю, если ответ пустой
                else:
                    error_text = await response.text()
                    print(f"Ошибка Ollama: Статус {response.status}, Ответ: {error_text}", file=sys.stderr)
                    return None  # Не меняем историю при ошибке
    except asyncio.TimeoutError:
        print("Ошибка Ollama: Таймаут запроса.", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Ошибка при обращении к Ollama: {e}", file=sys.stderr)
        return None


def play_raw_audio_sync(audio_bytes, samplerate, dtype='int16'):
    """Воспроизводит сырые аудио байты (блокирующая операция)."""
    if not audio_bytes or not samplerate: return
    try:
        audio_data = np.frombuffer(audio_bytes, dtype=dtype)
        sd.play(audio_data, samplerate=samplerate, blocking=True)  # Важно: blocking=True
        # sd.wait() # Можно добавить wait() для дополнительной надежности, но blocking=True обычно достаточно
    except Exception as e:
        print(f"Ошибка при воспроизведении sd.play: {e}", file=sys.stderr)


async def speak_text(text_to_speak):
    """Синтезирует речь и воспроизводит ее, используя блокировку tts_lock."""
    global piper_sample_rate, PIPER_EXE_PATH, VOICE_MODEL_PATH, tts_lock  # Добавили tts_lock
    if not piper_sample_rate or not os.path.exists(PIPER_EXE_PATH) or not os.path.exists(VOICE_MODEL_PATH):
        print("TTS недоступен, пропуск озвучки.")
        return

    # Пытаемся захватить блокировку TTS
    async with tts_lock:
        print(f"[TTS LOCK] Захвачен: \"{text_to_speak[:30]}...\"")
        command = [PIPER_EXE_PATH, '--model', VOICE_MODEL_PATH, '--output-raw']
        process = None
        audio_bytes = None
        try:
            # Генерация аудио
            process = await asyncio.create_subprocess_exec(
                *command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
                # Скрыть окно консоли piper.exe на Windows
            )
            # Увеличим таймаут, если генерация долгая
            audio_bytes, stderr_bytes = await asyncio.wait_for(process.communicate(input=text_to_speak.encode('utf-8')),
                                                               timeout=30)

            if process.returncode != 0:
                print(
                    f"Ошибка piper.exe: Exit code {process.returncode}\nStderr: {stderr_bytes.decode('utf-8', errors='ignore')}",
                    file=sys.stderr)
                audio_bytes = None  # Сбрасываем, чтобы не пытаться воспроизвести
            elif not audio_bytes:
                print(f"Ошибка: piper.exe не вернул аудио.\nStderr: {stderr_bytes.decode('utf-8', errors='ignore')}",
                      file=sys.stderr)

        except asyncio.TimeoutError:
            print("Ошибка TTS: Таймаут при ожидании piper.exe", file=sys.stderr)
            if process and process.returncode is None:
                try:
                    print("Попытка принудительно завершить piper.exe...")
                    process.kill()
                    await process.wait()
                    print("piper.exe завершен.")
                except ProcessLookupError:
                    pass  # Процесс уже завершился
                except Exception as kill_e:
                    print(f"Ошибка при остановке piper.exe: {kill_e}", file=sys.stderr)
        except FileNotFoundError:
            print(f"Критическая ошибка TTS: Не найден piper.exe по пути {PIPER_EXE_PATH}", file=sys.stderr)
            # Возможно, стоит остановить приложение, если TTS критичен
        except Exception as e:
            print(f"Ошибка при вызове piper.exe: {e}", file=sys.stderr)

        # Воспроизведение аудио (только если успешно сгенерировано)
        # Выполняется синхронно в отдельном потоке, но все еще внутри блока tts_lock
        if audio_bytes:
            try:
                # print(f"[TTS PLAY] Запуск воспроизведения ({len(audio_bytes)} байт)...")
                await asyncio.to_thread(play_raw_audio_sync, audio_bytes, piper_sample_rate)
                # print(f"[TTS PLAY] Воспроизведение завершено.")
            except Exception as e_play:
                print(f"Ошибка при воспроизведении аудио через asyncio.to_thread: {e_play}", file=sys.stderr)

        # Блокировка автоматически освобождается здесь
        print(f"[TTS LOCK] Освобожден: \"{text_to_speak[:30]}...\"")


def toggle_stt():
    """Переключает состояние STT (вкл/выкл)."""
    global stt_enabled, audio_queue, is_processing_response
    # Не позволяем включать STT, если бот прямо сейчас отвечает
    if not stt_enabled and is_processing_response:
        print("\n[INFO] Нельзя включить STT, пока бот отвечает.\n")
        return

    stt_enabled = not stt_enabled
    status_text = "ВКЛЮЧЕНО" if stt_enabled else "ВЫКЛЮЧЕНО"
    print("\n" + "-" * 30 + f"\n--- Распознавание голоса (STT) {status_text} ---\n" + "-" * 30)
    if not stt_enabled:
        # Очищаем очередь при ВЫКЛЮЧЕНИИ через хоткей
        with audio_queue.mutex: audio_queue.queue.clear()
        print("[INFO] Очередь аудио очищена при выключении STT хоткеем.")


# --- Twitch бот ---
class SimpleBot(twitchio.Client):
    """Класс Twitch бота."""

    def __init__(self, token, initial_channels):
        super().__init__(token=token, initial_channels=initial_channels)
        # Добавляем атрибут для хранения имени канала
        self.target_channel_name = initial_channels[0] if initial_channels else None

    async def event_ready(self):
        """Вызывается один раз при подключении бота."""
        print(f'Подключен к Twitch IRC как | {self.nick}')
        if self.connected_channels:
            # Получаем объект канала из подключенных
            channel_obj = self.get_channel(self.target_channel_name)
            if channel_obj:
                print(
                    f'Присоединился к каналу | {channel_obj.name}\n' + '-' * 40 + '\n--------- Бот готов читать чат ---------\n' + '-' * 40)
                # Можно отправить сообщение в чат о готовности
                # await channel_obj.send("Джордж Дроид на связи!")
            else:
                print(f'ОШИБКА: Не найден объект для канала {self.target_channel_name} среди подключенных.',
                      file=sys.stderr)
        else:
            print(f'НЕ УДАЛОСЬ присоединиться к каналу {self.target_channel_name}. Проверьте токен и имя канала.',
                  file=sys.stderr)

    async def event_message(self, message):
        """Обрабатывает сообщения из чата."""
        # Игнорируем сообщения от самого бота
        if message.echo:
            return

        global last_activity_time, BOT_NAME_FOR_CHECK, OBS_OUTPUT_FILE
        global stt_enabled, audio_queue, is_processing_response  # Добавили флаг

        # Проверка, относится ли сообщение к нужному каналу (на всякий случай)
        if message.channel.name != self.target_channel_name:
            return

        # Проверка триггеров (имя бота или выделение)
        content_lower = message.content.lower()
        trigger_name_parts = [part.lower() for part in BOT_NAME_FOR_CHECK.split() if len(part) > 2]
        bot_name_mentioned = any(trigger in content_lower for trigger in trigger_name_parts)
        message_tags = message.tags or {}
        is_highlighted = message_tags.get('msg-id') == 'highlighted-message'

        if not bot_name_mentioned and not is_highlighted:
            return  # Сообщение не для бота

        current_time_str = datetime.datetime.now().strftime('%H:%M:%S')

        # --- Проверка флага занятости ---
        if is_processing_response:
            print(f"[{current_time_str}] Бот занят. Игнорируется сообщение от {message.author.name}.")
            # Можно отправить короткий ответ в чат, что бот занят (но осторожно, чтобы не спамить)
            # await message.channel.send(f"@{message.author.name}, момент, обрабатываю предыдущий запрос!")
            return  # Выходим, если бот уже обрабатывает другой запрос

        # --- Начало обработки ---
        original_stt_state = False  # Инициализируем на случай ошибки до присвоения
        try:
            is_processing_response = True  # Устанавливаем флаг занятости
            print(f"[{current_time_str}] НАЧАЛО обработки сообщения от {message.author.name}.")

            last_activity_time = time.time()  # Обновляем время активности
            print(f"[{current_time_str}] {message.author.name}: {message.content}")

            original_stt_state = stt_enabled  # Запоминаем исходное состояние STT

            # Отключаем STT и чистим очередь
            stt_was_enabled = False
            if stt_enabled:
                print("[INFO] Отключаю STT для ответа (чат).")
                stt_enabled = False
                stt_was_enabled = True  # Запоминаем, что он БЫЛ включен
            # Чистим очередь в любом случае, чтобы убрать накопившееся, пока бот был занят
            with audio_queue.mutex:
                audio_queue.queue.clear()
            # print("[INFO] Очередь аудио очищена перед обработкой (чат).")

            # Очистка файла OBS
            try:
                with open(OBS_OUTPUT_FILE, 'w', encoding='utf-8') as f:
                    f.write("")
            except Exception as e:
                print(f"[{current_time_str}] ОШИБКА очистки файла OBS: {e}", file=sys.stderr)

            # Вызов LLM
            llm_response_text = await get_ollama_response(message.content)

            if llm_response_text:
                print(f"[{current_time_str}] Ответ Ollama: {llm_response_text}")
                try:
                    with open(OBS_OUTPUT_FILE, 'w', encoding='utf-8') as f:
                        f.write(llm_response_text)
                except Exception as e:
                    print(f"[{current_time_str}] ОШИБКА записи в файл OBS: {e}", file=sys.stderr)

                # Озвучивание (speak_text использует tts_lock)
                await speak_text(llm_response_text)

                # Очередь снова чистим ПОСЛЕ TTS на всякий случай
                with audio_queue.mutex:
                    audio_queue.queue.clear()
                # print("[INFO] Очередь аудио очищена после TTS (чат).")

            else:
                print(f"[{current_time_str}] Не удалось получить ответ от Ollama для {message.author.name}.")
                # Можно отправить сообщение об ошибке в чат
                # await message.channel.send(f"@{message.author.name}, что-то пошло не так, не могу сейчас ответить.")

            # Восстанавливаем состояние STT (ДО снятия флага занятости)
            # Включаем обратно, ТОЛЬКО если он был включен до начала обработки этого сообщения
            if stt_was_enabled:  # Используем флаг stt_was_enabled
                print("[INFO] Включаю STT обратно после ответа (чат).")
                stt_enabled = True
            # Если STT был выключен пользователем во время ответа - он останется выключенным

        except Exception as e:
            print(f"[{current_time_str}] КРИТИЧЕСКАЯ ОШИБКА в event_message: {e}", file=sys.stderr)
            # Попытка восстановить STT даже при ошибке
            try:
                # Включаем, если он был включен изначально (используем original_stt_state, если он успел присвоиться)
                if 'original_stt_state' in locals() and original_stt_state and not stt_enabled:
                    stt_enabled = True
                    print("[INFO] STT восстановлен после ошибки в event_message.")
            except Exception as e_restore:
                print(f"Ошибка восстановления STT: {e_restore}", file=sys.stderr)
        finally:
            # --- Окончание обработки ---
            is_processing_response = False  # Снимаем флаг занятости В ЛЮБОМ СЛУЧАЕ
            print(f"[{current_time_str}] КОНЕЦ обработки сообщения от {message.author.name}.")


# --- Асинхронный цикл обработки аудио и STT ---
async def stt_processing_loop():
    """Асинхронный цикл для VAD и STT."""
    global audio_queue, recording_active, stt_model, OBS_OUTPUT_FILE, last_activity_time
    global stt_enabled, is_processing_response  # Добавили флаг

    # Настройки VAD
    silence_blocks_needed = int(VAD_SILENCE_TIMEOUT_MS / (BLOCKSIZE / SOURCE_SAMPLE_RATE * 1000))
    min_speech_blocks = int(VAD_MIN_SPEECH_MS / (BLOCKSIZE / SOURCE_SAMPLE_RATE * 1000))
    speech_pad_blocks = int(VAD_SPEECH_PAD_MS / (BLOCKSIZE / SOURCE_SAMPLE_RATE * 1000))
    print(
        f"VAD Настройки: Порог={VAD_ENERGY_THRESHOLD:.3f}, Мин. речь={min_speech_blocks} бл., Пауза={silence_blocks_needed} бл., Паддинг={speech_pad_blocks} бл.")

    is_speaking_vad = False  # Переименовали, чтобы не путать с is_processing_response
    silence_blocks_count = 0
    speech_audio_buffer = []
    buffer_for_padding = []

    print("Цикл обработки STT запущен...")
    while recording_active.is_set():
        # Пропускаем итерацию, если STT выключен ИЛИ бот уже отвечает
        if not stt_enabled or is_processing_response:
            # Если бот занят, но VAD думает, что идет речь - сбрасываем VAD
            if is_processing_response and is_speaking_vad:
                is_speaking_vad = False;
                speech_audio_buffer = [];
                buffer_for_padding = [];
                silence_blocks_count = 0
                # print("[VAD INFO] Сброс VAD, так как бот занят.")
            await asyncio.sleep(0.1)
            continue

        # --- Обработка аудио из очереди ---
        try:
            block = audio_queue.get_nowait()  # Используем get_nowait
        except queue.Empty:
            await asyncio.sleep(0.01)
            # --- Обработка VAD при пустой очереди ---
            if is_speaking_vad:
                silence_blocks_count += 1
                if silence_blocks_count >= silence_blocks_needed:
                    if len(speech_audio_buffer) > min_speech_blocks:
                        # --- ЗАПУСК ОБРАБОТКИ РЕЧИ (ВЕТКА 1: QUEUE EMPTY) ---
                        final_buffer_copy = speech_audio_buffer.copy()
                        source_identifier = "STT (ветка 1: Queue Empty)"
                        is_speaking_vad = False;
                        speech_audio_buffer = [];
                        buffer_for_padding = [];
                        silence_blocks_count = 0  # Сброс VAD
                        # Запускаем обработку асинхронно, НЕ дожидаясь ее завершения здесь
                        asyncio.create_task(process_recognized_speech(final_buffer_copy, source_identifier))
                    else:
                        # print("Обнаружен слишком короткий звук, игнорируется (ветка 1).")
                        is_speaking_vad = False;
                        speech_audio_buffer = [];
                        buffer_for_padding = [];
                        silence_blocks_count = 0  # Сброс VAD
            continue  # К началу цикла while

        # --- Логика VAD ---
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
                        # --- ЗАПУСК ОБРАБОТКИ РЕЧИ (ВЕТКА 2: SILENCE DETECTED) ---
                        final_buffer_copy = speech_audio_buffer.copy()
                        source_identifier = "STT (ветка 2: Silence Detected)"
                        is_speaking_vad = False;
                        speech_audio_buffer = [];
                        buffer_for_padding = [];
                        silence_blocks_count = 0  # Сброс VAD
                        # Запускаем обработку асинхронно
                        asyncio.create_task(process_recognized_speech(final_buffer_copy, source_identifier))
                    else:
                        # print("Обнаружен слишком короткий звук, игнорируется (ветка 2).")
                        is_speaking_vad = False;
                        speech_audio_buffer = [];
                        buffer_for_padding = [];
                        silence_blocks_count = 0  # Сброс VAD

    print("Цикл обработки STT остановлен.")


# --- НОВАЯ АСИНХРОННАЯ ФУНКЦИЯ для обработки распознанной речи ---
async def process_recognized_speech(audio_buffer_list, source_id="STT"):
    """Обрабатывает сегмент аудио: STT -> LLM -> TTS, управляя флагом занятости."""
    global is_processing_response, stt_enabled, audio_queue, last_activity_time, OBS_OUTPUT_FILE

    current_time_str = datetime.datetime.now().strftime('%H:%M:%S')

    # Проверяем флаг еще раз + используем compare-and-swap для атомарности
    # (Хотя с GIL это излишне, но для ясности)
    if is_processing_response:
        print(f"[{current_time_str}] Бот занят (проверка в process_recognized_speech). Игнорируется {source_id}.")
        return

    # --- Начало обработки ---
    stt_was_enabled = False  # Флаг, что STT БЫЛ включен до начала обработки
    try:
        is_processing_response = True  # Устанавливаем флаг занятости
        print(f"[{current_time_str}] НАЧАЛО обработки речи ({source_id}).")

        # Подготовка аудио и STT
        # print(f"Обработка {len(audio_buffer_list)} аудио блоков...")
        full_audio_raw = np.concatenate(audio_buffer_list, axis=0)
        mono_audio = full_audio_raw.mean(axis=1) if SOURCE_CHANNELS > 1 else full_audio_raw
        resampled_for_stt = resample_audio(mono_audio, SOURCE_SAMPLE_RATE, TARGET_SAMPLE_RATE)
        recognized_text = None
        if resampled_for_stt is not None and resampled_for_stt.size > 0:
            recognized_text = await asyncio.to_thread(transcribe_audio_faster_whisper, resampled_for_stt)

        if recognized_text:
            last_activity_time = time.time()  # Обновляем активность только если что-то распознано
            print(f"STT Распознано ({source_id}): {recognized_text}")

            # Отключаем STT и чистим очередь
            if stt_enabled:
                print(f"[INFO] Отключаю STT для ответа ({source_id}).")
                stt_enabled = False
                stt_was_enabled = True  # Запоминаем, что отключали
            with audio_queue.mutex:
                audio_queue.queue.clear()
            # print(f"[INFO] Очередь аудио очищена перед обработкой ({source_id}).")

            # Очистка файла OBS
            try:
                with open(OBS_OUTPUT_FILE, 'w', encoding='utf-8') as f:
                    f.write("")
            except Exception as e:
                print(f"[{current_time_str}] ОШИБКА очистки файла OBS: {e}", file=sys.stderr)

            # Вызов LLM
            llm_response_text = await get_ollama_response(f"(Голосовое сообщение от Степана): {recognized_text}")

            if llm_response_text:
                print(f"[{current_time_str}] Ответ Ollama ({source_id}): {llm_response_text}")
                try:
                    with open(OBS_OUTPUT_FILE, 'w', encoding='utf-8') as f:
                        f.write(llm_response_text)
                except Exception as e:
                    print(f"[{current_time_str}] ОШИБКА записи в файл OBS: {e}", file=sys.stderr)

                # Озвучивание
                await speak_text(llm_response_text)

                # Очистка очереди ПОСЛЕ TTS
                with audio_queue.mutex:
                    audio_queue.queue.clear()
                # print(f"[INFO] Очередь аудио очищена после TTS ({source_id}).")
            else:
                print(f"[{current_time_str}] Не удалось получить ответ от Ollama ({source_id}).")

            # Восстанавливаем STT, если он БЫЛ включен
            if stt_was_enabled:
                print(f"[INFO] Включаю STT обратно после ответа ({source_id}).")
                stt_enabled = True
        else:
            print(f"STT: Не удалось распознать речь или аудио было некорректным ({source_id}).")
            # Если речь не распознана, STT не отключался, флаг занятости будет снят в finally

    except Exception as e:
        print(f"[{current_time_str}] КРИТИЧЕСКАЯ ОШИБКА в process_recognized_speech ({source_id}): {e}",
              file=sys.stderr)
        # Попытка восстановить STT
        try:
            # Включаем, если отключали
            if stt_was_enabled:
                stt_enabled = True
                print(f"[INFO] STT восстановлен после ошибки в process_recognized_speech ({source_id}).")
        except Exception as e_restore:
            print(f"Ошибка восстановления STT: {e_restore}", file=sys.stderr)
    finally:
        is_processing_response = False  # Снимаем флаг занятости В ЛЮБОМ СЛУЧАЕ
        print(f"[{current_time_str}] КОНЕЦ обработки речи ({source_id}).")


# --- Цикл случайных монологов ---
async def monologue_loop():
    """Асинхронный цикл для запуска монологов при бездействии."""
    global last_activity_time, recording_active, OBS_OUTPUT_FILE, stt_enabled, BOT_NAME_FOR_CHECK
    global audio_queue, is_processing_response  # Добавили флаг

    print("Цикл монологов запущен...")
    while recording_active.is_set():
        # Интервал проверки неактивности
        await asyncio.sleep(15)

        # Пропускаем, если STT выключен ИЛИ бот уже отвечает
        if not stt_enabled or is_processing_response:
            continue

        current_time_unix = time.time()
        time_since_last_activity = current_time_unix - last_activity_time

        if time_since_last_activity > INACTIVITY_THRESHOLD_SECONDS:
            current_time_str = datetime.datetime.now().strftime('%H:%M:%S')

            # --- Проверка флага занятости (повторно) ---
            if is_processing_response:
                # print(f"[{current_time_str}] Бот занят. Пропуск монолога.")
                continue

            # --- Начало обработки монолога ---
            stt_was_enabled = False  # Флаг, что STT БЫЛ включен
            try:
                is_processing_response = True  # Устанавливаем флаг занятости
                print(f"[{current_time_str}] НАЧАЛО обработки монолога.")
                print(
                    f"[{current_time_str}] Обнаружено бездействие ({time_since_last_activity:.0f} сек), запуск монолога...")

                # Отключаем STT и чистим очередь
                if stt_enabled:
                    print("[INFO] Отключаю STT для монолога.")
                    stt_enabled = False
                    stt_was_enabled = True
                with audio_queue.mutex:
                    audio_queue.queue.clear()
                # print("[INFO] Очередь аудио очищена перед монологом.")

                # Формирование промпта для монолога
                monologue_prompt = (f"Сгенерируй короткое (1-2 предложения) спонтанное размышление, интересный факт "
                                    f"или вопрос к чату от имени {BOT_NAME_FOR_CHECK}, чтобы заполнить тишину на стриме. "
                                    "Начни фразу естественно, например: 'Кстати, чат...', 'Задумался тут...', "
                                    "'А вы знали, что...', 'Степан, а ты когда-нибудь...', но НЕ как ответ на запрос "
                                    "('Хорошо, вот факт...'). Тема абсолютно случайна.")

                # Вызов LLM
                llm_response_text = await get_ollama_response(monologue_prompt)

                if llm_response_text:
                    print(f"[{current_time_str}] Монолог Ollama: {llm_response_text}")
                    try:
                        with open(OBS_OUTPUT_FILE, 'w', encoding='utf-8') as f:
                            f.write(llm_response_text)
                    except Exception as e:
                        print(f"[{current_time_str}] ОШИБКА записи монолога в файл OBS: {e}", file=sys.stderr)

                    # Озвучивание
                    await speak_text(llm_response_text)

                    # Очистка очереди ПОСЛЕ TTS
                    with audio_queue.mutex:
                        audio_queue.queue.clear()
                    # print("[INFO] Аудио очередь очищена после монолога TTS")

                    last_activity_time = time.time()  # Обновляем время активности ПОСЛЕ монолога
                else:
                    print(f"[{current_time_str}] Не удалось получить монолог от Ollama.")

                # Восстанавливаем STT, если отключали
                if stt_was_enabled:
                    print("[INFO] Включаю STT обратно после монолога.")
                    stt_enabled = True

            except Exception as e:
                print(f"[{current_time_str}] КРИТИЧЕСКАЯ ОШИБКА в monologue_loop: {e}", file=sys.stderr)
                # Попытка восстановить STT
                try:
                    if stt_was_enabled:
                        stt_enabled = True
                        print("[INFO] STT восстановлен после ошибки в monologue_loop.")
                except Exception as e_restore:
                    print(f"Ошибка восстановления STT: {e_restore}", file=sys.stderr)
            finally:
                # --- Окончание обработки монолога ---
                is_processing_response = False  # Снимаем флаг занятости
                print(f"[{current_time_str}] КОНЕЦ обработки монолога.")

    print("Цикл монологов остановлен.")


# --- Поток слушателя горячих клавиш ---
def hotkey_listener_thread():
    """Поток, слушающий горячую клавишу для вкл/выкл STT."""
    stt_hotkey = 'ctrl+;'  # Можно изменить на удобную комбинацию
    try:
        print(f"\nНажмите '{stt_hotkey}' для включения/выключения распознавания голоса.")
        keyboard.add_hotkey(stt_hotkey, toggle_stt)

        # --- ИЗМЕНЕНИЕ: Заменяем wait() на цикл ---
        while recording_active.is_set():
            # Просто проверяем флаг раз в полсекунды
            time.sleep(0.5)
        # --- КОНЕЦ ИЗМЕНЕНИЯ ---

        print("Поток слушателя горячих клавиш: получен сигнал завершения (is_set() стал False).")  # Уточнили сообщение
    except ImportError:
        print("\nОШИБКА: Библиотека 'keyboard' не найдена. Установите: pip install keyboard", file=sys.stderr)
        print("Горячая клавиша для STT работать не будет.")
    except Exception as e_hk_thread:
        print(f"\nОшибка в потоке слушателя горячих клавиш: {e_hk_thread}", file=sys.stderr)
        if "root privileges" in str(e_hk_thread).lower() or "permission denied" in str(e_hk_thread).lower():
            print(
                "Подсказка: На Linux/MacOS для глобальных хоткеев могут требоваться права суперпользователя (sudo) или разрешения доступа.")
    finally:
        try:
            # Убираем хоткей при завершении
            keyboard.remove_hotkey(stt_hotkey)
        except Exception:
            pass  # Игнорируем ошибки при очистке
        print("Поток слушателя горячих клавиш завершен.")


# --- Основная асинхронная функция запуска ---
async def main_async():
    """Инициализирует и запускает основные асинхронные задачи."""
    print("Запуск основного Twitch бота, цикла STT и цикла монологов...")
    client = SimpleBot(token=TWITCH_ACCESS_TOKEN, initial_channels=[TWITCH_CHANNEL])

    # Запускаем задачи
    twitch_task = asyncio.create_task(client.start(), name="TwitchTask")
    stt_task = asyncio.create_task(stt_processing_loop(), name="STTTask")
    monologue_task = asyncio.create_task(monologue_loop(), name="MonologueTask")

    all_main_tasks = [twitch_task, stt_task, monologue_task]

    # Ожидаем завершения любой из задач (или KeyboardInterrupt)
    done, pending = await asyncio.wait(all_main_tasks, return_when=asyncio.FIRST_COMPLETED)

    print("\nОдна из основных задач завершилась или получена ошибка.")
    for task in done:
        try:
            if task.exception():
                print(f"Задача {task.get_name()} завершилась с ошибкой: {task.exception()}", file=sys.stderr)
                # Можно добавить более детальное логирование ошибки
                # import traceback
                # traceback.print_exception(task.exception())
            elif task.cancelled():
                print(f"Задача {task.get_name()} была отменена.")
            else:
                print(f"Задача {task.get_name()} успешно завершилась.")
        except asyncio.InvalidStateError:
            print(f"Задача {task.get_name()} в неопределенном состоянии при проверке.")

    # Отменяем оставшиеся задачи
    print("Отмена оставшихся асинхронных задач...")
    cancelled_count = 0
    for task in pending:
        if task and not task.done():
            task.cancel()
            cancelled_count += 1
    print(f"Отменено {cancelled_count} задач.")

    # Даем время на обработку отмены
    if pending:
        await asyncio.wait(pending, timeout=1.0)


# --- Точка входа в программу ---
if __name__ == "__main__":
    # Проверки перед запуском
    print("-" * 40)
    if not TWITCH_ACCESS_TOKEN or not TWITCH_CHANNEL:
        print("КРИТИЧЕСКАЯ ОШИБКА: Не заданы TWITCH_ACCESS_TOKEN или TWITCH_CHANNEL в .env файле!", file=sys.stderr)
        sys.exit(1)
    if not OLLAMA_API_URL: print("Предупреждение: Не задан OLLAMA_API_URL, используется значение по умолчанию.")
    if not piper_sample_rate: print("Предупреждение: TTS (Piper) не инициализирован.")
    if not stt_model: print("Предупреждение: STT модель (Whisper) не загружена.")
    print("-" * 40)

    # Определение микрофона
    default_mic_index = None
    mic_name = "N/A"
    try:
        print("Доступные устройства записи (Input):")
        print(sd.query_devices())
        # Пытаемся получить устройство по умолчанию
        default_mic_info = sd.query_devices(kind='input')
        if isinstance(default_mic_info, dict) and 'index' in default_mic_info:
            default_mic_index = default_mic_info['index']
            mic_name = default_mic_info.get('name', mic_name)
        else:
            # Если не получилось, пробуем индекс по умолчанию из sd.default
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

    # Запуск фоновых потоков
    recording_active.set()  # Сигнализируем потокам, что можно работать

    recorder_thread = threading.Thread(target=audio_recording_thread, args=(default_mic_index,), daemon=True,
                                       name="AudioRecorder")
    recorder_thread.start()
    print("Поток записи аудио запущен.")

    # Запускаем поток хоткея только если keyboard импортировался
    hotkey_thread = None
    if 'keyboard' in sys.modules:
        hotkey_thread = threading.Thread(target=hotkey_listener_thread, daemon=True, name="HotkeyListener")
        hotkey_thread.start()
    else:
        print("Поток слушателя горячих клавиш не запущен (библиотека keyboard не найдена).")

    # Получаем или создаем цикл событий asyncio
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    # Основной цикл программы
    main_task = None
    try:
        # Запускаем основную асинхронную логику
        main_task = loop.create_task(main_async(), name="MainAsync")
        loop.run_until_complete(main_task)

    except KeyboardInterrupt:
        print("\nПолучен сигнал Ctrl+C...")
        if main_task and not main_task.done():
            main_task.cancel()
            loop.run_until_complete(main_task)  # Даем обработать отмену
    except asyncio.CancelledError:
        print("Главный цикл был отменен...")
    finally:
        print("Начало процедуры graceful shutdown...")

        # 1. Сигнализируем потокам о завершении
        print("Остановка фоновых потоков...")
        recording_active.clear()  # Сигнал для recorder_thread и hotkey_listener_thread

        # 2. Ожидаем завершения потоков
        active_threads = [t for t in [recorder_thread, hotkey_thread] if t and t.is_alive()]
        if active_threads:
            print(f"Ожидание завершения {len(active_threads)} потоков...")
            for t in active_threads:
                try:
                    t.join(timeout=3.0)  # Даем чуть больше времени
                except Exception as e_join:
                    print(f"Ошибка при ожидании потока {t.name}: {e_join}", file=sys.stderr)
                if t.is_alive():
                    print(f"Предупреждение: Поток {t.name} не завершился за таймаут.", file=sys.stderr)
            print("Ожидание потоков завершено (или таймаут).")

        # 3. Корректно завершаем asyncio задачи и цикл
        print("Завершение asyncio...")

        # Собираем все оставшиеся задачи (могли создаться новые)
        all_asyncio_tasks = asyncio.all_tasks(loop=loop)
        if all_asyncio_tasks:
            print(f"Отмена {len(all_asyncio_tasks)} оставшихся asyncio задач...")
            for task in all_asyncio_tasks:
                if not task.done():
                    task.cancel()
            try:
                # Даем задачам обработать отмену и собираем результаты/исключения
                loop.run_until_complete(asyncio.gather(*all_asyncio_tasks, return_exceptions=True))
                print("Сбор отмененных задач завершен.")
            except Exception as e_gather:
                print(f"Ошибка при финальном asyncio.gather: {e_gather}", file=sys.stderr)

        # Завершаем асинхронные генераторы
        try:
            print("Завершение асинхронных генераторов...")
            loop.run_until_complete(loop.shutdown_asyncgens())
            print("Асинхронные генераторы завершены.")
        except Exception as e_gens:
            print(f"Ошибка при shutdown_asyncgens: {e_gens}", file=sys.stderr)
        finally:
            # Закрываем цикл событий
            if not loop.is_closed():
                loop.close()
                print("Цикл asyncio закрыт.")

        print("-" * 40 + "\nПрограмма штатно завершена.\n" + "-" * 40)
        sys.exit(0)  # Явный выход с кодом 0