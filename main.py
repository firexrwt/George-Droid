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
import chess
import chess.engine
import random
import re

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
OLLAMA_MODEL_NAME = os.getenv('OLLAMA_MODEL_NAME', "llama3.1:8b-instruct-q5_K_S")

# --- Настройки Piper TTS ---
PIPER_EXE_PATH = os.getenv('PIPER_EXE_PATH', 'piper_tts_bin/piper.exe')
VOICE_MODEL_PATH = os.getenv('PIPER_VOICE_MODEL_PATH', 'voices/ru_RU-ruslan-medium.onnx')
VOICE_CONFIG_PATH = os.getenv('PIPER_VOICE_CONFIG_PATH', 'voices/ru_RU-ruslan-medium.onnx.json')

# --- Настройка файла для вывода текста в OBS ---
OBS_OUTPUT_FILE = "obs_ai_response.txt"

# --- Настройки Faster Whisper ---
STT_MODEL_SIZE = "medium"
STT_DEVICE = "cuda"
STT_COMPUTE_TYPE = "int8_float16"

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

# --- Шахматные переменные ---
chess_board = None
chess_engine = None
is_chess_game_active = False
player_color = None
bot_color = None
STOCKFISH_PATH = os.getenv('STOCKFISH_PATH', r"C:\stockfish\stockfish-windows-x86-64-avx2.exe")
GUI_SCRIPT_PATH = "pysimplegui_chess_gui.py"  # Имя GUI скрипта
gui_process = None
chess_game_task = None
ipc_player_move_queue = asyncio.Queue()

# --- Загрузка модели Faster Whisper ---
stt_model = None
try:
    from faster_whisper import WhisperModel

    print(f"Загрузка faster-whisper '{STT_MODEL_SIZE}'...")
    stt_model = WhisperModel(STT_MODEL_SIZE, device=STT_DEVICE, compute_type=STT_COMPUTE_TYPE)
    print("Модель faster-whisper загружена.")
except ImportError:
    print("ОШИБКА: faster-whisper не установлен.")
    stt_model = None
except Exception as e:
    print(f"Критическая ошибка загрузки faster-whisper: {e}")
    stt_model = None

# --- Чтение Sample Rate из конфига голоса Piper ---
piper_sample_rate = None
try:
    if os.path.exists(VOICE_CONFIG_PATH):
        with open(VOICE_CONFIG_PATH, 'r', encoding='utf-8') as f:
            piper_sample_rate = json.load(f).get('audio', {}).get('sample_rate')
        if piper_sample_rate:
            print(f"Piper SR: {piper_sample_rate}")
        else:
            print(f"ОШИБКА: Не найден 'sample_rate' в {VOICE_CONFIG_PATH}")
    else:
        print(f"ОШИБКА: Не найден JSON конфиг голоса: {os.path.abspath(VOICE_CONFIG_PATH)}")

    if not all([os.path.exists(PIPER_EXE_PATH), os.path.exists(VOICE_MODEL_PATH), piper_sample_rate]):
        print("ОШИБКА: TTS (Piper) не будет работать.")
        piper_sample_rate = None
except Exception as e:
    print(f"Критическая ошибка инициализации Piper TTS: {e}")
    piper_sample_rate = None

# --- Системный промпт ---
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

# --- Проверка имени бота ---
BOT_NAME_FOR_CHECK = "Джордж Дроид"
prompt_lines = SYSTEM_PROMPT.split('\n', 2)
if len(prompt_lines) > 1 and prompt_lines[0].startswith("## Твоя Личность:"):
    potential_name = prompt_lines[0].replace("## Твоя Личность:", "").strip()
    if potential_name:
        BOT_NAME_FOR_CHECK = potential_name
print(f"Имя бота для триггеров: '{BOT_NAME_FOR_CHECK}'")


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
                pass

    stream = None
    try:
        print(f"Запуск аудиопотока (устройство: {device_index or 'default'})...")
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


async def get_ollama_response(user_message):
    global conversation_history, OLLAMA_API_URL, OLLAMA_MODEL_NAME, SYSTEM_PROMPT
    is_monologue_request = user_message.startswith("Сгенерируй короткое")
    is_chess_commentary = "Шахматы:" in user_message or "ход:" in user_message.lower()

    if is_monologue_request or is_chess_commentary:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_message}]
    else:
        current_msg = {"role": "user", "content": user_message}
        temp_history = conversation_history + [current_msg]
        if len(temp_history) > MAX_HISTORY_LENGTH:
            temp_history = temp_history[-MAX_HISTORY_LENGTH:]
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + temp_history

    payload = {"model": OLLAMA_MODEL_NAME, "messages": messages, "stream": False}
    timeout = aiohttp.ClientTimeout(total=120 if is_chess_commentary else 60)

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(OLLAMA_API_URL, json=payload, timeout=timeout) as response:
                if response.status == 200:
                    data = await response.json()
                    content = data.get('message', {}).get('content')
                    if content:
                        if not is_monologue_request and not is_chess_commentary:
                            conversation_history.extend([current_msg, {"role": "assistant", "content": content}])
                            if len(conversation_history) > MAX_HISTORY_LENGTH * 2:
                                conversation_history = conversation_history[-(MAX_HISTORY_LENGTH * 2):]
                        return content.strip()
                    else:
                        print(f"Ollama пустой ответ: {data}", file=sys.stderr)
                        return None
                else:
                    print(f"Ошибка Ollama: {response.status}, {await response.text()}", file=sys.stderr)
                    return None
    except asyncio.TimeoutError:
        print(f"Ошибка Ollama: Таймаут ({timeout.total} сек).", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Ошибка Ollama ({type(e).__name__}): {e}", file=sys.stderr)
        return None


def play_raw_audio_sync(audio_bytes, samplerate, dtype='int16'):
    if not audio_bytes or not samplerate:
        return
    try:
        sd.play(np.frombuffer(audio_bytes, dtype=dtype), samplerate=samplerate, blocking=True)
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
                    process.kill()
                    await process.wait()
                    print("Piper убит.")
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
        with audio_queue.mutex:
            audio_queue.queue.clear()


def toggle_chat_interaction():
    global chat_interaction_enabled
    chat_interaction_enabled = not chat_interaction_enabled
    status = "ВКЛ" if chat_interaction_enabled else "ВЫКЛ"
    print(f"\n=== Реакция на чат {status} ===")


# --- Функции для Шахмат ---

async def initialize_chess_engine():
    global chess_engine
    if chess_engine:
        return True
    if not STOCKFISH_PATH or not os.path.exists(STOCKFISH_PATH):
        print(f"ОШИБКА: Stockfish не найден: {STOCKFISH_PATH}")
        await speak_text("Не могу найти шахматный движок.")
        return False
    try:
        print(f"Инициализация Stockfish: {STOCKFISH_PATH}...")
        transport, engine_proto = await chess.engine.popen_uci(STOCKFISH_PATH)
        chess_engine = engine_proto
        print("Stockfish инициализирован.")
        return True
    except Exception as e:
        print(f"Критическая ошибка инициализации Stockfish: {e}")
        await speak_text("Ошибка запуска шахматного модуля.")
        chess_engine = None
        return False


async def close_chess_engine():
    global chess_engine
    if chess_engine:
        print("Закрытие Stockfish...")
        try:
            await chess_engine.quit()
        except Exception as e:
            print(f"Ошибка при закрытии Stockfish: {e}")
        finally:
            chess_engine = None
            print("Stockfish закрыт.")


async def start_chess_game():
    global is_chess_game_active, chess_board, chess_engine, player_color, bot_color
    global gui_process, chess_game_task, is_processing_response, ipc_player_move_queue, GUI_SCRIPT_PATH

    player_starts_as = random.choice([chess.WHITE, chess.BLACK])
    player_color_str = "белыми" if player_starts_as == chess.WHITE else "черными"

    print(f"Попытка запуска шахмат (голосом). Игрок будет играть {player_color_str}.")
    if is_chess_game_active:
        print("Игра уже идет.")
        await speak_text("Мы уже играем!")
        return
    if is_processing_response:
        print("Бот занят, старт отложен.")
        await speak_text("Секунду, закончу и начнем.")
        return
    if not GUI_SCRIPT_PATH or not os.path.exists(GUI_SCRIPT_PATH):
        print(f"ОШИБКА: Скрипт GUI не найден: {GUI_SCRIPT_PATH}")
        await speak_text("Не могу найти файл доски.")
        return
    if not chess_engine and not await initialize_chess_engine():
        return

    is_processing_response = True
    print("[INFO] Реакция на чат будет ОТКЛЮЧЕНА на время игры.")
    await speak_text(f"Хорошо, Степан! Запускаю шахматы. Ты играешь {player_color_str}.")

    try:
        print(f"Запуск GUI: {GUI_SCRIPT_PATH}")
        python_exe = sys.executable
        creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        gui_process = await asyncio.create_subprocess_exec(
            python_exe, GUI_SCRIPT_PATH,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            creationflags=creationflags
        )
        await asyncio.sleep(2)

        if gui_process.returncode is not None:
            stderr_gui = await gui_process.stderr.read()
            error_msg = f"ОШИБКА: GUI завершился с кодом {gui_process.returncode}\nStderr: {stderr_gui.decode(errors='ignore')}"
            print(error_msg)
            gui_process = None
            raise Exception("GUI не запустился")

        print(f"GUI запущен (PID: {gui_process.pid}).")
        chess_board = chess.Board()
        player_color = player_starts_as
        bot_color = not player_color
        is_chess_game_active = True

        while not ipc_player_move_queue.empty():
            ipc_player_move_queue.get_nowait()

        start_cmd = f"new_game {'white' if player_color == chess.WHITE else 'black'}\n"
        print(f"Отправка в GUI: {start_cmd.strip()}")
        try:
            if gui_process.stdin:
                gui_process.stdin.write(start_cmd.encode('utf-8'))
                await gui_process.stdin.drain()
            else:
                raise IOError("Stdin GUI недоступен")
        except (BrokenPipeError, ConnectionResetError, IOError) as e:
            print(f"IPC Ошибка при старте GUI: {e}")
            raise

        chess_game_task = asyncio.create_task(chess_game_loop(), name="ChessGameLoop")
        print("Шахматный игровой цикл запущен.")

    except Exception as e:
        print(f"Критическая ошибка старта шахмат: {e}")
        await speak_text("Ой, что-то сломалось при запуске шахмат.")
        if gui_process and gui_process.returncode is None:
            try:
                gui_process.terminate()
                await gui_process.wait()
            except Exception as e_term:
                print(f"Ошибка терминации GUI при сбое старта: {e_term}")
        gui_process = None
        is_chess_game_active = False
        chess_game_task = None
    finally:
        is_processing_response = False
        print("[INFO] Блокировка снята после попытки старта.")


async def stop_chess_game(reason="Игра остановлена."):
    global is_chess_game_active, chess_board, player_color, bot_color
    global gui_process, chess_game_task, ipc_player_move_queue, last_activity_time

    print(f"Попытка остановки шахмат. Причина: {reason}")
    if not is_chess_game_active:
        print("Игра не активна.")
        return

    was_active = is_chess_game_active
    is_chess_game_active = False  # Сначала флаг

    if chess_game_task and not chess_game_task.done():
        print("Отмена игрового цикла...")
        chess_game_task.cancel()
        try:
            await chess_game_task
        except asyncio.CancelledError:
            print("Игровой цикл отменен.")
        except Exception as e:
            print(f"Ошибка ожидания отмены цикла: {e}")
        chess_game_task = None

    if gui_process and gui_process.returncode is None:
        print(f"Остановка GUI (PID: {gui_process.pid})...")
        try:
            stop_cmd = "stop_game\n"
            try:
                if gui_process.stdin:
                    gui_process.stdin.write(stop_cmd.encode('utf-8'))
                    await gui_process.stdin.drain()
                    await asyncio.sleep(0.5)
                else:
                    print("IPC: stdin GUI недоступен для 'stop_game'.")
            except (BrokenPipeError, ConnectionResetError, AttributeError):
                print("IPC: Канал в GUI закрыт/ошибка при 'stop_game'.")
            except Exception as e_ipc_stop:
                print(f"Ошибка отправки команды stop в GUI: {e_ipc_stop}")

            if gui_process.returncode is None:
                gui_process.terminate()
                await gui_process.wait()
                print("GUI терминирован.")
            else:
                print("GUI уже завершился.")
        except ProcessLookupError:
            print("GUI процесс не найден.")
        except Exception as e:
            print(f"Ошибка остановки GUI: {e}")
        finally:
            gui_process = None

    # await close_chess_engine() # Опционально

    chess_board = None;
    player_color = None;
    bot_color = None
    while not ipc_player_move_queue.empty():
        ipc_player_move_queue.get_nowait()

    print(f"Шахматная игра остановлена. Причина: {reason}")
    if was_active:
        await speak_text(reason)
    last_activity_time = time.time()
    print("[INFO] Реакция на чат и монологи снова ВОЗМОЖНЫ.")


async def parse_chess_move_from_text(text):
    global chess_board
    if not chess_board:
        return None

    text_lower = text.lower().strip()
    san_pattern = r'\b([KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?|O-O(?:-O)?)[+#]?\b'
    uci_pattern = r'\b[a-h][1-8][a-h][1-8][qrbn]?\b'
    potential_moves = re.findall(uci_pattern, text_lower) + re.findall(san_pattern, text_lower)

    print(f"Потенциальные ходы в '{text_lower}': {potential_moves}")
    if not potential_moves:
        return None

    for move_str in potential_moves:
        move = None
        try:
            move = chess_board.parse_uci(move_str)
        except ValueError:
            try:
                move = chess_board.parse_san(move_str)
            except ValueError:
                continue

        if move and chess_board.is_legal(move):
            print(f"Распознан легальный ход: {move.uci()}")
            return move
    print(f"Не найдено легальных ходов в: '{text}'")
    return None


async def get_chess_commentary(move: chess.Move, player_name: str):
    global chess_board, BOT_NAME_FOR_CHECK, is_processing_response
    if not chess_board or not move: return None
    if is_processing_response:
        print("[Commentary] Пропуск, бот занят.")
        return None

    is_processing_response = True
    commentary = None
    try:
        move_san = chess_board.san(move)
        prompt = f"Шахматы: {player_name} сделал ход: {move_san}. " \
                 f"FEN: {chess_board.fen()}. "
        if chess_board.is_checkmate():
            prompt += "Мат! "
        elif chess_board.is_stalemate():
            prompt += "Пат! "
        elif chess_board.is_check():
            prompt += "Шах! "
        elif chess_board.is_capture(move):
            prompt += "Взятие. "
        prompt += f"Я {BOT_NAME_FOR_CHECK}. Дай краткий (1-2 предл.), остроумный/аналитический комментарий к ходу."

        commentary = await get_ollama_response(prompt)
        if commentary:
            print(f"Комментарий к ходу {move_san}: {commentary}")
        else:
            print(f"Не сгенерирован комментарий к {move_san}.")
    except Exception as e:
        print(f"Ошибка комментария: {e}")
    finally:
        is_processing_response = False
        return commentary


async def chess_game_loop():
    global is_chess_game_active, chess_board, chess_engine, player_color, bot_color
    global gui_process, ipc_player_move_queue, is_processing_response

    print("Игровой цикл: Запущен.")
    # Задача чтения вывода GUI запускается здесь
    gui_reader_task = asyncio.create_task(read_gui_output(gui_process.stdout), name="GuiOutputReader")

    while is_chess_game_active:
        try:
            if chess_board.is_game_over(claim_draw=True):
                outcome = chess_board.outcome(claim_draw=True)
                winner = outcome.winner
                reason = f"Игра завершена! {outcome.termination.name.capitalize()}. "
                if winner is not None:
                    winner_name = "Белые" if winner == chess.WHITE else "Черные"
                    reason += f"Победили {winner_name}."
                    reason += " Я выиграл!" if winner == bot_color else " Ты выиграл!"
                else:
                    reason += "Ничья."
                print(f"Игровой цикл: {reason}")
                await stop_chess_game(reason)
                break

            move_made = None
            commentary_source = ""

            if chess_board.turn == player_color:  # Ход Игрока
                print("Игровой цикл: Ожидание хода игрока...")
                try:
                    player_move_uci = await asyncio.wait_for(ipc_player_move_queue.get(), timeout=600.0)
                    if player_move_uci is None:
                        await stop_chess_game("Сигнал остановки.")
                        break
                    move = chess.Move.from_uci(player_move_uci)
                    if chess_board.is_legal(move):
                        move_made = move
                        commentary_source = "Степан"
                    else:
                        print(f"Нелегальный ход: {player_move_uci}")
                        await speak_text("Так ходить нельзя.")
                        continue  # Ждем следующий ход
                except asyncio.TimeoutError:
                    await stop_chess_game("Время вышло.")
                    break
                except asyncio.CancelledError:
                    print("Ожидание игрока отменено.")
                    break
                except Exception as e:
                    print(f"Ошибка ожидания игрока: {e}")
                    await stop_chess_game("Ошибка.")
                    break
            else:  # Ход Бота
                print("Игровой цикл: Ход бота...")
                if not chess_engine:
                    await stop_chess_game("Ошибка движка.")
                    break
                if is_processing_response:
                    await asyncio.sleep(0.5)
                    continue

                try:
                    result = await asyncio.wait_for(chess_engine.play(chess_board, chess.engine.Limit(time=5.0)),
                                                    timeout=10.0)
                    move_made = result.move
                    commentary_source = f"Я ({BOT_NAME_FOR_CHECK})"
                    move_cmd = f"move {move_made.uci()}\n"
                    print(f"Отправка бот-хода в GUI: {move_cmd.strip()}")
                    try:
                        if gui_process and gui_process.returncode is None and gui_process.stdin:
                            gui_process.stdin.write(move_cmd.encode('utf-8'))
                            await gui_process.stdin.drain()
                        else:
                            raise BrokenPipeError("GUI недоступен")
                    except (BrokenPipeError, ConnectionResetError) as e:
                        print(f"IPC Ошибка: {e}")
                        await stop_chess_game("Потеря связи.")
                        break
                except Exception as e:
                    print(f"Ошибка хода бота ({type(e)}): {e}")
                    await stop_chess_game("Ошибка движка.")
                    break

            # Обработка сделанного хода
            if move_made:
                print(f"Игровой цикл: Обработка хода {move_made.uci()} от {commentary_source}")
                commentary = await get_chess_commentary(move_made, commentary_source)
                chess_board.push(move_made)
                print(f"Позиция:\n{chess_board}")
                if commentary:
                    await speak_text(commentary)

            await asyncio.sleep(0.1)

        except asyncio.CancelledError:
            print("Игровой цикл: Отменен.")
            break
        except Exception as e:
            print(f"Игровой цикл: Ошибка: {e}")
            await stop_chess_game("Критическая ошибка.")
            break

    print("Игровой цикл: Завершен.")
    if gui_reader_task and not gui_reader_task.done():
        gui_reader_task.cancel()
        try:
            await gui_reader_task
        except asyncio.CancelledError:
            pass
        except Exception as e_wait_reader:
            print(f"Ошибка ожидания gui_reader_task: {e_wait_reader}")
    print("Игровой цикл: Чтение GUI остановлено.")


async def read_gui_output(stream_reader):
    global ipc_player_move_queue, is_chess_game_active
    print("Чтение GUI: Запущено.")
    while is_chess_game_active and stream_reader and not stream_reader.at_eof():
        try:
            line_bytes = await stream_reader.readline()
            if not line_bytes:
                print("Чтение GUI: EOF.")
                break
            line = line_bytes.decode(errors='ignore').strip()
            if line.startswith("MOVE:"):
                move_uci = line.split(":", 1)[1].strip()
                if re.fullmatch(r'[a-h][1-8][a-h][1-8][qrbn]?', move_uci):
                    await ipc_player_move_queue.put(move_uci)
                else:
                    print(f"Чтение GUI: Некорректный ход: {move_uci}")
            elif line == "GUI_CLOSED":
                print("Чтение GUI: Сигнал закрытия.")
                break
        except asyncio.CancelledError:
            print("Чтение GUI: Отменено.");
            break
        except ConnectionResetError:
            print("Чтение GUI: Соединение разорвано.");
            break
        except Exception as e:
            print(f"Чтение GUI: Ошибка: {e}");
            await asyncio.sleep(1)
    print("Чтение GUI: Завершено.")
    if is_chess_game_active:
        await stop_chess_game("Графическая доска закрылась или потеряна связь.")


# --- Twitch бот ---
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
        if message.echo:
            return

        global chat_interaction_enabled, is_chess_game_active, is_processing_response
        global last_activity_time, BOT_NAME_FOR_CHECK, OBS_OUTPUT_FILE, stt_enabled, audio_queue

        # Игнор чата во время игры или если отключен хоткеем
        if is_chess_game_active or not chat_interaction_enabled:
            return
        if message.channel.name != self.target_channel_name:
            return

        # Реагируем только на упоминание или хайлайт
        content_lower = message.content.lower()
        trigger_parts = [p.lower() for p in BOT_NAME_FOR_CHECK.split() if len(p) > 2]
        mentioned = any(trig in content_lower for trig in trigger_parts)
        highlighted = message.tags.get('msg-id') == 'highlighted-message'
        if not mentioned and not highlighted:
            return

        current_time = datetime.datetime.now().strftime('%H:%M:%S')
        if is_processing_response:
            print(f"[{current_time}] Бот занят (чат). Игнор {message.author.name}.")
            return

        stt_was_on = False
        try:
            is_processing_response = True
            print(f"[{current_time}] НАЧАЛО обработки чата от {message.author.name}.")
            last_activity_time = time.time()
            print(f"[{current_time}] {message.author.name}: {message.content}")
            stt_was_on = stt_enabled
            if stt_enabled:
                print("[INFO] Откл STT для ответа (чат).")
                stt_enabled = False
            with audio_queue.mutex:
                audio_queue.queue.clear()
            try:
                open(OBS_OUTPUT_FILE, 'w').close()
            except Exception as e:
                print(f"[{current_time}] Ошибка очистки OBS: {e}")

            llm_response = await get_ollama_response(f"(Чат от {message.author.name}): {message.content}")
            if llm_response:
                print(f"[{current_time}] Ответ Ollama (чат): {llm_response}")
                try:
                    open(OBS_OUTPUT_FILE, 'w', encoding='utf-8').write(llm_response)
                except Exception as e:
                    print(f"[{current_time}] Ошибка записи в OBS: {e}")
                await speak_text(llm_response)
                with audio_queue.mutex:
                    audio_queue.queue.clear()
                last_activity_time = time.time()
            else:
                print(f"[{current_time}] Нет ответа Ollama для {message.author.name}.")
            if stt_was_on:
                print("[INFO] Вкл STT после ответа (чат).")
                stt_enabled = True
        except Exception as e:
            print(f"[{current_time}] КРИТ. ОШИБКА event_message: {e}")
            if stt_was_on:  # Пытаемся восстановить STT
                stt_enabled = True
        finally:
            is_processing_response = False
            print(f"[{current_time}] КОНЕЦ обработки чата от {message.author.name}.")


# --- Цикл обработки STT ---
async def stt_processing_loop():
    global audio_queue, recording_active, stt_model, stt_enabled, is_processing_response, is_chess_game_active
    silence_blocks = int(VAD_SILENCE_TIMEOUT_MS / (BLOCKSIZE / SOURCE_SAMPLE_RATE * 1000))
    min_speech = int(VAD_MIN_SPEECH_MS / (BLOCKSIZE / SOURCE_SAMPLE_RATE * 1000))
    pad_blocks = int(VAD_SPEECH_PAD_MS / (BLOCKSIZE / SOURCE_SAMPLE_RATE * 1000))
    is_speaking, silence_count, speech_buf, pad_buf = False, 0, [], []
    print("Цикл STT: Запущен.");
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
        if len(pad_buf) > pad_blocks * 2:
            pad_buf.pop(0)
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
    print("Цикл STT: Остановлен.")


# --- Обработка распознанной речи ---
async def process_recognized_speech(audio_buffer_list, source_id="STT"):
    """Обрабатывает распознанный текст: команды, ходы, обычный диалог."""
    global is_processing_response, stt_enabled, audio_queue, last_activity_time
    global is_chess_game_active, ipc_player_move_queue, OBS_OUTPUT_FILE

    current_time = datetime.datetime.now().strftime('%H:%M:%S')

    # Распознаем речь
    full_audio = np.concatenate(audio_buffer_list, axis=0)
    mono = full_audio.mean(axis=1) if SOURCE_CHANNELS > 1 else full_audio
    resampled = resample_audio(mono, SOURCE_SAMPLE_RATE, TARGET_SAMPLE_RATE)
    recognized_text = None
    if resampled is not None and resampled.size > 0:
        recognized_text = await asyncio.to_thread(transcribe_audio_faster_whisper, resampled)

    if not recognized_text:
        print(f"STT: Не распознано ({source_id}).")
        return  # Ничего не делаем, если речь не распознана

    last_activity_time = time.time()
    print(f"STT Распознано ({source_id}): {recognized_text}")
    text_lower = recognized_text.lower()

    # --- ПРОВЕРКА ШАХМАТНЫХ КОМАНД ---
    start_phrases = ["джордж давай в шахматы", "сыграем в шахматы", "запусти шахматы"]
    stop_phrases = ["джордж хватит играть", "стоп игра", "останови шахматы", "закончить партию"]

    if any(p in text_lower for p in start_phrases):
        print("Голос: СТАРТ шахмат.")
        await start_chess_game()
        return  # Команда обработана, выходим

    if any(p in text_lower for p in stop_phrases):
        print("Голос: СТОП шахмат.")
        await stop_chess_game()
        return  # Команда обработана, выходим

    # --- ПРОВЕРКА ШАХМАТНОГО ХОДА (ЕСЛИ ИГРА ИДЕТ) ---
    if is_chess_game_active:
        print("Игра активна, парсинг хода...")
        parsed_move = await parse_chess_move_from_text(recognized_text)
        if parsed_move:
            print(f"Голос: Ход {parsed_move.uci()}. В очередь.")
            # Не ставим флаг, просто отправляем ход в игровой цикл
            await ipc_player_move_queue.put(parsed_move.uci())
            return  # Ход обработан, выходим
        else:
            # Если игра идет, но это не ход и не команда стоп - просто игнорируем,
            # чтобы не забивать эфир сообщениями "не понял ход" на каждое слово.
            # Или можно добавить короткое TTS-сообщение об ошибке, НО нужно ставить флаг
            print("Не распознан ход во время активной игры.")
            # Если нужна реакция, то ставим флаг и говорим:
            # if not is_processing_response: # Проверяем, не занят ли уже TTS
            #     try:
            #         is_processing_response = True
            #         await speak_text("Не понял твой ход, Степан.")
            #     finally:
            #         is_processing_response = False
            return  # Выходим

    # --- ЕСЛИ ЭТО НЕ ШАХМАТНАЯ КОМАНДА/ХОД - ОБЫЧНАЯ ОБРАБОТКА РЕЧИ ---
    # Вот теперь можно ставить флаг, т.к. мы будем обращаться к Ollama/TTS
    if is_processing_response:
        print(f"[{current_time}] Бот занят (перед обычной обработкой). Игнор {source_id}.")
        return  # Выходим, если бот уже занят другим процессом

    stt_was_initially_enabled = stt_enabled
    try:
        is_processing_response = True  # Ставим флаг ЗДЕСЬ
        print(f"[{current_time}] НАЧАЛО обработки ОБЫЧНОЙ речи ({source_id}).")
        print("Обработка как обычное голосовое сообщение...")

        # Отключаем STT на время ответа БОТА
        if stt_enabled:
            print(f"[INFO] Откл STT для ответа ({source_id}).")
            stt_enabled = False
        with audio_queue.mutex:
            audio_queue.queue.clear()

        try:
            open(OBS_OUTPUT_FILE, 'w').close()  # Очистка OBS
        except Exception as e:
            print(f"[{current_time}] Ошибка очистки OBS: {e}")

        llm_response = await get_ollama_response(f"(Голос Степана): {recognized_text}")
        if llm_response:
            print(f"[{current_time}] Ответ Ollama ({source_id}): {llm_response}")
            try:
                open(OBS_OUTPUT_FILE, 'w', encoding='utf-8').write(llm_response)
            except Exception as e:
                print(f"[{current_time}] Ошибка записи в OBS: {e}")

            # TTS для ответа (speak_text сам управляет tts_lock)
            await speak_text(llm_response)
            with audio_queue.mutex:
                audio_queue.queue.clear()
            last_activity_time = time.time()  # Обновляем время активности
        else:
            print(f"[{current_time}] Нет ответа Ollama ({source_id}).")

    except Exception as e:
        print(f"[{current_time}] КРИТ. ОШИБКА process_speech ({source_id}): {e}")
    finally:
        is_processing_response = False  # Снимаем основную блокировку
        # Восстанавливаем STT в состояние, которое было ДО начала этой функции
        stt_enabled = stt_was_initially_enabled
        print(f"[{current_time}] КОНЕЦ обработки речи ({source_id}). STT: {stt_enabled}.")


# --- Цикл монологов ---
async def monologue_loop():
    global last_activity_time, recording_active, stt_enabled, BOT_NAME_FOR_CHECK
    global audio_queue, is_processing_response, is_chess_game_active, chat_interaction_enabled

    print("Цикл монологов: Запущен.")
    while recording_active.is_set():
        await asyncio.sleep(15)
        if is_chess_game_active or is_processing_response or not chat_interaction_enabled:
            continue
        if time.time() - last_activity_time > INACTIVITY_THRESHOLD_SECONDS:
            current_time = datetime.datetime.now().strftime('%H:%M:%S')
            if is_processing_response or is_chess_game_active or not chat_interaction_enabled:
                continue

            stt_was_initially_enabled = stt_enabled
            try:
                is_processing_response = True
                print(f"[{current_time}] Запуск монолога...")
                if stt_enabled:
                    print("[INFO] Откл STT для монолога.")
                    stt_enabled = False
                with audio_queue.mutex:
                    audio_queue.queue.clear()

                prompt = f"Сгенерируй короткую (1-2 предл.) реплику от {BOT_NAME_FOR_CHECK} для заполнения тишины."
                llm_response = await get_ollama_response(prompt)
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
                stt_enabled = stt_was_initially_enabled  # Восстанавливаем STT
                print(f"[{current_time}] КОНЕЦ монолога. STT: {stt_enabled}.")
    print("Цикл монологов: Остановлен.")


# --- Поток хоткеев ---
def hotkey_listener_thread():
    stt_hotkey = 'ctrl+;'
    chat_hotkey = "ctrl+apostrophe"
    reg_stt, reg_chat = False, False
    try:
        print(f"\nХоткей STT: '{stt_hotkey}', Хоткей Чата: '{chat_hotkey}'")
        keyboard.add_hotkey(stt_hotkey, toggle_stt);
        reg_stt = True
        keyboard.add_hotkey(chat_hotkey, toggle_chat_interaction);
        reg_chat = True
        while recording_active.is_set():
            time.sleep(0.5)
        print("Hotkey listener: завершение.")
    except ImportError:
        print("\nОШИБКА: 'keyboard' не найден.");
        return
    except Exception as e:
        print(f"\nОшибка hotkey_listener: {e}");
    finally:
        try:
            if reg_stt: keyboard.remove_hotkey(stt_hotkey)
            if reg_chat: keyboard.remove_hotkey(chat_hotkey)
            print("Хоткеи удалены.")
        except Exception as e:
            print(f"Ошибка удаления хоткеев: {e}")
        print("Поток хоткеев завершен.")


# --- Основная функция ---
async def main_async():
    global chess_engine, is_chess_game_active, recording_active
    print("Запуск AI Twitch Bot...");
    if not TWITCH_ACCESS_TOKEN:
        print("ОШИБКА: Нет TWITCH_ACCESS_TOKEN!");
        return

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
                    task.print_stack()
                    if task in [twitch_task, stt_task]:
                        print("Критическая ошибка, инициирую остановку...")
                        recording_active.clear()
                elif task.cancelled():
                    print(f"Задача {task.get_name()} была отменена.")
                else:
                    print(f"Задача {task.get_name()} успешно завершилась.")
            except asyncio.CancelledError:
                print(f"Задача {task.get_name()} отменена (проверка).")
            except Exception as e:
                print(f"Ошибка проверки задачи {task.get_name()}: {e}")

        global chess_game_task
        if chess_game_task and chess_game_task.done():
            print("Игровой цикл завершился.")
            try:
                chess_game_task.result()  # Проверить исключения
            except Exception as e_chess_done:
                print(f"Ошибка игрового цикла: {e_chess_done}")
            chess_game_task = None

        if not recording_active.is_set() or not active_tasks:
            break
        await asyncio.sleep(1)

    print("\n" + "=" * 10 + " Завершение работы (main_async) " + "=" * 10)
    if is_chess_game_active:
        await stop_chess_game("Завершение бота.")
    current_tasks = asyncio.all_tasks()
    tasks_to_cancel = [t for t in current_tasks if not t.done() and t is not asyncio.current_task()]
    if tasks_to_cancel:
        print(f"Отмена {len(tasks_to_cancel)} задач...");
        for task in tasks_to_cancel: task.cancel()
        await asyncio.gather(*tasks_to_cancel, return_exceptions=True)
    if client and client.is_connected():
        await client.close()
        print("Клиент Twitch закрыт.")
    await close_chess_engine()
    print("Завершение main_async завершено.")


# --- Точка входа ---
if __name__ == "__main__":
    print("-" * 40);
    print("Запуск программы...")
    if not all([TWITCH_ACCESS_TOKEN, TWITCH_CHANNEL]):
        print("ОШИБКА: Заполните .env!");
        sys.exit(1)
    if not STOCKFISH_PATH or not os.path.exists(STOCKFISH_PATH):
        print(f"ПРЕДУПРЕЖДЕНИЕ: Stockfish не найден!");
        STOCKFISH_PATH = None
    if not GUI_SCRIPT_PATH or not os.path.exists(GUI_SCRIPT_PATH):
        print(f"ПРЕДУПРЕЖДЕНИЕ: Скрипт GUI не найден!");
        GUI_SCRIPT_PATH = None
    if not stt_model: print("Предупреждение: STT не загружена.")
    if not piper_sample_rate: print("Предупреждение: TTS не инициализирован.")
    print("-" * 40)

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
    print(f"Микрофон: {default_mic} ({mic_name}) | SR={SOURCE_SAMPLE_RATE}, Ch={SOURCE_CHANNELS}");
    print("-" * 40)

    recording_active.set()
    recorder = threading.Thread(target=audio_recording_thread, args=(default_mic,), daemon=True, name="AudioRecorder");
    recorder.start()
    hotkeys = None
    if 'keyboard' in sys.modules:
        hotkeys = threading.Thread(target=hotkey_listener_thread, daemon=True, name="HotkeyListener");
        hotkeys.start()
    else:
        print("Хоткеи не работают ('keyboard' не найден).")

    loop = asyncio.new_event_loop();
    asyncio.set_event_loop(loop)
    main_task = None
    try:
        print("Запуск главного цикла asyncio...");
        main_task = loop.create_task(main_async(), name="MainLoop")
        loop.run_until_complete(main_task)
    except KeyboardInterrupt:
        print("\nCtrl+C...");
        recording_active.clear()
    except Exception as e_loop:
        print(f"Критическая ошибка главного цикла: {e_loop}");
        recording_active.clear()
    finally:
        print("\n" + "=" * 10 + " Финальное завершение " + "=" * 10)
        recording_active.clear()

        print("Ожидание потоков (до 3 сек)...")
        threads = [t for t in [recorder, hotkeys] if t and t.is_alive()]
        for t in threads:
            t.join(timeout=3.0)

        print("Ожидание и отмена asyncio задач (до 3 сек)...")
        if main_task and not main_task.done():
            main_task.cancel()
        tasks = [t for t in asyncio.all_tasks(loop=loop) if not t.done()]
        if tasks:
            try:
                loop.run_until_complete(asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=3.0))
            except asyncio.TimeoutError:
                print("Таймаут ожидания завершения задач asyncio.")
            except Exception as e_gather:
                print(f"Ошибка при gather: {e_gather}")

        try:
            loop.run_until_complete(loop.shutdown_asyncgens())
        except Exception as e_gens:
            print(f"Ошибка shutdown_asyncgens: {e_gens}", file=sys.stderr)
        finally:
            if not loop.is_closed():
                loop.close()
                print("Цикл asyncio закрыт.")

        print("-" * 40 + "\nПрограмма завершена.\n" + "-" * 40)