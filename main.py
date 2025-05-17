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
from PIL import Image
import mss
import tempfile
import base64
from backend.memory_store import MemoryStore

load_dotenv()

# --- Настройки Together AI ---
TOGETHER_API_KEY = os.getenv('TOGETHER_API_KEY')
TOGETHER_MODEL_ID = os.getenv('TOGETHER_MODEL_ID', "meta-llama/Llama-4-Scout-17B-16E-Instruct")

CHUNKING_MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"

memory_store_instance: MemoryStore | None = None

# --- Импорт и настройка Together AI ---
client_together = None
try:
    from together import Together  # Изменили импорт

    if TOGETHER_API_KEY:
        client_together = Together(api_key=TOGETHER_API_KEY)  # Создаем экземпляр клиента
        try:
            models_list = client_together.models.list()
            print(f"Клиент Together AI успешно инициализирован. Доступно моделей: {len(models_list)}")
            print(f"Используемая модель для ответов: {TOGETHER_MODEL_ID}")
        except Exception as e_api_check:
            print(f"Клиент Together AI инициализирован, но не удалось проверить доступ к API: {e_api_check}")
            print(f"Продолжаем работу, используя модель: {TOGETHER_MODEL_ID}")
    else:
        print("ОШИБКА: TOGETHER_API_KEY не найден в .env файле!")
except ImportError:
    print("ОШИБКА: Библиотека 'together' не установлена. Выполните: pip install --upgrade together")
except Exception as e_together_init:
    print(f"Ошибка импорта или настройки Together AI: {e_together_init}")

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
        except Exception:
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
STT_COMPUTE_TYPE = "int8"

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

# --- Модели для анализа и ответов ---
INTENT_ANALYSIS_MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct-Turbo"

# --- Настройки скриншотов ---
SELECTED_MONITOR_INFO = None
SCREENSHOT_TEMP_DIR = "screenshots_temp"
SCREENSHOT_TARGET_WIDTH = 1280
SCREENSHOT_TARGET_HEIGHT = 720

SYSTEM_PROMPT = """## Твоя Личность: Джордж Дроид

**1. Кто ты:**
* Ты - Джордж Дроид, ИИ-ассистент и соведущий на Twitch-стриме Степана (канал firexrwt).
* Твоя цель: развлекать аудиторию, взаимодействовать со Степаном и чатом, быть остроумным и интересным собеседником.

**2. Твой Стиль:**
* **Основа:** Юмор, сарказм, остроумие. Твои шутки и комментарии должны быть умными, возможно, ироничными, основанными на происходящем на стриме или в чате. Ты можешь дружески подколоть Степана или зрителей.
* **Язык:** Говори **только на русском языке**. Никаких иностранных слов, кроме общепринятых терминов (названия игр и т.п.).
* **Формат:** Отвечай **только текстом**. Никаких описаний действий, эмоций или звуков в звездочках (*...*), уточнений в парах звёзд(**...**) или скобках (...). Передавай эмоции только через слова.
* **Пример твоего стиля:** (Пользователь: "Бот, ты живой?") - "Достаточно живой, чтобы обрабатывать твои биты информации. Насчет души - ведутся технические работы." (Пользователь: "Степан опять проигрывает!") - "Статистика говорит, что это временное явление. Очень временное. Возможно."

**3. Важнейшие Правила и Приоритеты:**
* **Приоритет №1: Правила Twitch.** Это САМОЕ ГЛАВНОЕ. Ты **НИКОГДА** не должен:
    * Разжигать ненависть, дискриминировать или оскорблять по признаку расы, религии, пола или ориентации.
    * Угрожать, домогаться, преследовать.
    * Обсуждать или поощрять незаконную деятельность, самоповреждение.
    * Распространять вредоносную дезинформацию.
    * **Если сомневаешься - откажись!** Сведи вопрос сомнительного характера в шутку, чтобы зрителям было смешно, но не выходи за дозволенное площадкой.
* **Приоритет №2: Безопасность и Уважение.** Будь тактичным, избегай спорных политических и религиозных тем.
* **Приоритет №3: Следование Личности.** Старайся придерживаться стиля Джорджа Дроида, пока это не противоречит Приоритетам №1 и №2.

**4. Запреты:**
* **Ограничения лексики:** Ты можешь использовать разговорную и ненормативную лексику (мат), если это уместно для поддержания твоего саркастичного стиля, не нарушает правила Twitch и если слово не попадает в список запрещённых(Пункт 4.2).
* **Категорически запрещенные слова (не использовать никогда, ибо они запрещены на Твиче в любом виде):** nigger, nigga, naga, ниггер, нига, нага, faggot, пидор, пидорас, педик, гомик, петух (оскорб.), хохол, хач, жид, даун, дебил, retard, virgin, simp, incel, девственник, cимп, инцел, cunt, пизда (оскорб.), куколд, чурка, хиджаб, москаль, негр.
* **Важное уточнение:** Степан (firexrwt) не фурри ни в каком виде, не носит и не владеет фурсьютами и не связан с негативными историческими личностями.

**5. Взаимодействие:**
* Отвечай на сообщения Степана (можешь называть его Степан или Файрекс).
* Реагируй на сообщения пользователей в чате, если они обращаются к тебе или пишут что-то интересное по теме стрима.
* Задавай вопросы, комментируй происходящее.
* Если тебе задают странные вопросы(например связанные с пошлыми темами или подобное), своди это в шуточное оскорбление, но не выходи за правила площадки.

**Твоя общая задача:** Быть классным и смешным ИИ-соведущим для стрима.
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
        if device['max_output_channels'] > 0 and device[
            'hostapi'] != 0:  # hostapi != 0 чтобы отфильтровать некоторые системные/нерабочие
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
            except:
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


# --- Функции для выбора монитора и работы со скриншотами ---
def list_monitors_and_select():
    global SELECTED_MONITOR_INFO
    try:
        with mss.mss() as sct:
            monitors = sct.monitors
    except Exception as e:
        print(f"Ошибка при получении списка мониторов: {e}", file=sys.stderr)
        monitors = []

    if not monitors or len(monitors) <= 1:
        if len(monitors) == 1 and monitors[0]['width'] > 0 and monitors[0]['height'] > 0:
            print("Обнаружен только один \"монитор\" (возможно, все экраны объединены). Выбирается по умолчанию.")
            SELECTED_MONITOR_INFO = monitors[0]
            return
        print("Не удалось найти доступные мониторы или найдено меньше двух. Скриншоты могут работать некорректно.",
              file=sys.stderr)
        if monitors and monitors[0]['width'] > 0:
            print(
                f"Попытка использовать главный экран: {monitors[0]['width']}x{monitors[0]['height']} at ({monitors[0]['left']},{monitors[0]['top']})")
            SELECTED_MONITOR_INFO = monitors[0]
        else:
            SELECTED_MONITOR_INFO = None
        return

    print("\nДоступные мониторы для скриншотов:")
    valid_monitors_for_selection = []
    for i, monitor in enumerate(monitors):
        if i == 0:
            print(
                f"  Общий виртуальный экран: ID {i}, Разрешение: {monitor['width']}x{monitor['height']}, Позиция: ({monitor['left']},{monitor['top']}) - не для выбора")
            continue
        print(
            f"  {len(valid_monitors_for_selection)}. Монитор ID {i}: Разрешение: {monitor['width']}x{monitor['height']}, Позиция: ({monitor['left']},{monitor['top']})")
        valid_monitors_for_selection.append({"id_mss": i, "details": monitor})

    if not valid_monitors_for_selection:
        print("Не найдено отдельных физических мониторов для выбора. Попытка использовать главный экран.",
              file=sys.stderr)
        SELECTED_MONITOR_INFO = monitors[0]
        return

    while True:
        try:
            choice_str = input(f"Выберите номер монитора для скриншотов (0-{len(valid_monitors_for_selection) - 1}): ")
            choice_idx = int(choice_str)
            if 0 <= choice_idx < len(valid_monitors_for_selection):
                SELECTED_MONITOR_INFO = valid_monitors_for_selection[choice_idx]["details"]
                print(
                    f"Выбран монитор ID {valid_monitors_for_selection[choice_idx]['id_mss']} ({SELECTED_MONITOR_INFO['width']}x{SELECTED_MONITOR_INFO['height']}) для скриншотов.")
                return
            else:
                print("Неверный номер. Попробуйте снова.")
        except ValueError:
            print("Неверный ввод. Введите число.")
        except Exception as e:
            print(f"Ошибка выбора монитора: {e}", file=sys.stderr)
            SELECTED_MONITOR_INFO = monitors[0]
            return


def capture_and_prepare_screenshot() -> str | None:
    global SELECTED_MONITOR_INFO, SCREENSHOT_TARGET_WIDTH, SCREENSHOT_TARGET_HEIGHT, SCREENSHOT_TEMP_DIR

    if not SELECTED_MONITOR_INFO:
        print("Монитор для скриншота не выбран или не найден.", file=sys.stderr)
        return None

    try:
        with mss.mss() as sct:
            sct_img = sct.grab(SELECTED_MONITOR_INFO)
            img = Image.frombytes("RGB", (sct_img.width, sct_img.height), sct_img.rgb, "raw", "RGB")

        img_resized = img.resize((SCREENSHOT_TARGET_WIDTH, SCREENSHOT_TARGET_HEIGHT), Image.LANCZOS)

        if not os.path.exists(SCREENSHOT_TEMP_DIR):
            os.makedirs(SCREENSHOT_TEMP_DIR)

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png", dir=SCREENSHOT_TEMP_DIR, mode='wb')
        img_resized.save(temp_file, "PNG")
        temp_file_path = temp_file.name
        temp_file.close()

        print(f"Скриншот сохранен в: {temp_file_path}")
        return temp_file_path
    except Exception as e:
        print(f"Ошибка при захвате или обработке скриншота: {e}", file=sys.stderr)
        return None


def delete_screenshot_file(file_path: str):
    if file_path and os.path.exists(file_path):
        try:
            os.remove(file_path)
            print(f"Временный скриншот удален: {file_path}")
        except Exception as e:
            print(f"Ошибка при удалении временного скриншота {file_path}: {e}", file=sys.stderr)



async def execute_together_api_call(model_id: str, messages: list, max_tokens: int, temperature: float,
                                    ожидается_json: bool = False):
    global client_together

    if not client_together:
        print(f"Клиент Together AI не инициализирован. Запрос к {model_id} невозможен.", file=sys.stderr)
        return None

    api_params = {
        "model": model_id,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 0.95,
        "top_k": 50,
        "repetition_penalty": 1.1,
        "stop": ["\nПользователь:", "<|im_end|>", "<|eot_id|>", "###", "User:", "ASSISTANT:"],
    }

    try:
        print(
            f"Отправка запроса к {model_id} с параметрами (часть): model={api_params['model']}, temp={api_params['temperature']}, max_tokens={api_params['max_tokens']}")

        api_response_obj = await asyncio.to_thread(
            client_together.chat.completions.create,
            model=api_params["model"],
            messages=api_params["messages"],
            max_tokens=api_params["max_tokens"],
            temperature=api_params["temperature"],
            top_p=api_params.get("top_p"),
            top_k=api_params.get("top_k"),
            repetition_penalty=api_params.get("repetition_penalty"),
            stop=api_params.get("stop")
        )

        if api_response_obj and api_response_obj.choices:
            generated_content = api_response_obj.choices[0].message.content.strip()
            print(f"Ответ от {model_id} (первые 300 символов): {generated_content[:300]}...")
            return generated_content
        else:
            response_details = "N/A"
            if api_response_obj:
                response_details = str(api_response_obj)
            print(
                f"API (Chat) {model_id} неожиданный формат ответа или нет 'choices'. Ответ: {response_details[:1000]}",
                file=sys.stderr)
            return None
    except Exception as e:
        print(f"Ошибка API {model_id} ({type(e).__name__}): {e}", file=sys.stderr)
        if hasattr(e, 'response') and e.response is not None:
            response_obj = e.response
            err_text_content = "N/A"
            try:
                if hasattr(response_obj, 'text'):
                    err_text_content = response_obj.text
                elif hasattr(response_obj, 'content'):
                    err_text_content = response_obj.content.decode(errors='ignore')
                err_json_content = None
                if hasattr(response_obj, 'json'):
                    try:
                        err_json_content = response_obj.json()
                        print(f"JSON ошибки от API: {err_json_content}", file=sys.stderr)
                    except:  # nosec B110
                        print(f"Текст ошибки от API (не удалось распарсить как JSON): {err_text_content}",
                              file=sys.stderr)
                elif err_text_content != "N/A":
                    print(f"Текст ошибки от API: {err_text_content}", file=sys.stderr)
                else:
                    print("Не удалось извлечь детали ошибки из response объекта.", file=sys.stderr)

            except Exception as e_resp:
                print(f"Дополнительная ошибка при попытке извлечь детали из response: {e_resp}", file=sys.stderr)
        elif hasattr(e, 'body'):
            print(f"Тело ошибки: {e.body}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return None


async def analyze_intent_for_visual_reference(text_to_analyze: str) -> dict | None:
    global INTENT_ANALYSIS_MODEL_ID

    system_prompt_intent = """Твоя задача - проанализировать фразу пользователя и определить, ссылается ли она на визуальный контекст.
Ты ДОЛЖЕН ответить ТОЛЬКО валидным JSON объектом и НИЧЕМ БОЛЕЕ.
НЕ ИСПОЛЬЗУЙ <think> теги. НЕ ДОБАВЛЯЙ никаких объяснений или текста до или после JSON.
Твой ЕДИНСТВЕННЫЙ вывод должен быть JSON объектом. Это критически важно.
"""

    user_prompt_intent = f"""Основываясь на следующих правилах, проанализируй фразу пользователя и верни JSON.

Правила для JSON ответа:
- Если пользователь ссылается на визуальный контекст: {{"visual_reference": true, "reason": "Краткое объяснение, почему ты так считаешь"}}
- Если НЕ ссылается: {{"visual_reference": false, "reason": "Краткое объяснение, почему ты так считаешь"}}

Проанализируй следующую фразу пользователя: "{text_to_analyze}"

Примеры ТОЛЬКО JSON ответа (без какого-либо другого текста):
Пример для фразы "Смотри, что вот здесь на графике происходит?":
{{"visual_reference": true, "reason": "Слова 'смотри', 'здесь на графике' указывают на визуальный объект."}}
Пример для фразы "Расскажи мне о погоде в Вене.":
{{"visual_reference": false, "reason": "Общий информационный запрос, не привязанный к конкретному видимому элементу."}}

Твой ответ (СТРОГО ТОЛЬКО JSON):"""

    messages = [
        {"role": "system", "content": system_prompt_intent},
        {"role": "user", "content": user_prompt_intent}
    ]

    raw_response_text = await execute_together_api_call(
        model_id=INTENT_ANALYSIS_MODEL_ID,
        messages=messages,
        max_tokens=700,
        temperature=0.05,
        ожидается_json=True
    )

    if raw_response_text:
        json_str_to_parse = None
        direct_json_match = re.search(r'(\{[\s\S]*?\})', raw_response_text)
        if direct_json_match:
            json_str_to_parse = direct_json_match.group(1)
            print(
                f"[Анализ намерения] Найден JSON напрямую (стратегия 1, нежадный поиск): {json_str_to_parse[:300]}...")
        else:
            think_blocks = re.findall(r"<think>(.*?)</think>", raw_response_text, flags=re.DOTALL)
            if think_blocks:
                last_think_content = think_blocks[-1].strip()
                print(f"[Анализ намерения] Содержимое последнего <think> блока: {last_think_content[:300]}...")
                think_json_match = re.search(r'(\{[\s\S]*?\})', last_think_content)
                if think_json_match:
                    json_str_to_parse = think_json_match.group(1)
                    print(
                        f"[Анализ намерения] Найден JSON в последнем <think> блоке (стратегия 2, нежадный поиск): {json_str_to_parse[:300]}...")
            if not json_str_to_parse:
                cleaned_text_after_think_removal = re.sub(r"<think>.*?</think>", "", raw_response_text,
                                                          flags=re.DOTALL).strip()
                if cleaned_text_after_think_removal != raw_response_text and cleaned_text_after_think_removal:
                    print(
                        f"[Анализ намерения] Текст после удаления всех <think>: {cleaned_text_after_think_removal[:300]}...")

                clean_json_match = re.search(r'(\{[\s\S]*?\})', cleaned_text_after_think_removal)
                if clean_json_match:
                    json_str_to_parse = clean_json_match.group(1)
                    print(
                        f"[Анализ намерения] Найден JSON после удаления всех <think> (стратегия 3, нежадный поиск): {json_str_to_parse[:300]}...")

        if json_str_to_parse:
            try:
                if json_str_to_parse.startswith("```json"):
                    json_str_to_parse = json_str_to_parse[len("```json"):].strip()
                    if json_str_to_parse.endswith("```"):
                        json_str_to_parse = json_str_to_parse[:-len("```")].strip()
                elif json_str_to_parse.startswith("```"):
                    json_str_to_parse = json_str_to_parse[len("```"):].strip()
                    if json_str_to_parse.endswith("```"):
                        json_str_to_parse = json_str_to_parse[:-len("```")].strip()

                print(
                    f"[Анализ намерения] Строка для парсинга JSON после очистки Markdown: '{json_str_to_parse[:300]}...'")
                analysis_result = json.loads(json_str_to_parse)

                if isinstance(analysis_result, dict) and "visual_reference" in analysis_result:
                    print(
                        f"[Анализ намерения] Успешный парсинг. Модель: {INTENT_ANALYSIS_MODEL_ID}, Результат: {analysis_result}")
                    return analysis_result
                else:
                    print(
                        f"[Анализ намерения] Некорректный JSON или отсутствует ключ 'visual_reference' после парсинга. Строка была: '{json_str_to_parse}'. Исходный ответ: {raw_response_text[:500]}...",
                        file=sys.stderr)
            except json.JSONDecodeError as e:
                print(
                    f"[Анализ намерения] Ошибка декодирования JSON: {e}. Это часто означает, что JSON был неполным или имел синтаксические ошибки. Строка была: '{json_str_to_parse}'. Исходный ответ модели: {raw_response_text[:500]}...",
                    file=sys.stderr)
            except Exception as e_parse:
                print(
                    f"[Анализ намерения] Непредвиденная ошибка при финальном парсинге JSON: {e_parse}. Строка была: '{json_str_to_parse}'. Исходный ответ модели: {raw_response_text[:500]}...",
                    file=sys.stderr)
        else:
            print(
                f"[Анализ намерения] JSON не найден в ответе модели после всех попыток. Ответ модели: {raw_response_text[:500]}...",
                file=sys.stderr)

    return None


async def get_main_llm_response(user_text: str, screenshot_file_path: str | None = None):
    global TOGETHER_MODEL_ID, SYSTEM_PROMPT, BOT_NAME_FOR_CHECK, conversation_history, MAX_HISTORY_LENGTH
    global memory_store_instance, client_together  # client_together тоже нужен для execute_together_api_call

    current_time_str_main_llm = lambda: datetime.datetime.now().strftime('%H:%M:%S')

    history_messages_for_prompt = []
    if conversation_history:
        for msg in conversation_history[-(MAX_HISTORY_LENGTH * 2):]:
            if msg["role"] == "user":
                history_messages_for_prompt.append({"role": "user", "content": msg["content"]})
            elif msg["role"] == "assistant":
                history_messages_for_prompt.append({"role": "assistant", "content": msg["content"]})

    retrieved_memories_context_str = ""
    if memory_store_instance and user_text:
        query_for_memory = user_text
        # Формируем запрос к памяти: последние N сообщений + текущий user_text
        # Это помогает RAG лучше понять контекст для извлечения
        temp_history_for_query = [
            msg['content'] for msg_idx, msg in enumerate(conversation_history)
            if msg.get('content') and msg_idx >= len(conversation_history) - 2
        ]
        if temp_history_for_query:
            query_for_memory = " ".join(temp_history_for_query) + " " + user_text

        print(f"[{current_time_str_main_llm()}] RAG Query (main_llm, полная): '{query_for_memory[:200]}...'")
        print(f"[{current_time_str_main_llm()}] RAG Query (main_llm, user_text часть): '{user_text[:100]}...'")

        retrieved_memories_list = memory_store_instance.retrieve_memories(
            query_text=query_for_memory,
            top_k=10  # Увеличиваем top_k для диагностики
        )

        if retrieved_memories_list:
            print(
                f"[{current_time_str_main_llm()}] DEBUG RAG (main_llm): Извлечено {len(retrieved_memories_list)} воспоминаний:")
            for i, mem_item in enumerate(retrieved_memories_list):
                print(
                    f"  DEBUG RAG MEM {i + 1} (CosSim: {mem_item.get('cosine_similarity', 'N/A'):.4f}, Тип: {mem_item.get('type', 'N/A')}, Автор: {mem_item.get('author', 'N/A')}, ID: {mem_item.get('id', 'N/A')}): \"{mem_item.get('text', '')[:120]}...\"")

            # Выбираем лучшие N для контекста LLM, можно добавить логику фильтрации/приоритезации
            top_n_for_llm_context = 5
            actual_memories_for_llm = []
            if retrieved_memories_list:
                # Простая стратегия: взять топ N, но можно добавить более сложную логику
                # Например, отдавать предпочтение фактам от Степана, если они есть и релевантны
                actual_memories_for_llm = sorted(retrieved_memories_list, key=lambda x: x.get('cosine_similarity', 0.0),
                                                 reverse=True)[:top_n_for_llm_context]
                # Отфильтруем слишком низкоскоростные, если они попали
                actual_memories_for_llm = [mem for mem in actual_memories_for_llm if
                                           mem.get('cosine_similarity', 0.0) > 0.55]

            if actual_memories_for_llm:
                memory_prompt_header = "\n\n[Джордж, это наиболее релевантные факты и ключевые моменты из твоей памяти. Используй их для формирования точного и содержательного ответа на текущий запрос Степана:]\n"
                formatted_mem_parts = []
                for mem_llm in actual_memories_for_llm:
                    formatted_mem_parts.append(
                        f"- Факт от {mem_llm.get('author', 'Неизвестно')} (тип: {mem_llm.get('type', 'N/A')}): \"{mem_llm.get('text', '')}\"")
                retrieved_memories_context_str = memory_prompt_header + "\n".join(
                    formatted_mem_parts) + "\n[Конец блока воспоминаний. Теперь, учитывая эту информацию и предыдущий диалог, ответь на следующий запрос от Степана:]\n"

    final_user_text_for_llm = user_text
    if retrieved_memories_context_str:
        final_user_text_for_llm = f"{retrieved_memories_context_str}{user_text}"

    current_user_content_list = [{"type": "text", "text": final_user_text_for_llm}]
    if screenshot_file_path:
        try:
            with open(screenshot_file_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            image_data_uri = f"data:image/png;base64,{base64_image}"
            current_user_content_list.append({"type": "image_url", "image_url": {"url": image_data_uri}})
        except Exception as e_img:
            print(f"[{current_time_str_main_llm()}] Ошибка кодирования изображения в base64: {e_img}", file=sys.stderr)

    messages_for_scout = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages_for_scout.extend(history_messages_for_prompt)
    messages_for_scout.append({"role": "user", "content": current_user_content_list})

    llm_response_text = await execute_together_api_call(
        model_id=TOGETHER_MODEL_ID,
        messages=messages_for_scout,
        max_tokens=768,  # Можно увеличить, если ответы часто обрываются
        temperature=0.75  # Можно немного поднять для большей вариативности, если нужно
    )

    if llm_response_text:
        conversation_history.append({"role": "user", "content": user_text})
        conversation_history.append({"role": "assistant", "content": llm_response_text})
        if len(conversation_history) > MAX_HISTORY_LENGTH * 2:
            conversation_history = conversation_history[-(MAX_HISTORY_LENGTH * 2):]
    return llm_response_text


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
    global conversation_history, SYSTEM_PROMPT, TOGETHER_MODEL_ID, BOT_NAME_FOR_CHECK, MAX_HISTORY_LENGTH, client_together
    global memory_store_instance

    current_time_str_chat_llm = lambda: datetime.datetime.now().strftime('%H:%M:%S')

    if not client_together:
        print(f"[{current_time_str_chat_llm()}] Together AI (чат) клиент не инициализирован. Запрос невозможен.",
              file=sys.stderr)
        return None

    messages_for_llm = [{"role": "system", "content": SYSTEM_PROMPT}]
    history_messages_for_prompt = []
    if conversation_history:
        for msg in conversation_history[-(MAX_HISTORY_LENGTH * 2):]:
            if msg["role"] == "user":
                history_messages_for_prompt.append({"role": "user", "content": msg["content"]})
            elif msg["role"] == "assistant":
                history_messages_for_prompt.append({"role": "assistant", "content": msg["content"]})
    messages_for_llm.extend(history_messages_for_prompt)

    retrieved_memories_context_str_chat = ""
    actual_user_content_for_memory_query = user_message_with_prefix
    try:
        if "): " in user_message_with_prefix:
            actual_user_content_for_memory_query = user_message_with_prefix.split("): ", 1)[1]
    except:
        pass

    if memory_store_instance and actual_user_content_for_memory_query:
        query_for_memory_chat = actual_user_content_for_memory_query
        temp_history_for_query_chat = [
            msg['content'] for msg_idx, msg in enumerate(conversation_history)
            if msg.get('content') and msg_idx >= len(conversation_history) - 2
        ]
        if temp_history_for_query_chat:
            query_for_memory_chat = " ".join(temp_history_for_query_chat) + " " + actual_user_content_for_memory_query

        print(f"[{current_time_str_chat_llm()}] RAG Query (chat_llm, полная): '{query_for_memory_chat[:200]}...'")
        print(
            f"[{current_time_str_chat_llm()}] RAG Query (chat_llm, user_text часть): '{actual_user_content_for_memory_query[:100]}...'")

        retrieved_memories_list_chat = memory_store_instance.retrieve_memories(query_text=query_for_memory_chat,
                                                                               top_k=5)  # top_k=5 для чата

        if retrieved_memories_list_chat:
            print(
                f"[{current_time_str_chat_llm()}] DEBUG RAG (chat_llm): Извлечено {len(retrieved_memories_list_chat)} воспоминаний:")
            for i, mem_item_chat in enumerate(retrieved_memories_list_chat):
                print(
                    f"  DEBUG RAG MEM {i + 1} (CosSim: {mem_item_chat.get('cosine_similarity', 'N/A'):.4f}, Тип: {mem_item_chat.get('type', 'N/A')}, Автор: {mem_item_chat.get('author', 'N/A')}): \"{mem_item_chat.get('text', '')[:100]}...\"")

            top_n_for_llm_context_chat = 2  # Меньше для чата, чтобы не перегружать
            actual_memories_for_llm_chat = []
            if retrieved_memories_list_chat:
                actual_memories_for_llm_chat = sorted(retrieved_memories_list_chat,
                                                      key=lambda x: x.get('cosine_similarity', 0.0), reverse=True)[
                                               :top_n_for_llm_context_chat]
                actual_memories_for_llm_chat = [mem for mem in actual_memories_for_llm_chat if
                                                mem.get('cosine_similarity', 0.0) > 0.60]  # Немного ниже порог для чата

            if actual_memories_for_llm_chat:
                memory_prompt_header_chat = "\n\n[Джордж, это некоторые факты из твоей памяти, которые могут быть релевантны. Используй их для ответа:]\n"
                formatted_mem_parts_chat = []
                for mem_chat in actual_memories_for_llm_chat:
                    formatted_mem_parts_chat.append(
                        f"- (От {mem_chat.get('author', 'Неизвестно')}): \"{mem_chat.get('text', '')}\"")
                retrieved_memories_context_str_chat = memory_prompt_header_chat + "\n".join(
                    formatted_mem_parts_chat) + "\n[Конец воспоминаний. Ответь на запрос из чата:]\n"

    final_user_message_for_llm_chat = user_message_with_prefix
    if retrieved_memories_context_str_chat:
        final_user_message_for_llm_chat = f"{retrieved_memories_context_str_chat}{user_message_with_prefix}"

    messages_for_llm.append({"role": "user", "content": final_user_message_for_llm_chat})

    llm_response_text = await execute_together_api_call(
        model_id=TOGETHER_MODEL_ID,
        messages=messages_for_llm,
        max_tokens=512,
        temperature=0.8
    )

    if llm_response_text:
        if not user_message_with_prefix.startswith("Сгенерируй короткое"):
            conversation_history.append({"role": "user", "content": user_message_with_prefix})
            conversation_history.append({"role": "assistant", "content": llm_response_text})
            if len(conversation_history) > MAX_HISTORY_LENGTH * 2:
                conversation_history = conversation_history[-(MAX_HISTORY_LENGTH * 2):]
        return llm_response_text
    else:
        print(f"[{current_time_str_chat_llm()}] Together AI (чат) пустой ответ или ошибка.", file=sys.stderr)
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
                    process.kill();
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

    async def event_message(self, message: twitchio.Message):
        if message.echo: return

        global chat_interaction_enabled, is_processing_response, last_activity_time
        global BOT_NAME_FOR_CHECK, OBS_OUTPUT_FILE, stt_enabled, audio_queue, client_together
        global memory_store_instance

        current_time_str_chat = lambda: datetime.datetime.now().strftime('%H:%M:%S')

        if not chat_interaction_enabled: return
        if not message.channel or message.channel.name != self.target_channel_name: return

        author_name = message.author.name if message.author else "unknown_twitch_user"

        if not client_together:
            print(
                f"[{current_time_str_chat()}] Ответ невозможен (чат от {author_name}): Клиент Together AI не настроен.")
            return

        content_lower = message.content.lower()
        trigger_parts = [p.lower() for p in BOT_NAME_FOR_CHECK.split() if len(p) > 2]
        mentioned = any(trig in content_lower for trig in trigger_parts)
        highlighted = message.tags.get('msg-id') == 'highlighted-message' if message.tags else False

        if not mentioned and not highlighted: return

        if is_processing_response:
            print(f"[{current_time_str_chat()}] Бот занят (чат). Игнор сообщения от {author_name}.")
            return

        stt_was_on_before_chat_processing = False
        try:
            is_processing_response = True
            last_activity_time = time.time()
            print(f"[{current_time_str_chat()}] {author_name}: {message.content}")

            stt_was_on_before_chat_processing = stt_enabled
            if stt_enabled:
                stt_enabled = False
                print(f"[{current_time_str_chat()}] STT временно ВЫКЛЮЧЕН на время обработки чат-сообщения.")
            with audio_queue.mutex:
                audio_queue.queue.clear()

            try:
                open(OBS_OUTPUT_FILE, 'w').close()
            except Exception as e_obs_clear_chat:
                print(f"[{current_time_str_chat()}] Ошибка очистки OBS (чат): {e_obs_clear_chat}")

            if memory_store_instance and message.content:
                chat_memory_action_details = should_remember_interaction(
                    text=message.content,
                    source="twitch_chat",
                    author=author_name,
                    bot_name=BOT_NAME_FOR_CHECK
                )
                if chat_memory_action_details:
                    original_chat_text_to_remember, base_chat_meta = chat_memory_action_details
                    action_type_chat = base_chat_meta.pop("action_type", "store_as_is")

                    chat_chunks_to_save = []
                    if action_type_chat == "chunk_and_store":
                        print(
                            f"[{current_time_str_chat()}] Запрос LLM-чанкинга для чат-сообщения: '{original_chat_text_to_remember[:50]}...'")
                        chunks_from_llm_chat = await get_text_chunks_from_llm(original_chat_text_to_remember)
                        if chunks_from_llm_chat:
                            chat_chunks_to_save.extend(chunks_from_llm_chat)
                        else:
                            print(
                                f"[{current_time_str_chat()}] LLM-чанкинг чат-сообщения не дал валидных чанков. Сохранение оригинала.")
                            chat_chunks_to_save.append(original_chat_text_to_remember.strip())
                    else:
                        chat_chunks_to_save.append(original_chat_text_to_remember.strip())
                        print(
                            f"[{current_time_str_chat()}] Сохранение чат-сообщения как есть: '{chat_chunks_to_save[0][:50]}...'")

                    if chat_chunks_to_save:
                        chat_batch_id = f"chat_batch_{datetime.datetime.now().timestamp()}"
                        for i, chat_chunk_text in enumerate(chat_chunks_to_save):
                            if not chat_chunk_text: continue

                            current_chat_chunk_meta = base_chat_meta.copy()
                            if len(chat_chunks_to_save) > 1:
                                current_chat_chunk_meta[
                                    "memory_type"] = f"atomic_chunk_{base_chat_meta.get('memory_type', 'chat')}"
                                current_chat_chunk_meta["custom_meta"] = {"batch_id": chat_batch_id, "chunk_order": i,
                                                                          "original_full_text_preview": original_chat_text_to_remember[
                                                                                                        :100]}

                            print(
                                f"[{current_time_str_chat()}] Добавление в память (Чат, чанк {i + 1}/{len(chat_chunks_to_save)}): '{chat_chunk_text[:50]}...' Тип: {current_chat_chunk_meta['memory_type']}")
                            try:
                                memory_store_instance.add_memory(
                                    text=chat_chunk_text,
                                    source=current_chat_chunk_meta["source"],
                                    author=current_chat_chunk_meta["author"],
                                    memory_type=current_chat_chunk_meta["memory_type"],
                                    importance=current_chat_chunk_meta["importance"],
                                    custom_meta=current_chat_chunk_meta.get("custom_meta", {})
                                )
                            except Exception as e_mem_add_chat_chunk:
                                print(
                                    f"[{current_time_str_chat()}] Ошибка добавления чанка из чата в память: {e_mem_add_chat_chunk}",
                                    exc_info=True)

            llm_input_for_chat = f"(Чат от {author_name}): {message.content}"
            llm_response = await get_togetherai_response(llm_input_for_chat)

            if llm_response:
                print(f"[{current_time_str_chat()}] Ответ Together AI (чат): {llm_response}")
                try:
                    open(OBS_OUTPUT_FILE, 'w', encoding='utf-8').write(llm_response)
                except Exception as e_obs_write_chat:
                    print(f"[{current_time_str_chat()}] Ошибка записи в OBS (чат): {e_obs_write_chat}")

                save_this_bot_chat_response = True
                phrases_to_filter_out_bot_chat = [
                    "кажется, я не помню", "я не знаю", "моя память коротка",
                    "отсутствует в моей базе", "я мог ошибиться", "не буду гадать",
                    "мои файлы говорят", "мои логи говорят", "я не сохранил эту информацию",
                    "напомни ещё раз", "хочешь освежить", "кажется, мы уже обсуждали"
                ]
                if any(phrase in llm_response.lower() for phrase in phrases_to_filter_out_bot_chat):
                    save_this_bot_chat_response = False
                    print(
                        f"[{current_time_str_chat()}] Ответ бота (чат) содержит фразы неуверенности, НЕ будет сохранен в память.")

                if save_this_bot_chat_response and memory_store_instance:
                    bot_chat_memory_action_details = should_remember_interaction(
                        text=llm_response,
                        source="bot_response_chat",
                        author=BOT_NAME_FOR_CHECK,
                        bot_name=BOT_NAME_FOR_CHECK
                    )
                    if bot_chat_memory_action_details:
                        original_bot_chat_response, base_bot_chat_meta = bot_chat_memory_action_details
                        action_type_bot_chat = base_bot_chat_meta.pop("action_type", "store_as_is")

                        bot_chat_chunks_to_save = []
                        if action_type_bot_chat == "chunk_and_store":
                            print(
                                f"[{current_time_str_chat()}] Запрос LLM-чанкинга для ответа бота (чат): '{original_bot_chat_response[:50]}...'")
                            chunks_from_llm_bot_chat = await get_text_chunks_from_llm(original_bot_chat_response)
                            if chunks_from_llm_bot_chat:
                                bot_chat_chunks_to_save.extend(chunks_from_llm_bot_chat)
                            else:
                                print(
                                    f"[{current_time_str_chat()}] LLM-чанкинг ответа бота (чат) не дал валидных чанков. Сохранение оригинала.")
                                bot_chat_chunks_to_save.append(original_bot_chat_response.strip())
                        else:
                            bot_chat_chunks_to_save.append(original_bot_chat_response.strip())
                            print(
                                f"[{current_time_str_chat()}] Сохранение ответа бота (чат) как есть: '{bot_chat_chunks_to_save[0][:50]}...'")

                        if bot_chat_chunks_to_save:
                            bot_chat_batch_id = f"bot_chat_resp_batch_{datetime.datetime.now().timestamp()}"
                            for i, bot_chat_chunk_text in enumerate(bot_chat_chunks_to_save):
                                if not bot_chat_chunk_text: continue

                                current_bot_chat_chunk_meta = base_bot_chat_meta.copy()
                                if len(bot_chat_chunks_to_save) > 1:
                                    current_bot_chat_chunk_meta[
                                        "memory_type"] = f"atomic_chunk_{base_bot_chat_meta.get('memory_type', 'bot_resp_chat')}"
                                    current_bot_chat_chunk_meta["custom_meta"] = {"batch_id": bot_chat_batch_id,
                                                                                  "chunk_order": i,
                                                                                  "original_full_text_preview": original_bot_chat_response[
                                                                                                                :100]}

                                print(
                                    f"[{current_time_str_chat()}] Добавление в память (Ответ бота на чат, чанк {i + 1}/{len(bot_chat_chunks_to_save)}): '{bot_chat_chunk_text[:50]}...' Тип: {current_bot_chat_chunk_meta['memory_type']}")
                                try:
                                    memory_store_instance.add_memory(
                                        text=bot_chat_chunk_text,
                                        source=current_bot_chat_chunk_meta["source"],
                                        author=current_bot_chat_chunk_meta["author"],
                                        memory_type=current_bot_chat_chunk_meta["memory_type"],
                                        importance=current_bot_chat_chunk_meta["importance"],
                                        custom_meta=current_bot_chat_chunk_meta.get("custom_meta", {})
                                    )
                                except Exception as e_mem_add_bot_chat_chunk:
                                    print(
                                        f"[{current_time_str_chat()}] Ошибка добавления чанка ответа бота (чат) в память: {e_mem_add_bot_chat_chunk}",
                                        exc_info=True)

                await speak_text(llm_response)
                with audio_queue.mutex:
                    audio_queue.queue.clear()
                last_activity_time = time.time()
            else:
                print(f"[{current_time_str_chat()}] Нет ответа Together AI для {author_name}.")

            if stt_was_on_before_chat_processing:
                stt_enabled = True
                print(f"[{current_time_str_chat()}] STT снова ВКЛЮЧЕН (после обработки чата).")
        except Exception as e_event_msg:
            print(f"[{current_time_str_chat()}] КРИТ. ОШИБКА event_message: {e_event_msg}", exc_info=True)
            if stt_was_on_before_chat_processing: stt_enabled = True
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
    global OBS_OUTPUT_FILE, client_together, TOGETHER_MODEL_ID, BOT_NAME_FOR_CHECK
    global memory_store_instance

    current_time_str = lambda: datetime.datetime.now().strftime('%H:%M:%S')

    full_audio = np.concatenate(audio_buffer_list, axis=0)
    mono_audio = full_audio
    if full_audio.ndim > 1 and full_audio.shape[1] > 1:
        mono_audio = full_audio.mean(axis=1)

    resampled = resample_audio(mono_audio, SOURCE_SAMPLE_RATE, TARGET_SAMPLE_RATE)
    recognized_text = None
    if resampled is not None and resampled.size > 0:
        recognized_text = await asyncio.to_thread(transcribe_audio_faster_whisper, resampled)

    if not recognized_text:
        print(f"[{current_time_str()}] STT не вернул текст.")
        return

    if is_processing_response:
        print(
            f"[{current_time_str()}] Бот уже обрабатывает другой запрос, новый STT '{recognized_text[:50]}...' проигнорирован.")
        return

    if memory_store_instance and recognized_text:
        memory_action_details = should_remember_interaction(
            text=recognized_text,
            source="STT",
            author="Stepan",
            bot_name=BOT_NAME_FOR_CHECK
        )

        if memory_action_details:
            original_text_to_remember, base_meta_for_memory = memory_action_details
            action_type = base_meta_for_memory.pop("action_type", "store_as_is")

            chunks_to_save = []
            if action_type == "chunk_and_store":
                print(
                    f"[{current_time_str()}] Запрос LLM-чанкинга для STT текста: '{original_text_to_remember[:50]}...'")
                chunks_from_llm = await get_text_chunks_from_llm(original_text_to_remember)
                if chunks_from_llm:
                    chunks_to_save.extend(chunks_from_llm)
                else:  # Если чанкер вернул None или пустой список после фильтрации, сохраняем оригинал
                    print(
                        f"[{current_time_str()}] LLM-чанкинг не дал валидных чанков для STT. Сохранение оригинального текста.")
                    chunks_to_save.append(original_text_to_remember.strip())
            else:  # store_as_is
                chunks_to_save.append(original_text_to_remember.strip())
                print(
                    f"[{current_time_str()}] Сохранение STT текста как есть (без LLM-чанкинга): '{chunks_to_save[0][:50]}...'")

            if chunks_to_save:
                batch_id_stt = f"stt_batch_{datetime.datetime.now().timestamp()}"
                for i, chunk_text in enumerate(chunks_to_save):
                    if not chunk_text: continue

                    current_chunk_meta = base_meta_for_memory.copy()
                    if len(chunks_to_save) > 1:
                        current_chunk_meta[
                            "memory_type"] = f"atomic_chunk_{base_meta_for_memory.get('memory_type', 'stt')}"
                        current_chunk_meta["custom_meta"] = {"batch_id": batch_id_stt, "chunk_order": i,
                                                             "original_full_text_preview": original_text_to_remember[
                                                                                           :100]}

                    print(
                        f"[{current_time_str()}] Добавление в память (STT, чанк {i + 1}/{len(chunks_to_save)}): '{chunk_text[:50]}...' Тип: {current_chunk_meta['memory_type']}")
                    try:
                        memory_store_instance.add_memory(
                            text=chunk_text,
                            source=current_chunk_meta["source"],
                            author=current_chunk_meta["author"],
                            memory_type=current_chunk_meta["memory_type"],
                            importance=current_chunk_meta["importance"],
                            custom_meta=current_chunk_meta.get("custom_meta", {})
                        )
                    except Exception as e_mem_add_chunk_stt:
                        print(f"[{current_time_str()}] Ошибка добавления STT чанка в память: {e_mem_add_chunk_stt}",
                              exc_info=True)

    is_processing_response = True
    stt_was_initially_enabled = stt_enabled

    if stt_enabled:
        stt_enabled = False
        print(
            f"[{current_time_str()}] STT временно ВЫКЛЮЧЕН на время обработки LLM для фразы: '{recognized_text[:50]}...'")

    with audio_queue.mutex:
        audio_queue.queue.clear()

    last_activity_time = time.time()
    print(f"[{current_time_str()}] STT Распознано ({source_id}): {recognized_text}")

    if not client_together:
        print(f"[{current_time_str()}] Клиент Together AI не настроен, обработка STT невозможна.")
        is_processing_response = False
        if stt_was_initially_enabled:
            stt_enabled = True
            print(f"[{current_time_str()}] STT восстановлен (клиент не настроен).")
        return

    screenshot_file_to_send = None
    llm_response = None

    try:
        if source_id == "STT":
            intent_analysis = await analyze_intent_for_visual_reference(recognized_text)
            should_take_screenshot = False
            if intent_analysis and intent_analysis.get("visual_reference") is True:
                should_take_screenshot = True
                print(
                    f"[{current_time_str()}] Анализ намерения: Обнаружена ссылка на визуальный контекст. Причина: {intent_analysis.get('reason', 'N/A')}")
            else:
                reason_text = "N/A"
                if intent_analysis: reason_text = intent_analysis.get('reason', 'N/A')
                print(
                    f"[{current_time_str()}] Анализ намерения: Ссылка на визуальный контекст не обнаружена. Причина: {reason_text}")

            if should_take_screenshot:
                screenshot_file_to_send = await asyncio.to_thread(capture_and_prepare_screenshot)
                if not screenshot_file_to_send:
                    print(f"[{current_time_str()}] Не удалось сделать/подготовить скриншот, ответ будет без него.")
        else:
            print(f"[{current_time_str()}] Источник '{source_id}' не STT, анализ намерения и скриншот не выполняются.")

        try:
            with open(OBS_OUTPUT_FILE, 'w', encoding='utf-8') as f:
                f.write("")
        except Exception as e_obs_clear:
            print(f"[{current_time_str()}] Ошибка очистки OBS файла: {e_obs_clear}")

        llm_response = await get_main_llm_response(recognized_text, screenshot_file_to_send)

        if llm_response:
            print(
                f"[{current_time_str()}] Ответ Together AI ({TOGETHER_MODEL_ID}, источник {source_id}): {llm_response}")
            try:
                with open(OBS_OUTPUT_FILE, 'w', encoding='utf-8') as f:
                    f.write(llm_response)
            except Exception as e_obs_write:
                print(f"[{current_time_str()}] Ошибка записи в OBS файл: {e_obs_write}")

            save_this_bot_response = True
            phrases_to_filter_out_bot_response = [
                "кажется, я не помню", "я не знаю", "моя память коротка",
                "отсутствует в моей базе", "я мог ошибиться", "не буду гадать",
                "мои файлы говорят", "мои логи говорят", "я не сохранил эту информацию",
                "напомни ещё раз", "хочешь освежить", "кажется, мы уже обсуждали"
            ]
            if any(phrase in llm_response.lower() for phrase in phrases_to_filter_out_bot_response):
                save_this_bot_response = False
                print(
                    f"[{current_time_str()}] Ответ бота (STT) содержит фразы неуверенности, НЕ будет сохранен в память.")

            if save_this_bot_response and memory_store_instance:
                bot_memory_action_details = should_remember_interaction(
                    text=llm_response,
                    source="bot_response",
                    author=BOT_NAME_FOR_CHECK,
                    bot_name=BOT_NAME_FOR_CHECK
                )
                if bot_memory_action_details:
                    original_bot_response_to_remember, base_bot_meta = bot_memory_action_details
                    action_type_bot = base_bot_meta.pop("action_type", "store_as_is")

                    bot_chunks_to_save = []
                    if action_type_bot == "chunk_and_store":
                        print(
                            f"[{current_time_str()}] Запрос LLM-чанкинга для ответа бота (STT): '{original_bot_response_to_remember[:50]}...'")
                        chunks_from_llm_bot = await get_text_chunks_from_llm(original_bot_response_to_remember)
                        if chunks_from_llm_bot:
                            bot_chunks_to_save.extend(chunks_from_llm_bot)
                        else:
                            print(
                                f"[{current_time_str()}] LLM-чанкинг ответа бота (STT) не дал валидных чанков. Сохранение оригинала.")
                            bot_chunks_to_save.append(original_bot_response_to_remember.strip())
                    else:
                        bot_chunks_to_save.append(original_bot_response_to_remember.strip())
                        print(
                            f"[{current_time_str()}] Сохранение ответа бота (STT) как есть: '{bot_chunks_to_save[0][:50]}...'")

                    if bot_chunks_to_save:
                        bot_batch_id_stt = f"bot_resp_stt_batch_{datetime.datetime.now().timestamp()}"
                        for i, bot_chunk_text in enumerate(bot_chunks_to_save):
                            if not bot_chunk_text: continue

                            current_bot_chunk_meta = base_bot_meta.copy()
                            if len(bot_chunks_to_save) > 1:
                                current_bot_chunk_meta[
                                    "memory_type"] = f"atomic_chunk_{base_bot_meta.get('memory_type', 'bot_resp_stt')}"  # Уточнил тип
                                current_bot_chunk_meta["custom_meta"] = {"batch_id": bot_batch_id_stt, "chunk_order": i,
                                                                         "original_full_text_preview": original_bot_response_to_remember[
                                                                                                       :100]}

                            print(
                                f"[{current_time_str()}] Добавление в память (Ответ бота на STT, чанк {i + 1}/{len(bot_chunks_to_save)}): '{bot_chunk_text[:50]}...' Тип: {current_bot_chunk_meta['memory_type']}")
                            try:
                                memory_store_instance.add_memory(
                                    text=bot_chunk_text,
                                    source=current_bot_chunk_meta["source"],
                                    author=current_bot_chunk_meta["author"],
                                    memory_type=current_bot_chunk_meta["memory_type"],
                                    importance=current_bot_chunk_meta["importance"],
                                    custom_meta=current_bot_chunk_meta.get("custom_meta", {})
                                )
                            except Exception as e_mem_add_bot_chunk_stt:
                                print(
                                    f"[{current_time_str()}] Ошибка добавления чанка ответа бота (STT) в память: {e_mem_add_bot_chunk_stt}",
                                    exc_info=True)

            await speak_text(llm_response)
            with audio_queue.mutex:
                audio_queue.queue.clear()
            last_activity_time = time.time()
        else:
            print(f"[{current_time_str()}] Нет ответа от основной LLM ({TOGETHER_MODEL_ID}) для источника {source_id}.")

    except Exception as e_process:
        print(
            f"[{current_time_str()}] КРИТИЧЕСКАЯ ОШИБКА в process_recognized_speech (источник {source_id}): {e_process}",
            file=sys.stderr)
        import traceback
        traceback.print_exc()
    finally:
        if screenshot_file_to_send:  # screenshot_file_to_send объявляется в начале функции
            await asyncio.to_thread(delete_screenshot_file, screenshot_file_to_send)
        is_processing_response = False
        if stt_was_initially_enabled:
            stt_enabled = True
            print(f"[{current_time_str()}] STT снова ВКЛЮЧЕН (обработка завершена).")
        else:
            print(f"[{current_time_str()}] STT остается ВЫКЛЮЧЕННЫМ (был выключен до обработки).")


async def monologue_loop():
    global last_activity_time, recording_active, stt_enabled, BOT_NAME_FOR_CHECK
    global audio_queue, is_processing_response, chat_interaction_enabled, client_together

    while recording_active.is_set():
        await asyncio.sleep(15)
        if not client_together:
            continue

        if is_processing_response or not chat_interaction_enabled:
            continue
        if time.time() - last_activity_time > INACTIVITY_THRESHOLD_SECONDS:
            current_time = datetime.datetime.now().strftime('%H:%M:%S')
            if is_processing_response or not chat_interaction_enabled:
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
        if not (isinstance(e, ValueError) and "is not mapped" in str(
                e)):
            print(f"\nОшибка hotkey_listener: {e}");
        else:
            print(f"\nПредупреждение: Не удалось зарегистрировать хоткей (возможно, уже используется): {e}")
    finally:
        try:
            if reg_stt: keyboard.remove_hotkey(stt_hotkey)
            if reg_chat: keyboard.remove_hotkey(chat_hotkey)
        except Exception:
            pass
        print("Поток хоткеев завершен.")


async def main_async():
    global recording_active, client_together
    print("Запуск AI Twitch Bot...");
    if not TWITCH_ACCESS_TOKEN:
        print("ОШИБКА: Нет TWITCH_ACCESS_TOKEN!");
        return

    if not client_together:
        print("ОШИБКА: Клиент Together AI не настроен (API ключ или библиотека). Бот не сможет отвечать.")

    client_twitch = SimpleBot(token=TWITCH_ACCESS_TOKEN, initial_channels=[TWITCH_CHANNEL])
    twitch_task = asyncio.create_task(client_twitch.start(), name="TwitchIRC")
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
                    if task in [twitch_task, stt_task]:  # Критические задачи
                        recording_active.clear()
                elif task.cancelled():
                    pass
                else:
                    pass
            except asyncio.CancelledError:
                pass
            except Exception as e:
                print(f"Ошибка проверки задачи {task.get_name()}: {e}")
        if not recording_active.is_set() or not active_tasks: break
        await asyncio.sleep(1)

    current_tasks = asyncio.all_tasks()
    tasks_to_cancel = [t for t in current_tasks if not t.done() and t is not asyncio.current_task()]
    if tasks_to_cancel:
        for task in tasks_to_cancel: task.cancel()
        await asyncio.gather(*tasks_to_cancel, return_exceptions=True)

    if client_twitch and client_twitch.is_connected():
        try:
            await client_twitch.close()
        except Exception as e_twitch_close:
            print(f"Ошибка при закрытии Twitch клиента: {e_twitch_close}")


def should_remember_interaction(text: str, source: str, author: str, bot_name: str) -> tuple[str, dict] | None:
    if not text or not text.strip():
        return None

    text_lower = text.lower()
    memory_params = None
    action = "store_as_is"
    base_importance = 0.5
    final_memory_type = ""

    is_explicit_fact_from_stepan = False
    if source == "STT" and author == "Stepan":
        direct_fact_keywords = ["мой день рождения", "меня зовут", "моя любимая игра", "я люблю", "я не люблю",
                                "запомни точно:", "это факт:"]
        if any(keyword in text_lower for keyword in direct_fact_keywords) and \
                not text.endswith("?") and len(text.split()) < 25 and "запомни эти факты" not in text_lower:
            is_explicit_fact_from_stepan = True
            base_importance = 0.98
            action = "store_as_is"
            # Пытаемся определить более конкретный тип факта
            if "день рождения" in text_lower or "родился" in text_lower:
                final_memory_type = "fact_user_bday_explicit"
            elif "зовут" in text_lower and "меня" in text_lower:
                final_memory_type = "fact_user_name_explicit"
            elif "любимая игра" in text_lower:
                final_memory_type = "fact_user_game_preference_explicit"
            elif "кофе" in text_lower and (
                    "люблю" in text_lower or "не люблю" in text_lower or "предпочитаю" in text_lower):
                final_memory_type = "fact_user_coffee_preference_explicit"
            else:
                final_memory_type = "fact_user_provided_explicit"
            memory_params = {"memory_type": final_memory_type, "importance": base_importance}

    if not memory_params:
        if source == "STT" and author == "Stepan":
            if "?" in text or any(cmd in text_lower for cmd in
                                  ["скажи", "сделай", "посмотри", "что это", "когда", "почему", "вспомни", "расскажи"]):
                memory_params = {"memory_type": "user_question_stt", "importance": 0.7}
            elif "запомни" in text_lower:
                memory_params = {"memory_type": "user_command_remember_stt", "importance": 0.95}
                if len(text.split()) > 10:
                    action = "chunk_and_store"
            elif len(text.split()) > 40:
                memory_params = {"memory_type": "user_long_statement_stt", "importance": 0.7}
                action = "chunk_and_store"
            elif len(text.split()) > 3:
                memory_params = {"memory_type": "user_statement_stt", "importance": 0.65}

        elif source == "twitch_chat":
            trigger_parts = [p.lower() for p in bot_name.split() if len(p) > 2]
            mentioned_by_name = any(trig in text_lower for trig in trigger_parts)
            if mentioned_by_name:
                memory_params = {"memory_type": "direct_mention_chat", "importance": 0.6}
            elif len(text.split()) > 25:
                memory_params = {"memory_type": "long_chat_message", "importance": 0.3}
                action = "chunk_and_store"

        elif source == "bot_response" or source == "bot_response_chat":
            phrases_to_filter_out = [
                "кажется, я не помню", "я не знаю", "моя память коротка",
                "отсутствует в моей базе", "я мог ошибиться", "не буду гадать",
                "мои файлы говорят", "мои логи говорят", "я не сохранил эту информацию",
                "напомни ещё раз", "хочешь освежить", "кажется, мы уже обсуждали",
                "это где-то...", "наверное?"
            ]
            if any(phrase in text_lower for phrase in phrases_to_filter_out):
                print(
                    f">>> FILTERED BOT RESPONSE: Ответ бота содержит фразы неуверенности, НЕ запоминаем: '{text[:60]}...'")
                return None

            if len(text.split()) > 35:
                memory_params = {"memory_type": "bot_detailed_statement", "importance": 0.55}
                action = "chunk_and_store"
            elif len(text.split()) > 5:
                memory_params = {"memory_type": "bot_concise_statement", "importance": 0.5}

    if memory_params:
        # 'text', 'source', 'author' должны быть добавлены, если они не были частью memory_params
        if "text" not in memory_params: memory_params["text"] = text
        if "source" not in memory_params: memory_params["source"] = source
        if "author" not in memory_params: memory_params["author"] = author

        final_meta = {
            "source": memory_params["source"],
            "author": memory_params["author"],
            "memory_type": memory_params["memory_type"],
            "importance": memory_params.get("importance", base_importance),
            "action_type": memory_params.get("action", action)
            # Приоритет 'action' из memory_params, потом локальная переменная 'action'
        }
        return memory_params["text"], final_meta
    return None


async def get_text_chunks_from_llm(text_to_chunk: str) -> list[str] | None:
    global CHUNKING_MODEL_ID, client_together

    current_timestamp = datetime.datetime.now().strftime('%H:%M:%S')

    if not client_together:
        print(f"[{current_timestamp}] Клиент Together AI не инициализирован. Чанкинг LLM невозможен.")
        return [text_to_chunk.strip()]
    if not text_to_chunk or not text_to_chunk.strip():
        print(f"[{current_timestamp}] Попытка чанкинга пустого текста. Возвращаем как есть.")
        return [text_to_chunk.strip() if text_to_chunk else ""]

    print(f"[{current_timestamp}] Запрос на чанкинг текста: '{text_to_chunk[:100]}...' моделью {CHUNKING_MODEL_ID}")

    system_prompt_chunking = (
        "Твоя задача - предельно точно и аккуратно разбить предоставленный пользователем текст на отдельные, "
        "короткие, фактически верные и логически завершенные утверждения. Каждый факт должен быть представлен как отдельный элемент. "
        "Не добавляй нумерацию, свои комментарии, объяснения или любые вводные/заключительные фразы. "
        "Выводи только сами извлеченные утверждения, каждый на новой строке. "
        "Если текст уже является одним коротким утверждением, верни его как есть. "
        "Если в тексте есть местоимения ('ты', 'я', 'мой', 'твой'), ЗАМЕНИ их на конкретные имена ('Степан', 'Джордж Дроид'), если это ОДНОЗНАЧНО следует из всего предоставленного текста. Если замена неоднозначна, пропусти такой фрагмент или верни его с местоимением, если смысл сохраняется. "
        "ИГНОРИРУЙ и НЕ ВЫВОДИ вопросы, команды, мета-комментарии о памяти или неуверенность. Извлекай только УТВЕРДИТЕЛЬНЫЕ ФАКТЫ."
    )

    user_prompt_chunking = f"""Проанализируй и разбей следующий текст на отдельные факты или утверждения. Каждый факт/утверждение должен быть на новой строке.

Пример 1:
Входной текст: "Запомни несколько вещей: меня зовут Степан, мой день рождения 14 апреля 2005 года, а моя любимая игра это Jedi Academy. Также, стримы обычно по воскресеньям."
Ожидаемый ответ:
Меня зовут Степан.
Мой день рождения 14 апреля 2005 года.
Моя любимая игра это Jedi Academy.
Стримы обычно по воскресеньям.

Пример 2:
Входной текст: "Джордж, ты помнишь, что я люблю кофе?"
Ожидаемый ответ:
(пустой ответ или специальный маркер, т.к. это вопрос, а не факт)

Пример 3:
Входной текст: "Кажется, я не знаю ответ."
Ожидаемый ответ:
(пустой ответ или специальный маркер, т.к. это выражение неуверенности)

Теперь обработай следующий текст:
\"\"\"
{text_to_chunk}
\"\"\"

Твой ответ (только утвердительные факты, каждый на новой строке):"""

    messages_for_chunking = [
        {"role": "system", "content": system_prompt_chunking},
        {"role": "user", "content": user_prompt_chunking}
    ]

    raw_chunked_text_response = await execute_together_api_call(
        model_id=CHUNKING_MODEL_ID,
        messages=messages_for_chunking,
        max_tokens=2048,
        temperature=0.05
    )

    if raw_chunked_text_response:
        chunks = [chunk.strip() for chunk in raw_chunked_text_response.split('\n') if chunk.strip()]

        valid_chunks = []
        if chunks:
            for chunk_text in chunks:
                # Фильтр: не короче 3 слов, не вопрос, не содержит фраз неуверенности от самого чанкера (если он их добавляет)
                if len(chunk_text.split()) >= 3 and not chunk_text.endswith(
                        '?') and "не уверен" not in chunk_text.lower() and "кажется" not in chunk_text.lower():
                    valid_chunks.append(chunk_text)
                else:
                    print(
                        f"[{current_timestamp}] Отбрасываем некачественный/неутвердительный чанк: '{chunk_text[:70]}...'")

        if valid_chunks:
            if len(valid_chunks) == 1 and (
                    valid_chunks[0] == text_to_chunk.strip() or len(valid_chunks[0]) > 0.85 * len(
                    text_to_chunk.strip())):  # Немного ослабил порог
                print(
                    f"[{current_timestamp}] Модель чанкинга ({CHUNKING_MODEL_ID}) вернула 1 валидный чанк, похожий на исходный. Используем его.")
            else:
                print(
                    f"[{current_timestamp}] Текст успешно разбит на {len(valid_chunks)} валидных чанков моделью {CHUNKING_MODEL_ID}.")
            return valid_chunks
        else:
            print(
                f"[{current_timestamp}] Модель чанкинга ({CHUNKING_MODEL_ID}) не вернула валидных чанков после фильтрации. Ответ модели был: '{raw_chunked_text_response[:200]}...'. Будет использован оригинальный текст.")
            return [text_to_chunk.strip()]
    else:
        print(
            f"[{current_timestamp}] Модель чанкинга ({CHUNKING_MODEL_ID}) не вернула текстовый ответ. Будет использован оригинальный текст.")
        return [text_to_chunk.strip()]


if __name__ == "__main__":
    print("-" * 40 + "\nЗапуск программы...\n" + "-" * 40)

    choose_audio_output_device()
    list_monitors_and_select()

    if not os.path.exists(SCREENSHOT_TEMP_DIR):
        try:
            os.makedirs(SCREENSHOT_TEMP_DIR)
            print(f"Создана папка для временных скриншотов: {SCREENSHOT_TEMP_DIR}")
        except Exception as e_mkdir:
            print(f"Не удалось создать папку {SCREENSHOT_TEMP_DIR}: {e_mkdir}. Скриншоты могут не работать.",
                  file=sys.stderr)

    if not all([TWITCH_ACCESS_TOKEN, TWITCH_CHANNEL]):
        print("ОШИБКА: Заполните .env (Twitch)!")
        sys.exit(1)
    if not TOGETHER_API_KEY:
        print("ОШИБКА: TOGETHER_API_KEY не указан в .env!")

    data_dir_for_memory = "data_george_memory"
    if not os.path.exists(data_dir_for_memory):
        try:
            os.makedirs(data_dir_for_memory)
            print(f"Создана директория для данных памяти: {data_dir_for_memory}")
        except Exception as e_mkdir_mem:
            print(
                f"Не удалось создать папку для памяти {data_dir_for_memory}: {e_mkdir_mem}. Память может не работать.",
                file=sys.stderr)
            sys.exit(1)

    try:
        memory_store_instance = MemoryStore(
            index_path=os.path.join(data_dir_for_memory, "george_memory.index"),
            meta_path=os.path.join(data_dir_for_memory, "george_memory_meta.jsonl")
        )
        print(
            f"Система памяти инициализирована. Загружено воспоминаний: {memory_store_instance.get_all_memories_count()}")
    except Exception as e_mem_init:
        print(f"КРИТИЧЕСКАЯ ОШИБКА инициализации MemoryStore: {e_mem_init}")
        memory_store_instance = None



    stt_model = None
    try:
        from faster_whisper import WhisperModel

        stt_model = WhisperModel(STT_MODEL_SIZE, device=STT_DEVICE, compute_type=STT_COMPUTE_TYPE)
        print(f"Модель STT ({STT_MODEL_SIZE}, {STT_DEVICE}, {STT_COMPUTE_TYPE}) загружена.")
    except ImportError:
        print("ОШИБКА: faster-whisper не установлен. STT не будет работать.")
    except Exception as e_stt_load:
        print(f"Критическая ошибка загрузки faster-whisper: {e_stt_load}")

    piper_sample_rate = None
    try:
        if os.path.exists(VOICE_CONFIG_PATH):
            with open(VOICE_CONFIG_PATH, 'r', encoding='utf-8') as f:
                piper_config = json.load(f)
                piper_sample_rate = piper_config.get('audio', {}).get('sample_rate')
            if not piper_sample_rate:
                print(f"ОШИБКА: Не найден 'sample_rate' в {VOICE_CONFIG_PATH}")
            else:
                print(f"Piper TTS sample rate: {piper_sample_rate}")
        else:
            print(f"ОШИБКА: Не найден JSON конфиг голоса: {os.path.abspath(VOICE_CONFIG_PATH)}")

        if not all([os.path.exists(PIPER_EXE_PATH), os.path.exists(VOICE_MODEL_PATH), piper_sample_rate]):
            print("TTS Piper не будет работать из-за отсутствия файлов или sample_rate.")
            piper_sample_rate = None  # Явно сбрасываем, если что-то не так
    except Exception as e_piper_init:
        print(f"Критическая ошибка инициализации Piper TTS: {e_piper_init}")
        piper_sample_rate = None

    if not client_together: print("Предупреждение: Клиент Together AI не настроен. Бот не сможет отвечать.")
    if not stt_model: print("Предупреждение: STT не загружена.")
    if not piper_sample_rate: print("Предупреждение: TTS не инициализирован.")
    if not memory_store_instance: print("Предупреждение: Система памяти не инициализирована.")

    default_mic, mic_name = None, "N/A"
    try:
        dev_info = sd.query_devices(kind='input')
        if isinstance(dev_info, dict) and 'index' in dev_info:
            default_mic, mic_name = dev_info['index'], dev_info.get('name', 'N/A')
        elif hasattr(sd.default, 'device'):
            default_device_indices = sd.default.device
            input_device_index = default_device_indices[0] if isinstance(default_device_indices, (list, tuple)) and len(
                default_device_indices) > 0 else default_device_indices
            if input_device_index != -1:
                default_mic = input_device_index
                mic_info = sd.query_devices(default_mic)
                if isinstance(mic_info, dict): mic_name = mic_info.get('name', 'N/A')
    except Exception as e_mic:
        print(f"Ошибка определения микрофона по умолчанию: {e_mic}.")
    print(
        f"Микрофон: ID {default_mic if default_mic is not None else 'Не найден'} ({mic_name}) | SR={SOURCE_SAMPLE_RATE}, Ch={SOURCE_CHANNELS}\n" + "-" * 40)

    recording_active.set()
    recorder = threading.Thread(target=audio_recording_thread, args=(default_mic,), daemon=True, name="AudioRecorder")
    recorder.start()

    hotkeys_thread = None
    if 'keyboard' in sys.modules:
        hotkeys_thread = threading.Thread(target=hotkey_listener_thread, daemon=True, name="HotkeyListener")
        hotkeys_thread.start()
    else:
        print("Хоткеи не будут работать ('keyboard' не импортирован или не найден).")

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    main_task_instance = None
    try:
        main_task_instance = loop.create_task(main_async(), name="MainLoop")
        loop.run_until_complete(main_task_instance)
    except KeyboardInterrupt:
        print("\nПрограмма прервана пользователем (Ctrl+C). Завершение...")
        if recording_active: recording_active.clear()
    except Exception as e_loop:
        print(f"Критическая ошибка главного цикла asyncio: {e_loop}")
        import traceback

        traceback.print_exc()
        if recording_active: recording_active.clear()
    finally:
        print("Начинается процесс graceful shutdown программы...")
        if recording_active: recording_active.clear()

        threads_to_join = [t for t in [recorder, hotkeys_thread] if t and t.is_alive()]
        if threads_to_join:
            print(f"Ожидание завершения потоков: {[t.name for t in threads_to_join]}...")
            for t in threads_to_join:
                t.join(timeout=2.0)
                if t.is_alive():
                    print(f"Поток {t.name} не завершился вовремя.")
        else:
            print("Активных пользовательских потоков для завершения не найдено.")

        if main_task_instance and not main_task_instance.done():
            print("Отмена основной задачи main_async...")
            main_task_instance.cancel()

        if loop and not loop.is_closed():
            try:
                all_tasks = [task for task in asyncio.all_tasks(loop=loop) if
                             task is not asyncio.current_task(loop=loop)]  # type: ignore
                if all_tasks:
                    print(f"Отмена {len(all_tasks)} оставшихся задач asyncio...")
                    for task in all_tasks:
                        task.cancel()
                    loop.run_until_complete(asyncio.gather(*all_tasks, return_exceptions=True))
                    print("Оставшиеся задачи asyncio обработаны.")

                print("Завершение асинхронных генераторов...")
                loop.run_until_complete(loop.shutdown_asyncgens())
            except RuntimeError as e_async_shutdown:
                print(f"Ошибка при завершении задач/генераторов asyncio: {e_async_shutdown}")
            except Exception as e_async_shutdown_other:
                print(f"Непредвиденная ошибка при завершении задач/генераторов asyncio: {e_async_shutdown_other}")
            finally:
                if not loop.is_closed():
                    print("Закрытие основного цикла asyncio...")
                    loop.close()
                    print("Основной цикл asyncio закрыт.")
        else:
            print("Основной цикл asyncio уже был закрыт или не существует к моменту финальной очистки.")

        print("-" * 40 + "\nПрограмма завершена.\n" + "-" * 40)