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
from PIL import Image  # Pillow для обработки изображений
import mss  # Для скриншотов
import tempfile  # Для временных файлов
import base64

load_dotenv()

# --- Настройки Together AI ---
TOGETHER_API_KEY = os.getenv('TOGETHER_API_KEY')
TOGETHER_MODEL_ID = os.getenv('TOGETHER_MODEL_ID', "meta-llama/Llama-4-Scout-17B-16E-Instruct")

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


# --- Новая обобщенная функция для вызова API Together AI ---
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
                # Пытаемся получить текст ошибки, если возможно
                if hasattr(response_obj, 'text'):
                    err_text_content = response_obj.text
                elif hasattr(response_obj, 'content'):  # Иногда байты в content
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

    # Супер-строгий системный промпт
    system_prompt_intent = """Твоя задача - проанализировать фразу пользователя и определить, ссылается ли она на визуальный контекст.
Ты ДОЛЖЕН ответить ТОЛЬКО валидным JSON объектом и НИЧЕМ БОЛЕЕ.
НЕ ИСПОЛЬЗУЙ <think> теги. НЕ ДОБАВЛЯЙ никаких объяснений или текста до или после JSON.
Твой ЕДИНСТВЕННЫЙ вывод должен быть JSON объектом. Это критически важно.
"""

    # Пользовательский промпт: правила, фраза, примеры ТОЛЬКО JSON, и прямой призыв к JSON
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
        max_tokens=700,  # Оставляем увеличенное значение
        temperature=0.05,  # Еще ниже температура для максимальной предсказуемости
        ожидается_json=True
    )

    if raw_response_text:
        json_str_to_parse = None

        # Используем нежадный поиск JSON (добавляем ?)
        # r'(\{[\s\S]*?\})' - ищет первый корректный блок от { до }

        # 1. Попытка найти JSON напрямую во всем ответе
        direct_json_match = re.search(r'(\{[\s\S]*?\})', raw_response_text)
        if direct_json_match:
            json_str_to_parse = direct_json_match.group(1)  # group(1) т.к. скобки в regex создают группу
            print(
                f"[Анализ намерения] Найден JSON напрямую (стратегия 1, нежадный поиск): {json_str_to_parse[:300]}...")
        else:
            # 2. Если не найден, ищем содержимое последнего <think> блока
            think_blocks = re.findall(r"<think>(.*?)</think>", raw_response_text, flags=re.DOTALL)
            if think_blocks:
                last_think_content = think_blocks[-1].strip()
                print(f"[Анализ намерения] Содержимое последнего <think> блока: {last_think_content[:300]}...")
                think_json_match = re.search(r'(\{[\s\S]*?\})', last_think_content)
                if think_json_match:
                    json_str_to_parse = think_json_match.group(1)
                    print(
                        f"[Анализ намерения] Найден JSON в последнем <think> блоке (стратегия 2, нежадный поиск): {json_str_to_parse[:300]}...")

            # 3. Если все еще не найден, удаляем все <think> блоки и ищем в оставшемся
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
                # Очистка от Markdown ```json ... ``` или ``` ... ```
                # Важно делать это аккуратно, чтобы не повредить JSON, если ``` есть внутри строк

                # Удаляем ```json в начале и ``` в конце, если они есть
                if json_str_to_parse.startswith("```json"):
                    json_str_to_parse = json_str_to_parse[len("```json"):].strip()
                    if json_str_to_parse.endswith("```"):
                        json_str_to_parse = json_str_to_parse[:-len("```")].strip()
                # Если не было ```json, но есть просто ```
                elif json_str_to_parse.startswith("```"):
                    json_str_to_parse = json_str_to_parse[len("```"):].strip()
                    if json_str_to_parse.endswith("```"):
                        json_str_to_parse = json_str_to_parse[:-len("```")].strip()

                print(
                    f"[Анализ намерения] Строка для парсинга JSON после очистки Markdown: '{json_str_to_parse[:300]}...'")
                analysis_result = json.loads(json_str_to_parse)  # Здесь может быть ошибка, если JSON неполный!

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

    history_messages_for_prompt = []
    if conversation_history:
        for msg in conversation_history[-(MAX_HISTORY_LENGTH * 2):]:
            if msg["role"] == "user":
                history_messages_for_prompt.append({"role": "user", "content": msg["content"]})
            elif msg["role"] == "assistant":
                history_messages_for_prompt.append({"role": "assistant", "content": msg["content"]})

    current_user_content_list = [{"type": "text", "text": user_text}]

    if screenshot_file_path:
        try:
            with open(screenshot_file_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            image_data_uri = f"data:image/png;base64,{base64_image}"  # PNG, так как мы сохраняем в PNG
            current_user_content_list.append({
                "type": "image_url",
                "image_url": {"url": image_data_uri}
            })
            print(
                f"Изображение {screenshot_file_path} подготовлено для отправки (первые 100 символов data URI): {image_data_uri[:100]}...")
        except Exception as e:
            print(f"Ошибка кодирования изображения в base64: {e}", file=sys.stderr)
            pass

    messages_for_scout = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]
    messages_for_scout.extend(history_messages_for_prompt)
    messages_for_scout.append({"role": "user", "content": current_user_content_list})

    llm_response_text = await execute_together_api_call(
        model_id=TOGETHER_MODEL_ID,
        messages=messages_for_scout,
        max_tokens=512,
        temperature=0.8
    )

    if llm_response_text:
        conversation_history.append({"role": "user", "content": user_text})  # Сохраняем только текст запроса
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


async def get_togetherai_response(user_message_with_prefix: str):  # Для чата Twitch (текст-онли)
    global conversation_history, SYSTEM_PROMPT, TOGETHER_MODEL_ID, BOT_NAME_FOR_CHECK, MAX_HISTORY_LENGTH, client_together

    if not client_together:
        print(f"Together AI (чат) клиент не инициализирован. Запрос невозможен.", file=sys.stderr)
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

    messages_for_llm.append({"role": "user", "content": user_message_with_prefix})

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
        print(f"Together AI (чат) пустой ответ или ошибка.", file=sys.stderr)
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

    async def event_message(self, message):
        if message.echo: return

        global chat_interaction_enabled, is_processing_response, last_activity_time
        global BOT_NAME_FOR_CHECK, OBS_OUTPUT_FILE, stt_enabled, audio_queue, client_together

        if not chat_interaction_enabled: return
        if message.channel.name != self.target_channel_name: return
        if not client_together:
            print("Ответ невозможен (чат): Клиент Together AI не настроен.")
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
    global OBS_OUTPUT_FILE, client_together, TOGETHER_MODEL_ID

    current_time_str = lambda: datetime.datetime.now().strftime('%H:%M:%S')

    # --- Распознавание речи ---
    full_audio = np.concatenate(audio_buffer_list, axis=0)
    mono_audio = full_audio
    if full_audio.ndim > 1 and full_audio.shape[1] > 1:
        mono_audio = full_audio.mean(axis=1)

    resampled = resample_audio(mono_audio, SOURCE_SAMPLE_RATE, TARGET_SAMPLE_RATE)
    recognized_text = None
    if resampled is not None and resampled.size > 0:
        recognized_text = await asyncio.to_thread(transcribe_audio_faster_whisper, resampled)

    if not recognized_text:
        return
    if is_processing_response:
        print(
            f"[{current_time_str()}] Бот уже обрабатывает другой запрос, новый STT '{recognized_text[:50]}...' проигнорирован.")
        return

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
                reason = "N/A"
                if intent_analysis:
                    reason = intent_analysis.get('reason', 'N/A')
                print(
                    f"[{current_time_str()}] Анализ намерения: Ссылка на визуальный контекст не обнаружена. Причина: {reason}")

            if should_take_screenshot:
                screenshot_file_to_send = await asyncio.to_thread(capture_and_prepare_screenshot)
                if not screenshot_file_to_send:
                    print(f"[{current_time_str()}] Не удалось сделать/подготовить скриншот, ответ будет без него.")
        else:
            print(f"[{current_time_str()}] Источник '{source_id}' не STT, анализ намерения и скриншот не выполняются.")
        try:
            with open(OBS_OUTPUT_FILE, 'w', encoding='utf-8') as f:
                f.write("")
        except Exception as e:
            print(f"[{current_time_str()}] Ошибка очистки OBS файла: {e}")

        llm_response = await get_main_llm_response(recognized_text, screenshot_file_to_send)

        if llm_response:
            print(
                f"[{current_time_str()}] Ответ Together AI ({TOGETHER_MODEL_ID}, источник {source_id}): {llm_response}")
            try:
                with open(OBS_OUTPUT_FILE, 'w', encoding='utf-8') as f:
                    f.write(llm_response)
            except Exception as e:
                print(f"[{current_time_str()}] Ошибка записи в OBS файл: {e}")

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
        if screenshot_file_to_send:
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
            piper_sample_rate = None
    except Exception as e:
        print(f"Критическая ошибка инициализации Piper TTS: {e}")
        piper_sample_rate = None

    if not client_together: print("Предупреждение: Клиент Together AI не настроен. Бот не сможет отвечать.")
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
        print("\nПрограмма прервана пользователем (Ctrl+C). Завершение...")
        recording_active.clear()
    except Exception as e_loop:
        print(f"Критическая ошибка главного цикла: {e_loop}");
        import traceback

        traceback.print_exc()
        recording_active.clear()
    finally:
        print("Начинается процесс завершения программы...")
        recording_active.clear()  # Останавливаем все фоновые активности, связанные с записью

        # Завершаем потоки
        threads_to_join = [t for t in [recorder, hotkeys_thread] if t and t.is_alive()]
        if threads_to_join:
            print(f"Ожидание завершения потоков: {[t.name for t in threads_to_join]}...")
            for t in threads_to_join:
                t.join(timeout=2.0)  # Даем потокам немного времени на завершение
                if t.is_alive():
                    print(f"Поток {t.name} не завершился вовремя.")
        else:
            print("Активных пользовательских потоков для завершения не найдено.")

        # Отменяем основную асинхронную задачу, если она еще не завершена
        if main_task_instance and not main_task_instance.done():
            print("Отмена основной задачи main_async...")
            main_task_instance.cancel()

        # Очистка ресурсов asyncio
        if loop and not loop.is_closed():  # Проверяем, что цикл существует и еще не закрыт
            if loop.is_running():  # Только если цикл еще работает, пытаемся управлять задачами
                print("Цикл asyncio активен. Отмена и ожидание завершения оставшихся задач asyncio...")

                pending_async_tasks = []
                try:
                    # Безопасное получение текущей задачи, если цикл действительно работает
                    current_task_obj = asyncio.current_task(loop=loop)
                    pending_async_tasks = [t for t in asyncio.all_tasks(loop=loop) if t is not current_task_obj]
                except RuntimeError:
                    # Это может случиться, если цикл формально is_running(), но уже в процессе остановки
                    print(
                        "Не удалось получить current_task (возможно, цикл в процессе остановки). Попытка получить все задачи.")
                    try:
                        pending_async_tasks = [t for t in asyncio.all_tasks(loop=loop)]
                    except RuntimeError as e_get_all:
                        print(f"Не удалось получить все задачи asyncio: {e_get_all}")

                if pending_async_tasks:
                    print(f"Обнаружено {len(pending_async_tasks)} ожидающих задач asyncio для отмены/завершения.")
                    for task_to_cancel in pending_async_tasks:
                        if not task_to_cancel.done():  # Отменяем только незавершенные
                            task_to_cancel.cancel()
                    try:
                        # Ожидаем завершения отмененных задач
                        loop.run_until_complete(asyncio.gather(*pending_async_tasks, return_exceptions=True))
                        print("Оставшиеся задачи asyncio обработаны (завершены/отменены).")
                    except RuntimeError as e_loop_stopped_during_gather:
                        print(f"Ошибка при ожидании gather (возможно, цикл остановлен): {e_loop_stopped_during_gather}")
                    except Exception as e_gather_final:
                        print(f"Общая ошибка при ожидании завершения оставшихся задач asyncio: {e_gather_final}")
                else:
                    print("Нет ожидающих задач asyncio для обработки.")

                try:
                    print("Завершение асинхронных генераторов (цикл был запущен)...")
                    loop.run_until_complete(loop.shutdown_asyncgens())
                    print("Асинхронные генераторы завершены.")
                except RuntimeError as e_loop_stopped_during_gens:  # Если цикл остановился во время shutdown_asyncgens
                    print(
                        f"Ошибка при завершении генераторов (возможно, цикл остановлен): {e_loop_stopped_during_gens}")
                except Exception as e_shutdown_gens:
                    print(f"Общая ошибка при завершении асинхронных генераторов: {e_shutdown_gens}")

            else:  # Цикл существует, но не запущен (loop.is_running() is False)
                print("Цикл asyncio существует, но не запущен. Пропуск операций с задачами.")

            # Финальное закрытие цикла, если он еще не закрыт
            if not loop.is_closed():
                print("Закрытие основного цикла asyncio...")
                loop.close()
                print("Основной цикл asyncio закрыт.")
        else:
            print("Основной цикл asyncio уже был закрыт или не существует к моменту финальной очистки.")

        print("-" * 40 + "\nПрограмма завершена.\n" + "-" * 40)