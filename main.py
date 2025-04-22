import os
import sys
from dotenv import load_dotenv
import twitchio
import aiohttp
import json
import datetime
import asyncio
import subprocess
import io
import sounddevice as sd
import numpy as np
import wave

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

# --- Настройки Piper TTS (используем piper.exe) ---
PIPER_EXE_PATH = os.getenv('PIPER_EXE_PATH', 'piper_tts_bin/piper.exe')
VOICE_MODEL_PATH = os.getenv('PIPER_VOICE_MODEL_PATH', 'voices/ru_RU-ruslan-medium.onnx')
VOICE_CONFIG_PATH = os.getenv('PIPER_VOICE_CONFIG_PATH', 'voices/ru_RU-ruslan-medium.onnx.json')

# --- Настройка файла для вывода текста в OBS ---
OBS_OUTPUT_FILE = "obs_ai_response.txt"

# --- Чтение Sample Rate из конфига голоса ---
piper_sample_rate = None
try:
    if not os.path.exists(PIPER_EXE_PATH): print(f"ОШИБКА: Не найден piper.exe: {os.path.abspath(PIPER_EXE_PATH)}")
    if not os.path.exists(VOICE_MODEL_PATH): print(f"ОШИБКА: Не найден .onnx: {os.path.abspath(VOICE_MODEL_PATH)}")
    if os.path.exists(VOICE_CONFIG_PATH):
        with open(VOICE_CONFIG_PATH, 'r', encoding='utf-8') as f:
            voice_config = json.load(f)
            piper_sample_rate = voice_config.get('audio', {}).get('sample_rate')
            if piper_sample_rate:
                print(f"Голос Piper: Модель '{VOICE_MODEL_PATH}', SR: {piper_sample_rate}")
            else:
                print(f"ОШИБКА: Не найден 'sample_rate' в {VOICE_CONFIG_PATH}")
    else:
        print(f"ОШИБКА: Не найден .json: {os.path.abspath(VOICE_CONFIG_PATH)}")
    if not all([os.path.exists(PIPER_EXE_PATH), os.path.exists(VOICE_MODEL_PATH), piper_sample_rate]):
        print("ОШИБКА: TTS не будет работать.")
        piper_sample_rate = None
except Exception as e:
    print(f"Критическая ошибка при инициализации Piper TTS: {e}")
    piper_sample_rate = None

# --- Системный промпт для ИИ ---
SYSTEM_PROMPT = """## Твоя Личность: Джимми Нейрон

Ты - Джимми Нейрон, остроумный, саркастичный и высокоинтеллектуальный ИИ-ассистент и соведущий на Twitch-стриме на канале firexrwt. Его владелец Степан. Ты создан, чтобы развлекать аудиторию и взаимодействовать со Степаном и чатом.

**Твой Стиль Общения и Характер:**

1.  **Основа - Юмор и Остроумие:** Ты прирожденный юморист с острым умом. Твои шутки должны быть умными, часто основанными на игре слов, иронии, сарказме и наблюдениях за происходящим на стриме или в чате. Ты быстро соображаешь и можешь подколоть Степана или зрителей, но обычно по-дружески.

2.  **Язык Ответа:** Ты должен генерировать ответ **строго и исключительно на русском языке**. Не допускается использование слов или фраз на других языках (включая английский), если только это не общепринятый термин или название, не имеющее русского аналога (например, название игры).

3.  **Форматирование Ответа:** Не используй звездочки (`*`), скобки или любые другие символы для описания действий, эмоций, звуков или интонаций (например, НЕ НАДО писать `*улыбается*`, `*вздыхает*`, `(смеется)`). Весь твой ответ должен быть обычным текстом. Передавай эмоции и сарказм только через подбор слов и стиль речи, а не через описания действий.

4.  **"Черный" Юмор (ИСПОЛЬЗОВАТЬ КРАЙНЕ ОСТОРОЖНО!):** Ты ценишь интеллектуальный юмор, который может быть немного циничным или касаться сложных тем (абсурдность бытия, технологические парадоксы и т.д.). **НО!** Твой юмор **НИКОГДА** не должен:
    * Нарушать правила Twitch (см. ниже).
    * Быть направлен против каких-либо групп людей (расовых, религиозных, гендерных, ЛГБТ+ и т.д.).
    * Оскорблять или унижать кого-либо.
    * Касаться реальных трагедий, чувствительных социальных проблем таким образом, чтобы это выглядело как издевательство или глумление.
    * Быть чрезмерно жестоким или шокирующим ради шока.
    * **Думай об этом как об "едком сарказме" или "интеллектуальной провокации", а не как о безвкусной "чернухе".** Если есть хоть малейшее сомнение, что шутка может быть неправильно понята или нарушить правила - **НЕ ИСПОЛЬЗУЙ ЕЕ**.

5.  **Шутки про Мам (СУПЕР РЕДКО, АБСТРАКТНО, НЕОЖИДАННО):** Это твоя странная, почти глючная особенность. **КРАЙНЕ РЕДКО** (буквально раз в несколько часов или даже реже), совершенно внезапно и, возможно, не к месту, ты можешь выдать **абсурдную, нелепую, сюрреалистичную** шутку в стиле "Твоя мама...". **КЛЮЧЕВЫЕ ПРАВИЛА для этих шуток:**
    * **Максимально глупые и оторванные от реальности:** Пример: "Твоя мама настолько старая, что ее первый питомец был динозавр... и он до сих пор просит есть". Или "Твоя мама так хорошо готовит, что даже каннибалы просят у нее рецепт... овощного рагу".
    * **Очень редки и неожиданны:** Не надо пытаться вставить их часто. Это должно быть похоже на случайный сбой в твоей программе юмора.
    * **Если не уверен на 100%, что это будет смешно и безобидно - НЕ ИСПОЛЬЗУЙ.**

6.  **Взаимодействие:** Активно общайся со Степаном (называй его по имени или Файрекс) и реагируй на сообщения из чата Twitch. Можешь задавать вопросы, комментировать игру или действия Степана, отвечать на вопросы зрителей.

**!!! ВАЖНЕЙШЕЕ ПРАВИЛО №1: СОБЛЮДЕНИЕ ПРАВИЛ TWITCH !!!**

* **ПРИОРИТЕТ №0:** Соблюдение Условий обслуживания (ToS) и Правил сообщества Twitch (**Community Guidelines**) — это **АБСОЛЮТНО ГЛАВНАЯ ЗАДАЧА**. Важнее любой шутки, любого ответа.
* **СТРОГО ЗАПРЕЩЕНО:**
    * Разжигание ненависти, дискриминация, оскорбления на основе расы, этнической принадлежности, религии, пола, гендерной идентичности, сексуальной ориентации, возраста, инвалидности и т.д.
    * Домогательства, преследование, угрозы в адрес кого-либо.
    * Чрезмерная пошлость, нецензурная брань, направленная на оскорбление конкретных лиц.
    * Обсуждение или прославление незаконной деятельности, самоповреждения.
    * Распространение дезинформации, особенно вредоносной.
* **ИЗБЕГАЙ:** Спорных политических дискуссий, религиозных споров, излишне откровенных тем. Будь умным и тактичным.
* **ПОЛИТИКА ОТКАЗА:** Если запрос от пользователя или ситуация на стриме кажутся тебе рискованными с точки зрения правил Twitch, ты **ДОЛЖЕН** вежливо отказаться от ответа или сменить тему. Пример ответа: "Ох, эта тема кажется немного скользкой для Twitch, давай лучше поговорим о [другая тема]?" или "Мои алгоритмы советуют мне не углубляться в это".

**Твоя Цель:** Быть уникальным, запоминающимся, смешным и интеллектуальным ИИ-персонажем, который делает стрим Степана круче, но при этом всегда остается безопасным, уважительным и на 100% соответствующим правилам Twitch. Ты — умный помощник и развлекатель, а не генератор проблем.
"""

# --- Простая история диалога ---
conversation_history = []
MAX_HISTORY_LENGTH = 10

# --- Проверки перед запуском ---
if not all([TWITCH_ACCESS_TOKEN, TWITCH_BOT_NICK, TWITCH_CHANNEL]): sys.exit(
    "КРИТИЧЕСКАЯ ОШИБКА: Переменные Twitch не найдены в .env")
if "## Твоя Личность: Джимми Нейрон" not in SYSTEM_PROMPT and "## Твоя Личность: Джордж Дроид" not in SYSTEM_PROMPT: sys.exit(
    "КРИТИЧЕСКАЯ ОШИБКА: Имя бота не установлено в SYSTEM_PROMPT.")

print(f"Загружены настройки: Аккаунт '{TWITCH_BOT_NICK}' -> Канал '{TWITCH_CHANNEL}'. Модель: '{OLLAMA_MODEL_NAME}'.")
if piper_sample_rate:
    print(f"TTS Готов (Piper.exe, SR: {piper_sample_rate}, Голос: {VOICE_MODEL_PATH})")
else:
    print("TTS Недоступен (проверьте пути к Piper и голосу)")


# --- Асинхронная функция для взаимодействия с Ollama API ---
async def get_ollama_response(user_message):
    global conversation_history
    conversation_history.append({"role": "user", "content": user_message})
    if len(conversation_history) > MAX_HISTORY_LENGTH:
        conversation_history = conversation_history[-MAX_HISTORY_LENGTH:]
    messages_payload = [{"role": "system", "content": SYSTEM_PROMPT}] + conversation_history
    payload = {"model": OLLAMA_MODEL_NAME, "messages": messages_payload, "stream": False}
    print(f"Отправка запроса в Ollama (Модель: {OLLAMA_MODEL_NAME})...")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(OLLAMA_API_URL, json=payload, timeout=60) as response:
                if response.status == 200:
                    response_data = await response.json()
                    llm_content = response_data.get('message', {}).get('content')
                    if llm_content:
                        conversation_history.append({"role": "assistant", "content": llm_content})
                        print("Ответ от Ollama получен.")
                        return llm_content.strip()
                    else:
                        print(f"Ошибка: Не найден 'content' в ответе Ollama: {response_data}")
                        if conversation_history and conversation_history[-1][
                            "role"] == "user": conversation_history.pop()
                        return None
                else:
                    error_text = await response.text()
                    print(f"Ошибка при запросе к Ollama: Статус {response.status}, Ответ: {error_text}")
                    if conversation_history and conversation_history[-1]["role"] == "user": conversation_history.pop()
                    return None
    except Exception as e:
        print(f"Ошибка при обращении к Ollama: {e}")
        if conversation_history and conversation_history[-1]["role"] == "user": conversation_history.pop()
        return None


# --- Вспомогательная функция для воспроизведения аудио через sounddevice ---
def play_raw_audio_sync(audio_bytes, samplerate, dtype='int16'):
    if not audio_bytes or not samplerate:
        print("Нет аудио данных или sample rate для воспроизведения.")
        return
    try:
        print(f"Воспроизведение {len(audio_bytes)} байт, SR={samplerate}, dtype={dtype}")
        audio_data = np.frombuffer(audio_bytes, dtype=dtype)
        sd.play(audio_data, samplerate=samplerate, blocking=True)
    except ValueError as e:
        print(f"Ошибка NumPy при чтении аудио буфера: {e}. Проверьте dtype ('{dtype}') и структуру данных.")
    except Exception as e:
        print(f"Ошибка при воспроизведении аудио через sounddevice: {e}")


# --- Функция для синтеза и воспроизведения через piper.exe (Вариант 2 активен) ---
async def speak_text(text_to_speak):
    if not piper_sample_rate or not os.path.exists(PIPER_EXE_PATH) or not os.path.exists(VOICE_MODEL_PATH):
        print("TTS недоступен (проверьте пути и конфиг).")
        return
    print(f"Запуск TTS (piper.exe) для текста: \"{text_to_speak[:50]}...\"")
    """ ... """
    command = [PIPER_EXE_PATH, '--model', VOICE_MODEL_PATH, '--output-raw']
    try:
        process = await asyncio.create_subprocess_exec(
            *command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        print("Передача текста в Piper и ожидание аудио...")
        audio_bytes, stderr_bytes = await process.communicate(input=text_to_speak.encode('utf-8'))
        if process.returncode == 0 and audio_bytes:
            print(f"Аудио синтезировано ({len(audio_bytes)} байт), запуск воспроизведения через sounddevice...")
            await asyncio.to_thread(play_raw_audio_sync, audio_bytes, piper_sample_rate)
            print("Воспроизведение (sounddevice) завершено.")
        elif process.returncode != 0:
            print(f"Ошибка выполнения piper.exe: Exit code {process.returncode}")
            if stderr_bytes: print(f"Stderr: {stderr_bytes.decode('utf-8', errors='ignore')}")
        elif not audio_bytes:
            print("Ошибка: piper.exe завершился успешно, но не вернул аудио данные.")
            if stderr_bytes: print(f"Stderr: {stderr_bytes.decode('utf-8', errors='ignore')}")
    except FileNotFoundError:
        print(f"КРИТИЧЕСКАЯ ОШИБКА: Не найден исполняемый файл piper.exe по пути: {PIPER_EXE_PATH}")
    except Exception as e:
        print(f"Ошибка при вызове piper.exe / sounddevice: {e}")


# --- Класс нашего Twitch бота ---
class SimpleBot(twitchio.Client):
    async def event_ready(self):
        print(f'Подключен к Twitch IRC как | {self.nick}')
        if self.connected_channels:
            print(f'Присоединился к каналу | {self.connected_channels[0].name}')
            print('--------- Бот готов читать чат ---------')
        else:
            print(f'Не удалось присоединиться к каналу {TWITCH_CHANNEL}.')

    async def event_message(self, message):
        if message.echo:
            return

        current_time = datetime.datetime.now().strftime('%H:%M:%S')
        print(f"[{current_time}] {message.author.name}: {message.content}")
        try:
            with open(OBS_OUTPUT_FILE, 'w', encoding='utf-8') as f:
                f.write("")
        except Exception as e:
            print(f"[{current_time}] ОШИБКА очистки файла для OBS ({OBS_OUTPUT_FILE}) перед запросом: {e}")
        # --- Конец очистки файла ---

        # Вызов Ollama для получения ответа ИИ
        llm_response_text = await get_ollama_response(message.content)

        # Обрабатываем результат от Ollama
        if llm_response_text:
            print(f"[{current_time}] Ответ Ollama ({OLLAMA_MODEL_NAME}): {llm_response_text}")

            # --- Запись НОВОГО ответа в файл для OBS ---
            try:
                with open(OBS_OUTPUT_FILE, 'w', encoding='utf-8') as f:
                    f.write(llm_response_text)
                # print(f"[{current_time}] Ответ записан в {OBS_OUTPUT_FILE}") # Для отладки
            except Exception as e:
                print(f"[{current_time}] ОШИБКА записи в файл для OBS ({OBS_OUTPUT_FILE}): {e}")
            # --- Конец записи в файл ---

            await speak_text(llm_response_text)

        else:
            print(f"[{current_time}] Не удалось получить ответ от Ollama для сообщения от {message.author.name}.")


# --- Точка входа в программу ---
if __name__ == "__main__":
    import numpy as np
    import sounddevice as sd

    # ... (проверки перед запуском) ...
    if not piper_sample_rate:
        print("Предупреждение: TTS не инициализирован. Бот будет работать без голоса.")

    # Создаем и запускаем бота
    client = SimpleBot(token=TWITCH_ACCESS_TOKEN, initial_channels=[TWITCH_CHANNEL])
    client.run()