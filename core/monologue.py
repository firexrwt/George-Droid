import asyncio
import datetime
import time

from ai.llm_client import is_client_ready
from audio.audio_processor import speak_text
from config.settings import INACTIVITY_THRESHOLD_SECONDS, BOT_NAME_FOR_CHECK, OBS_OUTPUT_FILE

# Глобальные переменные (будут установлены извне)
recording_active = None
chat_interaction_enabled = True
last_activity_time = time.time()
audio_queue = None
is_processing_event = None
stt_enabled = True


def set_globals(rec_active, chat_enabled, audio_q, processing_event):
    """Установка глобальных переменных"""
    global recording_active, chat_interaction_enabled, audio_queue, is_processing_event
    recording_active = rec_active
    chat_interaction_enabled = chat_enabled
    audio_queue = audio_q
    is_processing_event = processing_event


def get_last_activity_time():
    """Получение времени последней активности"""
    return last_activity_time


def set_last_activity_time(new_time):
    """Установка времени последней активности"""
    global last_activity_time
    last_activity_time = new_time


def get_stt_enabled():
    """Получение состояния STT"""
    return stt_enabled


def set_stt_enabled(enabled):
    """Установка состояния STT"""
    global stt_enabled
    stt_enabled = enabled


async def get_monologue_response(prompt: str):
    """Получение ответа для монолога"""
    from twitch.bot import get_togetherai_response
    from core.processor import get_conversation_history

    conversation_history = get_conversation_history()
    return await get_togetherai_response(prompt, conversation_history, None)


async def monologue_loop():
    """Цикл монологов во время тишины"""
    global last_activity_time, stt_enabled

    while recording_active.is_set():
        await asyncio.sleep(15)

        if not is_client_ready():
            continue

        # Проверяем флаг обработки через Event
        if is_processing_event.is_set() or not chat_interaction_enabled:
            continue

        if time.time() - last_activity_time > INACTIVITY_THRESHOLD_SECONDS:
            current_time = datetime.datetime.now().strftime('%H:%M:%S')

            if is_processing_event.is_set() or not chat_interaction_enabled:
                continue

            stt_was_initially_enabled = stt_enabled
            try:
                # Устанавливаем флаг обработки
                is_processing_event.set()
                if stt_enabled:
                    stt_enabled = False

                if audio_queue:
                    with audio_queue.mutex:
                        audio_queue.queue.clear()

                prompt = f"Сгенерируй короткую (1-2 предл.) реплику от {BOT_NAME_FOR_CHECK} для заполнения тишины."
                llm_response = await get_monologue_response(prompt)

                if llm_response:
                    print(f"[{current_time}] Монолог: {llm_response}")
                    try:
                        with open(OBS_OUTPUT_FILE, 'w', encoding='utf-8') as f:
                            f.write(llm_response)
                    except Exception as e:
                        print(f"[{current_time}] Ошибка записи монолога в OBS: {e}")

                    await speak_text(llm_response)

                    if audio_queue:
                        with audio_queue.mutex:
                            audio_queue.queue.clear()
                    last_activity_time = time.time()
                else:
                    print(f"[{current_time}] Монолог не сгенерирован.")

            except Exception as e:
                print(f"[{current_time}] КРИТ. ОШИБКА monologue_loop: {e}")
            finally:
                # Гарантированно сбрасываем флаг обработки
                is_processing_event.clear()
                stt_enabled = stt_was_initially_enabled