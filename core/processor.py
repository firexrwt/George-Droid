import asyncio
import base64
import datetime
import queue
import time
import threading
import numpy as np
import sounddevice as sd

from audio.audio_processor import (
    resample_audio, transcribe_audio_faster_whisper, speak_text, convert_to_mono
)
from audio.vad import VoiceActivityDetector
from ai.llm_client import execute_together_api_call, is_client_ready
from ai.memory_handler import (
    analyze_intent_for_visual_reference, analyze_query_context,
    should_remember_interaction, extract_facts_from_interaction
)
from vision.screenshot_handler import capture_and_prepare_screenshot, delete_screenshot_file
from config.settings import (
    SOURCE_SAMPLE_RATE, SOURCE_CHANNELS, TARGET_DTYPE, BLOCKSIZE,
    OBS_OUTPUT_FILE, TOGETHER_MODEL_ID, SYSTEM_PROMPT, MAX_HISTORY_LENGTH,
    BOT_NAME_FOR_CHECK
)

# Глобальные переменные
audio_queue = queue.Queue()
recording_active = threading.Event()
last_activity_time = time.time()
stt_enabled = True
is_processing_event = threading.Event()
conversation_history = []
vad_detector = VoiceActivityDetector()
memory_store_instance = None


def set_memory_store(memory_store):
    """Установка экземпляра хранилища памяти"""
    global memory_store_instance
    memory_store_instance = memory_store


def toggle_stt():
    """Переключение STT"""
    global stt_enabled, audio_queue
    stt_enabled = not stt_enabled
    status = "ВКЛ" if stt_enabled else "ВЫКЛ"
    print(f"\n--- STT {status} ---")
    if not stt_enabled:
        with audio_queue.mutex:
            audio_queue.queue.clear()
        vad_detector.force_reset()


def audio_recording_thread(device_index=None):
    """Поток записи аудио"""
    global audio_queue, recording_active, stt_enabled

    def audio_callback(indata, frames, time, status):
        if status:
            print(f"Audio Status: {status}")

        # Тройная проверка условий + проверка размера очереди
        if (recording_active.is_set() and
                stt_enabled and
                not is_processing_event.is_set() and
                audio_queue.qsize() < 1000):
            try:
                audio_queue.put_nowait(indata.copy())
            except queue.Full:
                pass
            except Exception as e:
                print(f"[AUDIO] Unexpected error in callback: {e}")

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
        print(f"Критическая ошибка аудиозаписи: {e}")
    finally:
        if stream and not stream.closed:
            try:
                stream.stop()
                stream.close()
            except Exception as e_close:
                print(f"Ошибка закрытия аудиопотока: {e_close}")
        print("Поток записи аудио остановлен.")


async def stt_processing_loop():
    """Основной цикл обработки STT"""
    global audio_queue, recording_active, stt_enabled, vad_detector

    while recording_active.is_set():
        # Проверяем Event вместо переменной
        if not stt_enabled or is_processing_event.is_set():
            if is_processing_event.is_set() and vad_detector.is_speaking:
                vad_detector.force_reset()
            await asyncio.sleep(0.1)
            continue

        try:
            block = audio_queue.get_nowait()
        except queue.Empty:
            await asyncio.sleep(0.01)
            continue

        # Обработка блока через VAD
        ready_audio_buffer = vad_detector.process_audio_block(block)

        if ready_audio_buffer:
            # Готов сегмент для распознавания
            asyncio.create_task(process_recognized_speech(ready_audio_buffer, "STT"))


async def get_main_llm_response(user_text: str, screenshot_file_path: str | None = None):
    """Получение основного ответа от LLM"""
    global conversation_history, memory_store_instance

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
        # Анализируем контекст - о ком идёт речь
        query_context = await analyze_query_context(user_text, "Stepan")
        query_subject = query_context.get("subject", "Stepan")

        # Формируем поисковый запрос
        query_for_memory = user_text
        temp_history_for_query = [
            msg['content'] for msg_idx, msg in enumerate(conversation_history)
            if msg.get('content') and msg_idx >= len(conversation_history) - 2
        ]
        if temp_history_for_query:
            query_for_memory = " ".join(temp_history_for_query) + " " + user_text

        print(f"[{current_time_str_main_llm()}] RAG Query (main_llm): '{query_for_memory[:200]}...'")
        print(f"[{current_time_str_main_llm()}] Контекст запроса: речь о '{query_subject}'")

        # Получаем больше воспоминаний для фильтрации
        retrieved_memories_list = memory_store_instance.retrieve_memories(
            query_text=query_for_memory,
            top_k=20  # Берём больше для последующей фильтрации
        )

        if retrieved_memories_list:
            # Группируем по релевантности
            highly_relevant = []  # Прямое совпадение автора + высокий score
            author_relevant = []  # Совпадение автора + средний score
            keyword_relevant = []  # Содержит ключевые слова
            other_relevant = []  # Остальные с хорошим score

            # Ключевые слова из запроса
            query_keywords = set(word.lower() for word in query_for_memory.split()
                                 if
                                 len(word) > 3 and word.lower() not in ['когда', 'какой', 'какая', 'что', 'как', 'твоя',
                                                                        'твой', 'твоё'])

            # Добавляем специфичные ключевые слова
            if "день рождения" in user_text.lower() or "родился" in user_text.lower():
                query_keywords.update(["день", "рождения", "родился", "апреля", "апрель"])
            if "любимая игра" in user_text.lower() or "игра" in user_text.lower():
                query_keywords.update(["игра", "любимая", "играть", "jedi", "academy"])

            for mem in retrieved_memories_list:
                mem_text_lower = mem.get('text', '').lower()
                mem_author = mem.get('author', '')
                mem_type = mem.get('type', '')
                score = mem.get('cosine_similarity', 0.0)

                # Проверяем упоминание субъекта в тексте
                subject_mentioned = query_subject.lower() in mem_text_lower

                # Проверяем наличие ключевых слов
                keywords_found = sum(1 for kw in query_keywords if kw in mem_text_lower)
                keyword_ratio = keywords_found / len(query_keywords) if query_keywords else 0

                # Особые случаи для личной информации
                is_personal_fact = 'personal_info' in mem_type or 'preference' in mem_type

                # Приоритезация
                if is_personal_fact and (subject_mentioned or mem_author == query_subject):
                    highly_relevant.append(mem)
                elif subject_mentioned and score > 0.2:
                    highly_relevant.append(mem)
                elif mem_author == query_subject and score > 0.15:
                    author_relevant.append(mem)
                elif keyword_ratio > 0.3 and score > 0.2:
                    keyword_relevant.append(mem)
                elif score > 0.5:
                    other_relevant.append(mem)

            # Собираем финальный список
            actual_memories_for_llm = []

            # Добавляем по приоритету
            actual_memories_for_llm.extend(highly_relevant[:4])
            actual_memories_for_llm.extend(author_relevant[:3])
            actual_memories_for_llm.extend(keyword_relevant[:2])
            actual_memories_for_llm.extend(other_relevant[:1])

            # Убираем дубликаты
            seen_ids = set()
            unique_memories = []
            for mem in actual_memories_for_llm:
                if mem.get('id') not in seen_ids:
                    seen_ids.add(mem.get('id'))
                    unique_memories.append(mem)
            actual_memories_for_llm = unique_memories[:7]  # Максимум 7 фактов

            print(
                f"[{current_time_str_main_llm()}] После умной фильтрации: {len(actual_memories_for_llm)} воспоминаний")

            if actual_memories_for_llm:
                # Группируем факты по категориям
                facts_by_category = {}
                for mem in actual_memories_for_llm:
                    category = mem.get('type', 'other').replace('fact_', '')
                    if category not in facts_by_category:
                        facts_by_category[category] = []
                    facts_by_category[category].append(mem)

                memory_prompt_header = f"\n\n[Джордж, вот информация из твоей памяти о {query_subject}:]\n"
                formatted_sections = []

                # Приоритетные категории
                priority_order = ['personal_info', 'preference', 'event', 'statement', 'other']

                for category in priority_order:
                    if category in facts_by_category:
                        section_facts = facts_by_category[category]
                        if category == 'personal_info':
                            section_title = "ЛИЧНАЯ ИНФОРМАЦИЯ"
                        elif category == 'preference':
                            section_title = "ПРЕДПОЧТЕНИЯ"
                        elif category == 'event':
                            section_title = "СОБЫТИЯ"
                        else:
                            section_title = category.upper()

                        section_text = f"\n{section_title}:\n"
                        for fact in section_facts[:3]:
                            entities = fact.get('custom_meta', {}).get('entities', {})
                            entities_str = f" [дата: {entities.get('date')}]" if entities.get('date') else ""
                            section_text += f"• {fact.get('text', '')}{entities_str}\n"

                        formatted_sections.append(section_text)

                retrieved_memories_context_str = memory_prompt_header + "".join(
                    formatted_sections) + "\n[Используй эту информацию для точного ответа]\n"

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
            print(f"[{current_time_str_main_llm()}] Ошибка кодирования изображения в base64: {e_img}")

    messages_for_scout = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages_for_scout.extend(history_messages_for_prompt)
    messages_for_scout.append({"role": "user", "content": current_user_content_list})

    llm_response_text = await execute_together_api_call(
        model_id=TOGETHER_MODEL_ID,
        messages=messages_for_scout,
        max_tokens=768,
        temperature=0.75
    )

    if llm_response_text:
        conversation_history.append({"role": "user", "content": user_text})
        conversation_history.append({"role": "assistant", "content": llm_response_text})
        if len(conversation_history) > MAX_HISTORY_LENGTH * 2:
            conversation_history = conversation_history[-(MAX_HISTORY_LENGTH * 2):]
    return llm_response_text


async def process_recognized_speech(audio_buffer_list, source_id="STT"):
    """Обработка распознанной речи"""
    global stt_enabled, audio_queue, last_activity_time, memory_store_instance

    current_time_str = lambda: datetime.datetime.now().strftime('%H:%M:%S')

    full_audio = np.concatenate(audio_buffer_list, axis=0)
    mono_audio = convert_to_mono(full_audio)

    resampled = resample_audio(mono_audio, SOURCE_SAMPLE_RATE, 16000)  # TARGET_SAMPLE_RATE константа
    recognized_text = None
    if resampled is not None and resampled.size > 0:
        recognized_text = await asyncio.to_thread(transcribe_audio_faster_whisper, resampled)

    if not recognized_text:
        print(f"[{current_time_str()}] STT не вернул текст.")
        return

    # Проверяем, не обрабатывается ли уже другой запрос
    if is_processing_event.is_set():
        print(
            f"[{current_time_str()}] Бот уже обрабатывает другой запрос, новый STT '{recognized_text[:50]}...' проигнорирован.")
        return

    # Обработка памяти
    if memory_store_instance and recognized_text:
        memory_action_details = should_remember_interaction(
            text=recognized_text,
            source="STT",
            author="Stepan",
            bot_name=BOT_NAME_FOR_CHECK
        )

        if memory_action_details:
            original_text_to_remember, base_meta_for_memory = memory_action_details

            if base_meta_for_memory.get("needs_fact_extraction", False):
                # Извлекаем факты через LLM
                extracted_facts = await extract_facts_from_interaction(
                    recognized_text,
                    source="STT",
                    author="Stepan"
                )

                if extracted_facts:
                    # Сохраняем каждый извлечённый факт
                    for fact_data in extracted_facts:
                        if fact_data.get('confidence', 0) < 0.6:
                            continue

                        fact_text = fact_data.get('fact', '')
                        if not fact_text:
                            continue

                        importance = fact_data.get('confidence', 0.5)
                        if fact_data.get('category') == 'personal_info':
                            importance *= 1.2

                        memory_meta = {
                            "source": "STT_extracted",
                            "author": "Stepan",
                            "memory_type": f"fact_{fact_data.get('category', 'other')}",
                            "importance": min(importance, 0.99),
                            "custom_meta": {
                                "original_text": recognized_text[:200],
                                "entities": fact_data.get('entities', {}),
                                "extraction_confidence": fact_data.get('confidence', 0.5)
                            }
                        }

                        print(f"[{current_time_str()}] Сохранение извлечённого факта: '{fact_text}'")

                        try:
                            memory_store_instance.add_memory(
                                text=fact_text,
                                source=memory_meta["source"],
                                author=memory_meta["author"],
                                memory_type=memory_meta["memory_type"],
                                importance=memory_meta["importance"],
                                custom_meta=memory_meta["custom_meta"],
                                check_for_semantic_duplicates=True,
                                semantic_similarity_threshold=0.9
                            )
                        except Exception as e_mem_add:
                            print(f"[{current_time_str()}] Ошибка добавления факта: {e_mem_add}")
                else:
                    # Если извлечение не дало результатов, сохраняем как есть
                    try:
                        memory_store_instance.add_memory(
                            text=original_text_to_remember,
                            source=base_meta_for_memory["source"],
                            author=base_meta_for_memory["author"],
                            memory_type=base_meta_for_memory["memory_type"],
                            importance=base_meta_for_memory["importance"],
                            custom_meta={"original_text": original_text_to_remember}
                        )
                    except Exception as e_mem_add_fallback:
                        print(f"[{current_time_str()}] Ошибка добавления исходного текста: {e_mem_add_fallback}")
            else:
                # Не требуется извлечение фактов, сохраняем как есть
                try:
                    memory_store_instance.add_memory(
                        text=original_text_to_remember,
                        source=base_meta_for_memory["source"],
                        author=base_meta_for_memory["author"],
                        memory_type=base_meta_for_memory["memory_type"],
                        importance=base_meta_for_memory["importance"]
                    )
                except Exception as e_mem_add_direct:
                    print(f"[{current_time_str()}] Ошибка прямого добавления: {e_mem_add_direct}")

    # Дальше идёт основная обработка
    is_processing_event.set()
    stt_was_initially_enabled = stt_enabled

    if stt_enabled:
        stt_enabled = False
        print(
            f"[{current_time_str()}] STT временно ВЫКЛЮЧЕН на время обработки LLM для фразы: '{recognized_text[:50]}...'")

    with audio_queue.mutex:
        audio_queue.queue.clear()
    vad_detector.force_reset()

    await asyncio.sleep(0.05)

    last_activity_time = time.time()
    print(f"[{current_time_str()}] STT Распознано ({source_id}): {recognized_text}")

    if not is_client_ready():
        print(f"[{current_time_str()}] Клиент Together AI не настроен, обработка STT невозможна.")
        is_processing_event.clear()
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
                if intent_analysis:
                    reason_text = intent_analysis.get('reason', 'N/A')
                print(
                    f"[{current_time_str()}] Анализ намерения: Ссылка на визуальный контекст не обнаружена. Причина: {reason_text}")

            if should_take_screenshot:
                screenshot_file_to_send = await asyncio.to_thread(capture_and_prepare_screenshot)
                if not screenshot_file_to_send:
                    print(f"[{current_time_str()}] Не удалось сделать/подготовить скриншот, ответ будет без него.")

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

            # Сохранение ответа бота в память
            save_this_bot_response = True
            phrases_to_filter_out_bot_response = [
                "кажется, я не помню", "я не знаю", "моя память коротка",
                "отсутствует в моей базе", "я мог ошибиться", "не буду гадать"
            ]
            if any(phrase in llm_response.lower() for phrase in phrases_to_filter_out_bot_response):
                save_this_bot_response = False

            if save_this_bot_response and memory_store_instance:
                bot_memory_action_details = should_remember_interaction(
                    text=llm_response,
                    source="bot_response",
                    author=BOT_NAME_FOR_CHECK,
                    bot_name=BOT_NAME_FOR_CHECK
                )
                if bot_memory_action_details:
                    bot_text, bot_meta = bot_memory_action_details
                    try:
                        memory_store_instance.add_memory(
                            text=bot_text,
                            source=bot_meta["source"],
                            author=bot_meta["author"],
                            memory_type=bot_meta["memory_type"],
                            importance=bot_meta["importance"],
                            custom_meta={"response_to": recognized_text[:100]}
                        )
                        print(f"[{current_time_str()}] Ответ бота сохранён в память")
                    except Exception as e_bot_mem:
                        print(f"[{current_time_str()}] Ошибка сохранения ответа бота: {e_bot_mem}")

            await speak_text(llm_response)
            with audio_queue.mutex:
                audio_queue.queue.clear()
            last_activity_time = time.time()  # Обновляем время после TTS
        else:
            print(f"[{current_time_str()}] Нет ответа от основной LLM ({TOGETHER_MODEL_ID}) для источника {source_id}.")

    except Exception as e_process:
        print(
            f"[{current_time_str()}] КРИТИЧЕСКАЯ ОШИБКА в process_recognized_speech (источник {source_id}): {e_process}")
        import traceback
        traceback.print_exc()
    finally:
        if screenshot_file_to_send:
            await asyncio.to_thread(delete_screenshot_file, screenshot_file_to_send)

        is_processing_event.clear()
        await asyncio.sleep(0.05)

        if stt_was_initially_enabled:
            stt_enabled = True
            print(f"[{current_time_str()}] STT снова ВКЛЮЧЕН (обработка завершена).")
        else:
            print(f"[{current_time_str()}] STT остается ВЫКЛЮЧЕННЫМ (был выключен до обработки).")


# Функции для экспорта
def get_audio_queue():
    return audio_queue


def get_recording_active():
    return recording_active


def get_processing_event():
    return is_processing_event


def get_conversation_history():
    return conversation_history


def get_last_activity_time():
    return last_activity_time


def set_last_activity_time(new_time):
    global last_activity_time
    last_activity_time = new_time