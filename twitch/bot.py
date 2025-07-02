import datetime
import queue
import time
import twitchio

from ai.llm_client import is_client_ready
from ai.memory_handler import should_remember_interaction, extract_facts_from_interaction
from audio.audio_processor import speak_text
from config.settings import BOT_NAME_FOR_CHECK, OBS_OUTPUT_FILE

# Глобальные переменные (будут инициализированы извне)
chat_interaction_enabled = True
last_activity_time = time.time()
audio_queue = None
memory_store_instance = None
is_processing_event = None
stt_enabled = True


def set_globals(audio_q, memory_store, processing_event):
    """Установка глобальных переменных"""
    global audio_queue, memory_store_instance, is_processing_event
    audio_queue = audio_q
    memory_store_instance = memory_store
    is_processing_event = processing_event


def toggle_chat_interaction():
    """Переключение взаимодействия с чатом"""
    global chat_interaction_enabled
    chat_interaction_enabled = not chat_interaction_enabled
    status = "ВКЛ" if chat_interaction_enabled else "ВЫКЛ"
    print(f"\n=== Реакция на чат {status} ===")


async def get_togetherai_response(user_message_with_prefix: str, conversation_history: list, memory_store_instance):
    """Получение ответа от Together AI для чата"""
    from ai.llm_client import execute_together_api_call
    from config.settings import TOGETHER_MODEL_ID, SYSTEM_PROMPT, MAX_HISTORY_LENGTH

    current_time_str_chat_llm = lambda: datetime.datetime.now().strftime('%H:%M:%S')

    if not is_client_ready():
        print(f"[{current_time_str_chat_llm()}] Together AI (чат) клиент не инициализирован. Запрос невозможен.")
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

        print(f"[{current_time_str_chat_llm()}] RAG Query (chat_llm): '{query_for_memory_chat[:200]}...'")

        retrieved_memories_list_chat = memory_store_instance.retrieve_memories(query_text=query_for_memory_chat,
                                                                               top_k=5)

        if retrieved_memories_list_chat:
            print(
                f"[{current_time_str_chat_llm()}] DEBUG RAG (chat_llm): Извлечено {len(retrieved_memories_list_chat)} воспоминаний:")
            for i, mem_item_chat in enumerate(retrieved_memories_list_chat):
                print(
                    f"  DEBUG RAG MEM {i + 1} (CosSim: {mem_item_chat.get('cosine_similarity', 'N/A'):.4f}): \"{mem_item_chat.get('text', '')[:100]}...\"")

            top_n_for_llm_context_chat = 2
            actual_memories_for_llm_chat = []
            if retrieved_memories_list_chat:
                actual_memories_for_llm_chat = sorted(retrieved_memories_list_chat,
                                                      key=lambda x: x.get('cosine_similarity', 0.0), reverse=True)[
                                               :top_n_for_llm_context_chat]
                actual_memories_for_llm_chat = [mem for mem in actual_memories_for_llm_chat if
                                                mem.get('cosine_similarity', 0.0) > 0.60]

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
        print(f"[{current_time_str_chat_llm()}] Together AI (чат) пустой ответ или ошибка.")
        return None


class SimpleBot(twitchio.Client):
    def __init__(self, token, initial_channels, conversation_history):
        super().__init__(token=token, initial_channels=initial_channels)
        self.target_channel_name = initial_channels[0] if initial_channels else None
        self.conversation_history = conversation_history

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
        if message.echo:
            return

        global chat_interaction_enabled, last_activity_time, stt_enabled, audio_queue

        current_time_str_chat = lambda: datetime.datetime.now().strftime('%H:%M:%S')

        if not chat_interaction_enabled:
            return
        if not message.channel or message.channel.name != self.target_channel_name:
            return

        author_name = message.author.name if message.author else "unknown_twitch_user"

        if not is_client_ready():
            print(
                f"[{current_time_str_chat()}] Ответ невозможен (чат от {author_name}): Клиент Together AI не настроен.")
            return

        content_lower = message.content.lower()
        trigger_parts = [p.lower() for p in BOT_NAME_FOR_CHECK.split() if len(p) > 2]
        mentioned = any(trig in content_lower for trig in trigger_parts)
        highlighted = message.tags.get('msg-id') == 'highlighted-message' if message.tags else False

        if not mentioned and not highlighted:
            return

        # Проверяем флаг обработки через Event
        if is_processing_event.is_set():
            print(f"[{current_time_str_chat()}] Бот занят (чат). Игнор сообщения от {author_name}.")
            return

        stt_was_on_before_chat_processing = False
        try:
            # Устанавливаем флаг обработки
            is_processing_event.set()
            last_activity_time = time.time()
            print(f"[{current_time_str_chat()}] {author_name}: {message.content}")

            stt_was_on_before_chat_processing = stt_enabled
            if stt_enabled:
                stt_enabled = False
                print(f"[{current_time_str_chat()}] STT временно ВЫКЛЮЧЕН на время обработки чат-сообщения.")
            if audio_queue:
                with audio_queue.mutex:
                    audio_queue.queue.clear()

            try:
                open(OBS_OUTPUT_FILE, 'w').close()
            except Exception as e_obs_clear_chat:
                print(f"[{current_time_str_chat()}] Ошибка очистки OBS (чат): {e_obs_clear_chat}")

            # Обработка памяти для чата
            if memory_store_instance and message.content:
                chat_memory_action_details = should_remember_interaction(
                    text=message.content,
                    source="twitch_chat",
                    author=author_name,
                    bot_name=BOT_NAME_FOR_CHECK
                )
                if chat_memory_action_details:
                    original_chat_text_to_remember, base_chat_meta = chat_memory_action_details

                    if base_chat_meta.get("needs_fact_extraction", False):
                        # Извлекаем факты через LLM
                        extracted_facts = await extract_facts_from_interaction(
                            message.content,
                            source="twitch_chat",
                            author=author_name
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
                                    "source": "twitch_chat_extracted",
                                    "author": author_name,
                                    "memory_type": f"fact_{fact_data.get('category', 'other')}",
                                    "importance": min(importance, 0.99),
                                    "custom_meta": {
                                        "original_text": message.content[:200],
                                        "entities": fact_data.get('entities', {}),
                                        "extraction_confidence": fact_data.get('confidence', 0.5)
                                    }
                                }

                                print(
                                    f"[{current_time_str_chat()}] Сохранение извлечённого факта из чата: '{fact_text}'")

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
                                    print(f"[{current_time_str_chat()}] Ошибка добавления факта из чата: {e_mem_add}")
                        else:
                            # Если извлечение не дало результатов, сохраняем как есть
                            try:
                                memory_store_instance.add_memory(
                                    text=original_chat_text_to_remember,
                                    source=base_chat_meta["source"],
                                    author=base_chat_meta["author"],
                                    memory_type=base_chat_meta["memory_type"],
                                    importance=base_chat_meta["importance"]
                                )
                            except Exception as e_mem_add_fallback:
                                print(
                                    f"[{current_time_str_chat()}] Ошибка добавления исходного текста чата: {e_mem_add_fallback}")
                    else:
                        # Не требуется извлечение фактов, сохраняем как есть
                        try:
                            memory_store_instance.add_memory(
                                text=original_chat_text_to_remember,
                                source=base_chat_meta["source"],
                                author=base_chat_meta["author"],
                                memory_type=base_chat_meta["memory_type"],
                                importance=base_chat_meta["importance"]
                            )
                        except Exception as e_mem_add_direct:
                            print(f"[{current_time_str_chat()}] Ошибка прямого добавления чата: {e_mem_add_direct}")

            llm_input_for_chat = f"(Чат от {author_name}): {message.content}"
            llm_response = await get_togetherai_response(llm_input_for_chat, self.conversation_history,
                                                         memory_store_instance)

            if llm_response:
                print(f"[{current_time_str_chat()}] Ответ Together AI (чат): {llm_response}")
                try:
                    open(OBS_OUTPUT_FILE, 'w', encoding='utf-8').write(llm_response)
                except Exception as e_obs_write_chat:
                    print(f"[{current_time_str_chat()}] Ошибка записи в OBS (чат): {e_obs_write_chat}")

                # Сохранение ответа бота в память
                save_this_bot_chat_response = True
                phrases_to_filter_out_bot_chat = [
                    "кажется, я не помню", "я не знаю", "моя память коротка",
                    "отсутствует в моей базе", "я мог ошибиться", "не буду гадать"
                ]
                if any(phrase in llm_response.lower() for phrase in phrases_to_filter_out_bot_chat):
                    save_this_bot_chat_response = False

                if save_this_bot_chat_response and memory_store_instance:
                    bot_chat_memory_action_details = should_remember_interaction(
                        text=llm_response,
                        source="bot_response_chat",
                        author=BOT_NAME_FOR_CHECK,
                        bot_name=BOT_NAME_FOR_CHECK
                    )
                    if bot_chat_memory_action_details:
                        bot_text, bot_meta = bot_chat_memory_action_details
                        try:
                            memory_store_instance.add_memory(
                                text=bot_text,
                                source=bot_meta["source"],
                                author=bot_meta["author"],
                                memory_type=bot_meta["memory_type"],
                                importance=bot_meta["importance"],
                                custom_meta={"response_to": message.content[:100]}
                            )
                            print(f"[{current_time_str_chat()}] Ответ бота (чат) сохранён в память")
                        except Exception as e_bot_mem:
                            print(f"[{current_time_str_chat()}] Ошибка сохранения ответа бота (чат): {e_bot_mem}")

                await speak_text(llm_response)
                if audio_queue:
                    with audio_queue.mutex:
                        audio_queue.queue.clear()

                # Обновляем время активности через функцию из процессора
                from core.processor import set_last_activity_time
                set_last_activity_time(time.time())
            else:
                print(f"[{current_time_str_chat()}] Нет ответа Together AI для {author_name}.")

            if stt_was_on_before_chat_processing:
                stt_enabled = True
                print(f"[{current_time_str_chat()}] STT снова ВКЛЮЧЕН (после обработки чата).")
        except Exception as e_event_msg:
            print(f"[{current_time_str_chat()}] КРИТ. ОШИБКА event_message: {e_event_msg}", exc_info=True)
            if stt_was_on_before_chat_processing:
                stt_enabled = True
        finally:
            # Гарантированно сбрасываем флаг обработки
            is_processing_event.clear()