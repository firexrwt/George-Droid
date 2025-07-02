import re
import json
import datetime
from ai.llm_client import execute_together_api_call
from config.settings import INTENT_ANALYSIS_MODEL_ID, CHUNKING_MODEL_ID


async def analyze_intent_for_visual_reference(text_to_analyze: str) -> dict | None:
    """Анализ намерения пользователя на предмет визуальной ссылки"""
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
            print(f"[Анализ намерения] Найден JSON напрямую: {json_str_to_parse[:300]}...")
        else:
            think_blocks = re.findall(r"<think>(.*?)</think>", raw_response_text, flags=re.DOTALL)
            if think_blocks:
                last_think_content = think_blocks[-1].strip()
                think_json_match = re.search(r'(\{[\s\S]*?\})', last_think_content)
                if think_json_match:
                    json_str_to_parse = think_json_match.group(1)

            if not json_str_to_parse:
                cleaned_text_after_think_removal = re.sub(r"<think>.*?</think>", "", raw_response_text,
                                                          flags=re.DOTALL).strip()
                if cleaned_text_after_think_removal != raw_response_text and cleaned_text_after_think_removal:
                    clean_json_match = re.search(r'(\{[\s\S]*?\})', cleaned_text_after_think_removal)
                    if clean_json_match:
                        json_str_to_parse = clean_json_match.group(1)

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

                analysis_result = json.loads(json_str_to_parse)

                if isinstance(analysis_result, dict) and "visual_reference" in analysis_result:
                    print(f"[Анализ намерения] Успешный парсинг. Результат: {analysis_result}")
                    return analysis_result
                else:
                    print(f"[Анализ намерения] Некорректный JSON или отсутствует ключ 'visual_reference'")
            except json.JSONDecodeError as e:
                print(f"[Анализ намерения] Ошибка декодирования JSON: {e}")
            except Exception as e_parse:
                print(f"[Анализ намерения] Непредвиденная ошибка при парсинге JSON: {e_parse}")
        else:
            print(f"[Анализ намерения] JSON не найден в ответе модели")

    return None


async def analyze_query_context(query_text: str, author: str) -> dict:
    """Анализирует контекст запроса - о ком идёт речь"""
    system_prompt = """Определи, о ком идёт речь в запросе. Ответь ТОЛЬКО JSON.

Правила:
- Если есть "я", "мой", "моя", "мне", "меня" - речь об авторе запроса
- Если упоминается имя - речь об этом человеке
- Если контекст неясен - предполагаем автора

Формат ответа:
{"subject": "имя_человека", "confidence": 0.1-1.0}"""

    user_prompt = f"""Автор запроса: {author}
Запрос: "{query_text}"

Ответ (ТОЛЬКО JSON):"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    try:
        response = await execute_together_api_call(
            model_id=INTENT_ANALYSIS_MODEL_ID,
            messages=messages,
            max_tokens=200,
            temperature=0.1
        )

        if response:
            json_match = re.search(r'\{.*?\}', response)
            if json_match:
                return json.loads(json_match.group())
    except:
        pass

    return {"subject": author, "confidence": 0.7}


async def extract_facts_from_interaction(raw_text: str, source: str, author: str) -> list[dict] | None:
    """Извлекает структурированные факты из сырого текста через LLM"""
    if not raw_text.strip():
        return None

    current_time = datetime.datetime.now().strftime('%H:%M:%S')

    # Контекстуальный промпт в зависимости от источника
    context_hint = ""
    if source == "STT" and author == "Stepan":
        context_hint = "Это устная речь Степана (стримера), могут быть оговорки и разговорные выражения."
    elif source == "twitch_chat":
        context_hint = f"Это сообщение из чата Twitch от пользователя {author}."

    system_prompt = """Ты - система извлечения фактов для AI-ассистента по имени Джордж Дроид.
Твоя задача - извлечь ТОЛЬКО конкретные, полезные факты из предоставленного текста.

КРИТИЧЕСКИ ВАЖНО: Отвечай ТОЛЬКО валидным JSON массивом фактов. Никакого другого текста!

Правила извлечения:
1. Извлекай только КОНКРЕТНЫЕ факты (даты, имена, предпочтения, события)
2. Игнорируй: приветствия, вопросы, команды, междометия, повторы
3. Нормализуй даты в формат YYYY-MM-DD
4. Исправляй очевидные опечатки
5. Если факт неполный или неясный - пропусти его
6. ВАЖНО: Для фактов о людях ВСЕГДА указывай имя человека в тексте факта
7. Создавай расширенные формулировки для лучшего поиска:
   - Включай альтернативные формулировки (день рождения/родился/дата рождения)
   - Добавляй контекст (любимая игра/предпочитает играть/часто играет)

Примеры правильного извлечения:
- "Меня зовут Степан" → {"fact": "Степана зовут Степан. Имя стримера - Степан.", "category": "personal_info", "confidence": 1.0, "entities": {"person": "Степан"}}
- "Мой день рождения 14 апреля" → {"fact": "День рождения Степана - 14 апреля. Степан родился 14 апреля.", "category": "personal_info", "confidence": 0.9, "entities": {"person": "Степан", "date": "YYYY-04-14"}}
- "Люблю играть в Jedi Academy" → {"fact": "Степан любит играть в Star Wars Jedi Knight Jedi Academy. Любимая игра Степана - Jedi Academy.", "category": "preference", "confidence": 0.9, "entities": {"person": "Степан", "game": "Jedi Academy"}}

Формат ответа - JSON массив:
[
  {
    "fact": "текст факта с именем человека и альтернативными формулировками",
    "category": "personal_info|preference|event|statement|other",
    "confidence": 0.1-1.0,
    "entities": {"person": "имя", "date": "YYYY-MM-DD", ...}
  }
]

Если фактов нет, верни пустой массив: []"""

    user_prompt = f"""{context_hint}
Автор высказывания: {author}

Извлеки факты из следующего текста:
"{raw_text}"

Ответ (ТОЛЬКО JSON):"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    try:
        response = await execute_together_api_call(
            model_id=CHUNKING_MODEL_ID,
            messages=messages,
            max_tokens=1024,
            temperature=0.1
        )

        if response:
            # Извлекаем JSON из ответа
            json_match = re.search(r'\[.*?\]', response, re.DOTALL)
            if json_match:
                facts = json.loads(json_match.group())
                print(f"[{current_time}] Извлечено {len(facts)} фактов из '{raw_text[:50]}...'")

                # Дополнительная валидация и обогащение
                validated_facts = []
                for fact in facts:
                    if fact.get('fact') and fact.get('category'):
                        # Убеждаемся, что имя автора есть в факте
                        if author not in fact['fact'] and 'personal' in fact.get('category', ''):
                            fact['fact'] = re.sub(r'\bя\b', author, fact['fact'], flags=re.IGNORECASE)
                            fact['fact'] = re.sub(r'\bмой\b', f"{author} имеет", fact['fact'], flags=re.IGNORECASE)
                            fact['fact'] = re.sub(r'\bмоя\b', f"у {author}", fact['fact'], flags=re.IGNORECASE)
                            fact['fact'] = re.sub(r'\bмне\b', author, fact['fact'], flags=re.IGNORECASE)

                        validated_facts.append(fact)

                return validated_facts if validated_facts else None

    except Exception as e:
        print(f"[{current_time}] Ошибка извлечения фактов: {e}")

    return None


def should_remember_interaction(text: str, source: str, author: str, bot_name: str) -> tuple[str, dict] | None:
    """Определяет, нужно ли запоминать взаимодействие и как"""
    if not text or not text.strip():
        return None

    text_lower = text.lower()
    base_importance = 0.5
    needs_fact_extraction = False

    if source == "STT" and author == "Stepan":
        # Проверяем, не является ли это просто командой или вопросом
        is_pure_question = text.endswith("?") and len(text.split()) < 10
        is_command = any(cmd in text_lower for cmd in ["скажи", "сделай", "покажи", "включи", "выключи"])

        if is_pure_question or is_command:
            # Короткие вопросы и команды не требуют извлечения фактов
            memory_params = {
                "memory_type": "user_question_stt" if is_pure_question else "user_command_stt",
                "importance": 0.3,
                "needs_fact_extraction": False
            }
        else:
            # Всё остальное идёт на извлечение фактов
            needs_fact_extraction = True

            # Повышаем важность для явных указаний запомнить
            if "запомни" in text_lower or "это важно" in text_lower:
                base_importance = 0.9
            else:
                base_importance = 0.7

            memory_params = {
                "memory_type": "stt_for_extraction",
                "importance": base_importance,
                "needs_fact_extraction": True
            }

    # Для чата - извлекаем факты только из важных сообщений
    elif source == "twitch_chat":
        trigger_parts = [p.lower() for p in bot_name.split() if len(p) > 2]
        mentioned_by_name = any(trig in text_lower for trig in trigger_parts)

        if mentioned_by_name:
            # Если сообщение содержит факты о пользователе или важную информацию
            fact_indicators = ["я", "мой", "моя", "мне", "у меня", "родился", "живу", "работаю", "учусь"]
            contains_personal_info = any(indicator in text_lower for indicator in fact_indicators)

            if contains_personal_info and len(text.split()) > 5:
                needs_fact_extraction = True
                memory_params = {
                    "memory_type": "chat_for_extraction",
                    "importance": 0.6,
                    "needs_fact_extraction": True
                }
            else:
                memory_params = {
                    "memory_type": "direct_mention_chat",
                    "importance": 0.4,
                    "needs_fact_extraction": False
                }
        elif len(text.split()) > 40:
            # Длинные сообщения могут содержать факты
            memory_params = {
                "memory_type": "long_chat_message",
                "importance": 0.3,
                "needs_fact_extraction": True
            }
        else:
            return None

    # Для ответов бота - сохраняем только уверенные утверждения
    elif source in ["bot_response", "bot_response_chat"]:
        phrases_to_filter_out = [
            "кажется, я не помню", "я не знаю", "моя память коротка",
            "отсутствует в моей базе", "я мог ошибиться", "не буду гадать",
            "мои файлы говорят", "мои логи говорят", "я не сохранил эту информацию",
            "напомни ещё раз", "хочешь освежить", "кажется, мы уже обсуждали",
            "это где-то...", "наверное?", "возможно", "вероятно"
        ]

        if any(phrase in text_lower for phrase in phrases_to_filter_out):
            print(
                f">>> FILTERED BOT RESPONSE: Ответ бота содержит фразы неуверенности, НЕ запоминаем: '{text[:60]}...'")
            return None

        # Ответы бота обычно уже структурированы, не требуют извлечения
        if len(text.split()) > 35:
            memory_params = {
                "memory_type": "bot_detailed_statement",
                "importance": 0.5,
                "needs_fact_extraction": False
            }
        elif len(text.split()) > 5:
            memory_params = {
                "memory_type": "bot_concise_statement",
                "importance": 0.45,
                "needs_fact_extraction": False
            }
        else:
            return None
    else:
        return None

    if memory_params:
        final_meta = {
            "source": source,
            "author": author,
            "memory_type": memory_params["memory_type"],
            "importance": memory_params.get("importance", base_importance),
            "needs_fact_extraction": memory_params.get("needs_fact_extraction", needs_fact_extraction),
            "original_text": text
        }

        return text, final_meta

    return None