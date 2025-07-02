import asyncio
from config.settings import TOGETHER_API_KEY

# Глобальная переменная для клиента
client_together = None


def init_together_client():
    """Инициализация клиента Together AI"""
    global client_together

    if not TOGETHER_API_KEY:
        print("ОШИБКА: TOGETHER_API_KEY не найден в .env файле!")
        return False

    try:
        from together import Together
        client_together = Together(api_key=TOGETHER_API_KEY)

        # Проверяем доступность API
        try:
            models_list = client_together.models.list()
            print(f"Клиент Together AI успешно инициализирован. Доступно моделей: {len(models_list)}")
            return True
        except Exception as e_api_check:
            print(f"Клиент Together AI инициализирован, но не удалось проверить доступ к API: {e_api_check}")
            print(f"Продолжаем работу...")
            return True

    except ImportError:
        print("ОШИБКА: Библиотека 'together' не установлена. Выполните: pip install --upgrade together")
        return False
    except Exception as e_together_init:
        print(f"Ошибка импорта или настройки Together AI: {e_together_init}")
        return False


async def execute_together_api_call(model_id: str, messages: list, max_tokens: int, temperature: float,
                                    ожидается_json: bool = False):
    """Выполнение запроса к Together AI API"""
    global client_together

    if not client_together:
        print(f"Клиент Together AI не инициализирован. Запрос к {model_id} невозможен.")
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
        print(f"Отправка запроса к {model_id} (temp={temperature}, max_tokens={max_tokens})")

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
            print(f"API {model_id} неожиданный формат ответа или нет 'choices'. Ответ: {response_details[:1000]}")
            return None

    except Exception as e:
        print(f"Ошибка API {model_id} ({type(e).__name__}): {e}")

        # Подробная диагностика ошибки
        if hasattr(e, 'response') and e.response is not None:
            response_obj = e.response
            err_text_content = "N/A"
            try:
                if hasattr(response_obj, 'text'):
                    err_text_content = response_obj.text
                elif hasattr(response_obj, 'content'):
                    err_text_content = response_obj.content.decode(errors='ignore')

                if hasattr(response_obj, 'json'):
                    try:
                        err_json_content = response_obj.json()
                        print(f"JSON ошибки от API: {err_json_content}")
                    except:
                        print(f"Текст ошибки от API: {err_text_content}")
                elif err_text_content != "N/A":
                    print(f"Текст ошибки от API: {err_text_content}")

            except Exception as e_resp:
                print(f"Дополнительная ошибка при извлечении деталей: {e_resp}")
        elif hasattr(e, 'body'):
            print(f"Тело ошибки: {e.body}")

        import traceback
        traceback.print_exc()
        return None


def is_client_ready():
    """Проверка готовности клиента"""
    return client_together is not None