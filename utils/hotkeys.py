import time
import threading

from config.settings import STT_HOTKEY, CHAT_HOTKEY

# Глобальные переменные (будут установлены извне)
recording_active = None


def set_globals(rec_active):
    """Установка глобальных переменных"""
    global recording_active
    recording_active = rec_active


def toggle_stt():
    """Переключение STT"""
    from core.processor import toggle_stt as core_toggle_stt
    core_toggle_stt()


def toggle_chat_interaction():
    """Переключение взаимодействия с чатом"""
    from twitch.bot import toggle_chat_interaction as chat_toggle
    chat_toggle()


def hotkey_listener_thread():
    """Поток для обработки горячих клавиш"""
    reg_stt, reg_chat = False, False

    try:
        import keyboard
        print(f"\nХоткей STT: '{STT_HOTKEY}', Хоткей Чата: '{CHAT_HOTKEY}'")

        try:
            keyboard.add_hotkey(STT_HOTKEY, toggle_stt)
            reg_stt = True
            print(f"✓ Зарегистрирован хоткей STT: {STT_HOTKEY}")
        except Exception as e_stt:
            print(f"✗ Ошибка регистрации хоткея STT: {e_stt}")

        try:
            keyboard.add_hotkey(CHAT_HOTKEY, toggle_chat_interaction)
            reg_chat = True
            print(f"✓ Зарегистрирован хоткей чата: {CHAT_HOTKEY}")
        except Exception as e_chat:
            print(f"✗ Ошибка регистрации хоткея чата: {e_chat}")

        # Главный цикл
        while recording_active.is_set():
            time.sleep(0.5)

    except ImportError:
        print("\nОШИБКА: Библиотека 'keyboard' не найдена. Хоткеи не будут работать.")
        print("Установите её командой: pip install keyboard")
        return
    except Exception as e:
        if not (isinstance(e, ValueError) and "is not mapped" in str(e)):
            print(f"\nОшибка hotkey_listener: {e}")
        else:
            print(f"\nПредупреждение: Не удалось зарегистрировать хоткей (возможно, уже используется): {e}")
    finally:
        # Очистка зарегистрированных хоткеев
        try:
            import keyboard
            if reg_stt:
                keyboard.remove_hotkey(STT_HOTKEY)
                print(f"✓ Хоткей STT ({STT_HOTKEY}) удалён")
            if reg_chat:
                keyboard.remove_hotkey(CHAT_HOTKEY)
                print(f"✓ Хоткей чата ({CHAT_HOTKEY}) удалён")
        except Exception as e_cleanup:
            print(f"Ошибка при очистке хоткеев: {e_cleanup}")

        print("Поток хоткеев завершен.")


def start_hotkey_listener():
    """Запуск потока обработки горячих клавиш"""
    if not recording_active:
        print("ОШИБКА: recording_active не установлен для hotkeys")
        return None

    hotkeys_thread = threading.Thread(
        target=hotkey_listener_thread,
        daemon=True,
        name="HotkeyListener"
    )
    hotkeys_thread.start()
    return hotkeys_thread


def test_hotkeys():
    """Тестирование доступности горячих клавиш"""
    try:
        import keyboard
        print("✓ Библиотека keyboard доступна")

        # Тестируем парсинг хоткеев
        try:
            keyboard.parse_hotkey(STT_HOTKEY)
            print(f"✓ Хоткей STT '{STT_HOTKEY}' корректен")
        except Exception as e:
            print(f"✗ Хоткей STT '{STT_HOTKEY}' некорректен: {e}")

        try:
            keyboard.parse_hotkey(CHAT_HOTKEY)
            print(f"✓ Хоткей чата '{CHAT_HOTKEY}' корректен")
        except Exception as e:
            print(f"✗ Хоткей чата '{CHAT_HOTKEY}' некорректен: {e}")

        return True
    except ImportError:
        print("✗ Библиотека keyboard не установлена")
        return False
    except Exception as e:
        print(f"✗ Ошибка тестирования хоткеев: {e}")
        return False