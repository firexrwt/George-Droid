import asyncio
import os
import sys
import threading
import ctypes


# Инициализация DLL (CUDNN) в самом начале
def init_cudnn():
    """Инициализация CUDNN библиотек"""
    try:
        from config.settings import CUDNN_PATH

        if os.path.exists(CUDNN_PATH):
            os.add_dll_directory(CUDNN_PATH)
        else:
            print(f"Предупреждение: Путь CUDNN не найден: {CUDNN_PATH}")

        libs_to_try = [
            "cudnn_ops64_9.dll", "cudnn_cnn64_9.dll", "cudnn_engines_precompiled64_9.dll",
            "cudnn_heuristic64_9.dll", "cudnn_engines_runtime_compiled64_9.dll",
            "cudnn_adv64_9.dll", "cudnn_graph64_9.dll", "cudnn64_9.dll",
            "cudnn64_8.dll", "cudnn_ops64_8.dll", "cudnn_cnn64_8.dll"
        ]

        loaded_libs = 0
        for lib in libs_to_try:
            try:
                ctypes.WinDLL(lib)
                loaded_libs += 1
            except (FileNotFoundError, OSError):
                pass

        if loaded_libs == 0:
            print("Предупреждение: Не удалось загрузить ни одну DLL CUDNN.")
        else:
            print(f"✓ Загружено CUDNN библиотек: {loaded_libs}")

    except ImportError:
        print("Предупреждение: Библиотека ctypes не найдена. Пропуск загрузки CUDNN DLL.")
    except Exception as e:
        print(f"Ошибка настройки DLL: {e}")


def init_components():
    """Инициализация всех компонентов системы"""
    print("=" * 50)
    print("🤖 George-Droid - AI Streaming Companion")
    print("=" * 50)

    # Проверка настроек
    from config.settings import (
        TWITCH_ACCESS_TOKEN, TWITCH_CHANNEL, TOGETHER_API_KEY,
        SCREENSHOT_TEMP_DIR
    )

    if not all([TWITCH_ACCESS_TOKEN, TWITCH_CHANNEL]):
        print("❌ ОШИБКА: Заполните .env файл (Twitch настройки)!")
        sys.exit(1)
    if not TOGETHER_API_KEY:
        print("❌ ОШИБКА: TOGETHER_API_KEY не указан в .env!")
        sys.exit(1)

    # Создание необходимых директорий
    for directory in ["data_george_memory", SCREENSHOT_TEMP_DIR]:
        if not os.path.exists(directory):
            try:
                os.makedirs(directory)
                print(f"✓ Создана директория: {directory}")
            except Exception as e:
                print(f"❌ Не удалось создать {directory}: {e}")
                sys.exit(1)

    # Инициализация AI клиента
    from ai.llm_client import init_together_client
    ai_ready = init_together_client()
    if not ai_ready:
        print("⚠️  Предупреждение: AI клиент не готов")

    # Инициализация аудио компонентов
    from audio.audio_processor import init_stt_model, init_piper_tts, set_output_device
    from audio.device_manager import choose_audio_output_device, get_default_microphone

    stt_ready = init_stt_model()
    tts_ready = init_piper_tts()

    if not stt_ready:
        print("⚠️  Предупреждение: STT не загружена")
    if not tts_ready:
        print("⚠️  Предупреждение: TTS не инициализирован")

    # Выбор аудио устройств
    print("\n" + "=" * 50)
    print("🔊 НАСТРОЙКА АУДИО УСТРОЙСТВ")
    print("=" * 50)

    chosen_output_device = choose_audio_output_device()
    set_output_device(chosen_output_device)

    default_mic, mic_name = get_default_microphone()
    print(f"🎙️  Микрофон: ID {default_mic} ({mic_name})")

    # Выбор мониторов для скриншотов
    from vision.screenshot_handler import list_monitors_and_select, cleanup_temp_screenshots

    print("\n" + "=" * 50)
    print("🖥️  НАСТРОЙКА МОНИТОРОВ")
    print("=" * 50)

    list_monitors_and_select()
    cleanup_temp_screenshots()

    # Инициализация памяти
    from backend.memory_store import MemoryStore

    try:
        memory_store = MemoryStore(
            index_path=os.path.join("data_george_memory", "george_memory.index"),
            meta_path=os.path.join("data_george_memory", "george_memory_meta.jsonl")
        )
        print(f"🧠 Система памяти инициализирована. Воспоминаний: {memory_store.get_all_memories_count()}")
    except Exception as e:
        print(f"❌ КРИТИЧЕСКАЯ ОШИБКА инициализации памяти: {e}")
        memory_store = None

    return {
        'ai_ready': ai_ready,
        'stt_ready': stt_ready,
        'tts_ready': tts_ready,
        'memory_store': memory_store,
        'default_mic': default_mic
    }


async def main_async():
    """Основная асинхронная функция"""
    from config.settings import TWITCH_ACCESS_TOKEN, TWITCH_CHANNEL
    from core.processor import (
        audio_recording_thread, stt_processing_loop, set_memory_store,
        get_audio_queue, get_recording_active, get_processing_event,
        get_conversation_history
    )
    from core.monologue import monologue_loop, set_globals as monologue_set_globals
    from twitch.bot import SimpleBot, set_globals as twitch_set_globals
    from utils.diagnostics import run_all_diagnostics, set_globals as diag_set_globals
    from utils.hotkeys import start_hotkey_listener, set_globals as hotkey_set_globals, test_hotkeys

    # Получаем состояние компонентов
    components = init_components()

    if not components['ai_ready']:
        print("❌ AI клиент не готов. Бот не сможет отвечать.")

    # Устанавливаем память в процессор
    if components['memory_store']:
        set_memory_store(components['memory_store'])

    # Получаем общие объекты
    audio_queue = get_audio_queue()
    recording_active = get_recording_active()
    is_processing_event = get_processing_event()
    conversation_history = get_conversation_history()

    # Настраиваем глобальные переменные для модулей
    twitch_set_globals(audio_queue, components['memory_store'], is_processing_event)
    monologue_set_globals(recording_active, True, audio_queue, is_processing_event)
    diag_set_globals(recording_active, audio_queue, is_processing_event)
    hotkey_set_globals(recording_active)

    # Запуск основных компонентов
    print("\n" + "=" * 50)
    print("🚀 ЗАПУСК СИСТЕМЫ")
    print("=" * 50)

    recording_active.set()

    # Запуск потока записи аудио
    recorder = threading.Thread(
        target=audio_recording_thread,
        args=(components['default_mic'],),
        daemon=True,
        name="AudioRecorder"
    )
    recorder.start()
    print("✓ Поток записи аудио запущен")

    # Запуск потока горячих клавиш
    hotkeys_thread = None
    if test_hotkeys():
        hotkeys_thread = start_hotkey_listener()
        if hotkeys_thread:
            print("✓ Поток горячих клавиш запущен")
    else:
        print("⚠️  Хоткеи не будут работать")

    # Создание Twitch бота
    client_twitch = SimpleBot(
        token=TWITCH_ACCESS_TOKEN,
        initial_channels=[TWITCH_CHANNEL],
        conversation_history=conversation_history
    )

    # Запуск асинхронных задач
    print("✓ Запуск асинхронных задач...")

    tasks = [
        asyncio.create_task(client_twitch.start(), name="TwitchIRC"),
        asyncio.create_task(stt_processing_loop(), name="STTLoop"),
        asyncio.create_task(monologue_loop(), name="MonologueLoop"),
        asyncio.create_task(run_all_diagnostics(), name="Diagnostics")
    ]

    print("\n" + "🟢 Система запущена успешно!")
    print("📋 Управление:")
    print("   Ctrl+; - Переключить STT")
    print("   Ctrl+' - Переключить чат")
    print("   Ctrl+C - Выход")
    print("=" * 50)

    # Основной цикл
    while recording_active.is_set():
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

        for task in done:
            try:
                exc = task.exception()
                if exc:
                    print(f"\n❌ ОШИБКА в задаче {task.get_name()}: {exc}")
                    if task.get_name() in ["TwitchIRC", "STTLoop"]:  # Критические задачи
                        print("🚨 Критическая задача упала, завершение программы...")
                        recording_active.clear()
                        break
            except asyncio.CancelledError:
                pass
            except Exception as e:
                print(f"Ошибка проверки задачи {task.get_name()}: {e}")

        # Обновляем список активных задач
        tasks = list(pending)

        if not recording_active.is_set() or not tasks:
            break

        await asyncio.sleep(1)

    # Завершение работы
    print("\n" + "=" * 50)
    print("🛑 ЗАВЕРШЕНИЕ РАБОТЫ")
    print("=" * 50)

    # Отмена всех задач
    current_tasks = asyncio.all_tasks()
    tasks_to_cancel = [t for t in current_tasks if not t.done() and t is not asyncio.current_task()]

    if tasks_to_cancel:
        print(f"⏹️  Отмена {len(tasks_to_cancel)} активных задач...")
        for task in tasks_to_cancel:
            task.cancel()
        await asyncio.gather(*tasks_to_cancel, return_exceptions=True)

    # Закрытие Twitch клиента
    if client_twitch and client_twitch.is_connected():
        try:
            await client_twitch.close()
            print("✓ Twitch клиент закрыт")
        except Exception as e:
            print(f"⚠️  Ошибка закрытия Twitch клиента: {e}")

    # Ожидание завершения потоков
    threads_to_join = [t for t in [recorder, hotkeys_thread] if t and t.is_alive()]
    if threads_to_join:
        print(f"⏳ Ожидание завершения {len(threads_to_join)} потоков...")
        for t in threads_to_join:
            t.join(timeout=2.0)
            if t.is_alive():
                print(f"⚠️  Поток {t.name} не завершился вовремя")

    print("✅ Завершение работы завершено успешно")


def main():
    """Главная функция входа"""
    init_cudnn()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        loop.run_until_complete(main_async())
    except KeyboardInterrupt:
        print("\n⚠️  Программа прервана пользователем (Ctrl+C)")
    except Exception as e:
        print(f"\n❌ КРИТИЧЕСКАЯ ОШИБКА: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Очистка asyncio
        try:
            print("🧹 Очистка асинхронных ресурсов...")

            # Завершение генераторов
            loop.run_until_complete(loop.shutdown_asyncgens())

            # Закрытие цикла
            if not loop.is_closed():
                loop.close()

        except Exception as e:
            print(f"⚠️  Ошибка при очистке: {e}")

        print("👋 George-Droid завершён")


if __name__ == "__main__":
    main()