import asyncio
import time
import datetime

# Глобальные переменные (будут установлены извне)
recording_active = None
audio_queue = None
is_processing_event = None
stt_enabled = True


def set_globals(rec_active, audio_q, processing_event):
    """Установка глобальных переменных"""
    global recording_active, audio_queue, is_processing_event
    recording_active = rec_active
    audio_queue = audio_q
    is_processing_event = processing_event


def get_stt_enabled():
    """Получение состояния STT"""
    return stt_enabled


def set_stt_enabled(enabled):
    """Установка состояния STT"""
    global stt_enabled
    stt_enabled = enabled


async def diagnostic_loop():
    """Периодическая диагностика состояния системы"""
    while recording_active.is_set():
        await asyncio.sleep(5)  # Каждые 5 секунд

        if not audio_queue:
            continue

        queue_size = audio_queue.qsize()
        processing = is_processing_event.is_set()

        if queue_size > 100 or (processing and queue_size > 10):
            print(f"[DIAG] Warning: Audio queue size: {queue_size}, Processing: {processing}, STT: {stt_enabled}")

            # Принудительная очистка при переполнении
            if queue_size > 500:
                with audio_queue.mutex:
                    audio_queue.queue.clear()
                print("[DIAG] Force cleared audio queue due to overflow")


async def processing_watchdog():
    """Следит за тем, чтобы флаг обработки не завис"""
    MAX_PROCESSING_TIME = 120  # 2 минуты максимум на обработку

    while recording_active.is_set():
        await asyncio.sleep(10)  # Проверка каждые 10 секунд

        if is_processing_event.is_set():
            # Запоминаем время начала обработки
            start_time = time.time()

            # Ждём завершения или таймаута
            while is_processing_event.is_set() and recording_active.is_set():
                if time.time() - start_time > MAX_PROCESSING_TIME:
                    print(f"[WATCHDOG] Processing timeout detected! Force clearing after {MAX_PROCESSING_TIME}s")
                    is_processing_event.clear()

                    # Восстанавливаем STT если был включен
                    if not stt_enabled:
                        stt_enabled = True
                        print("[WATCHDOG] Force re-enabled STT")
                    break

                await asyncio.sleep(1)


async def memory_consolidation_loop():
    """Периодически анализирует и консолидирует память"""
    memory_store_instance = None

    # Получаем экземпляр памяти из процессора
    try:
        from core.processor import memory_store_instance
    except ImportError:
        pass

    while recording_active.is_set():
        await asyncio.sleep(300)  # Каждые 5 минут

        if not memory_store_instance:
            continue

        try:
            # Получаем все воспоминания
            all_memories = memory_store_instance.retrieve_memories(
                "анализ всех фактов",  # Общий запрос
                top_k=100
            )

            # Группируем похожие факты
            # TODO: Реализовать умную консолидацию через LLM

            print(f"[Memory] Проверка консолидации: {len(all_memories)} фактов в памяти")

        except Exception as e:
            print(f"[Memory] Ошибка консолидации: {e}")


def get_system_stats():
    """Получение статистики системы"""
    return {
        "queue_size": audio_queue.qsize() if audio_queue else 0,
        "processing": is_processing_event.is_set() if is_processing_event else False,
        "stt_enabled": stt_enabled,
        "recording_active": recording_active.is_set() if recording_active else False,
        "timestamp": datetime.datetime.now().strftime('%H:%M:%S')
    }


def print_system_status():
    """Вывод состояния системы"""
    stats = get_system_stats()
    print(f"[STATUS] {stats['timestamp']} - Queue: {stats['queue_size']}, "
          f"Processing: {stats['processing']}, STT: {stats['stt_enabled']}, "
          f"Recording: {stats['recording_active']}")


async def status_monitor_loop():
    """Периодический вывод статуса системы"""
    while recording_active.is_set():
        await asyncio.sleep(30)  # Каждые 30 секунд
        print_system_status()


# Объединённая функция для запуска всех диагностических задач
async def run_all_diagnostics():
    """Запуск всех диагностических задач"""
    tasks = [
        diagnostic_loop(),
        processing_watchdog(),
        memory_consolidation_loop(),
        status_monitor_loop()
    ]

    await asyncio.gather(*tasks, return_exceptions=True)