import os
import sys
import tempfile
import mss
from PIL import Image

from config.settings import SCREENSHOT_TEMP_DIR, SCREENSHOT_TARGET_WIDTH, SCREENSHOT_TARGET_HEIGHT

# Глобальная переменная для выбранного монитора
selected_monitor_info = None


def list_monitors_and_select():
    """Показывает список мониторов и позволяет выбрать один"""
    global selected_monitor_info

    try:
        with mss.mss() as sct:
            monitors = sct.monitors
    except Exception as e:
        print(f"Ошибка при получении списка мониторов: {e}", file=sys.stderr)
        monitors = []

    if not monitors or len(monitors) <= 1:
        if len(monitors) == 1 and monitors[0]['width'] > 0 and monitors[0]['height'] > 0:
            print("Обнаружен только один \"монитор\" (возможно, все экраны объединены). Выбирается по умолчанию.")
            selected_monitor_info = monitors[0]
            return
        print("Не удалось найти доступные мониторы или найдено меньше двух. Скриншоты могут работать некорректно.",
              file=sys.stderr)
        if monitors and monitors[0]['width'] > 0:
            print(
                f"Попытка использовать главный экран: {monitors[0]['width']}x{monitors[0]['height']} at ({monitors[0]['left']},{monitors[0]['top']})")
            selected_monitor_info = monitors[0]
        else:
            selected_monitor_info = None
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
        selected_monitor_info = monitors[0]
        return

    while True:
        try:
            choice_str = input(f"Выберите номер монитора для скриншотов (0-{len(valid_monitors_for_selection) - 1}): ")
            choice_idx = int(choice_str)
            if 0 <= choice_idx < len(valid_monitors_for_selection):
                selected_monitor_info = valid_monitors_for_selection[choice_idx]["details"]
                print(
                    f"Выбран монитор ID {valid_monitors_for_selection[choice_idx]['id_mss']} ({selected_monitor_info['width']}x{selected_monitor_info['height']}) для скриншотов.")
                return
            else:
                print("Неверный номер. Попробуйте снова.")
        except ValueError:
            print("Неверный ввод. Введите число.")
        except Exception as e:
            print(f"Ошибка выбора монитора: {e}", file=sys.stderr)
            selected_monitor_info = monitors[0]
            return


def capture_and_prepare_screenshot() -> str | None:
    """Захват и подготовка скриншота"""
    global selected_monitor_info

    if not selected_monitor_info:
        print("Монитор для скриншота не выбран или не найден.", file=sys.stderr)
        return None

    try:
        with mss.mss() as sct:
            sct_img = sct.grab(selected_monitor_info)
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
    """Удаление временного файла скриншота"""
    if file_path and os.path.exists(file_path):
        try:
            os.remove(file_path)
            print(f"Временный скриншот удален: {file_path}")
        except Exception as e:
            print(f"Ошибка при удалении временного скриншота {file_path}: {e}", file=sys.stderr)


def cleanup_temp_screenshots():
    """Очистка старых временных скриншотов"""
    if os.path.exists(SCREENSHOT_TEMP_DIR):
        try:
            for filename in os.listdir(SCREENSHOT_TEMP_DIR):
                if filename.endswith('.png'):
                    file_path = os.path.join(SCREENSHOT_TEMP_DIR, filename)
                    os.remove(file_path)
            print("Временные скриншоты очищены")
        except Exception as e:
            print(f"Ошибка очистки временных скриншотов: {e}", file=sys.stderr)