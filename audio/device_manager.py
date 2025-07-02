import sounddevice as sd
import sys


def list_audio_devices(kind='output'):
    """Список доступных аудио устройств"""
    devices = sd.query_devices()
    valid_devices = []
    print(f"\nДоступные устройства вывода ({kind}):")
    for i, device in enumerate(devices):
        if device['max_output_channels'] > 0 and device['hostapi'] != 0:
            print(f"  {len(valid_devices)}. {device['name']} (ID: {device['index']})")
            valid_devices.append(device)
    return valid_devices


def choose_audio_output_device():
    """Выбор аудио устройства для вывода"""
    output_devices = list_audio_devices()
    if not output_devices:
        print("Не найдено подходящих устройств вывода. Будет использовано системное по умолчанию (если доступно).")
        try:
            default_output_idx = sd.default.device[1] if isinstance(sd.default.device, (list, tuple)) and len(
                sd.default.device) > 1 else sd.default.device
            if default_output_idx != -1:
                chosen_output_device_id = default_output_idx
                print(
                    f"Установлено системное устройство вывода по умолчанию: ID {chosen_output_device_id} ({sd.query_devices(chosen_output_device_id)['name']})")
            else:
                chosen_output_device_id = None
        except Exception as e_default:
            print(f"Ошибка при попытке установить системное устройство вывода по умолчанию: {e_default}")
            chosen_output_device_id = None
        return chosen_output_device_id

    while True:
        try:
            default_device_prompt_info = ""
            try:
                default_output_idx_for_prompt = sd.default.device[1] if isinstance(sd.default.device,
                                                                                   (list, tuple)) and len(
                    sd.default.device) > 1 else sd.default.device
                if default_output_idx_for_prompt != -1:
                    default_device_prompt_info = f" [{default_output_idx_for_prompt}]"
            except:
                pass

            choice_str = input(
                f"Выберите номер устройства вывода (или Enter для системного по умолчанию{default_device_prompt_info}): ")

            if not choice_str:
                try:
                    default_output_idx = sd.default.device[1] if isinstance(sd.default.device, (list, tuple)) and len(
                        sd.default.device) > 1 else sd.default.device
                    if default_output_idx != -1:
                        chosen_output_device_id = default_output_idx
                        print(
                            f"Используется системное устройство вывода по умолчанию: ID {chosen_output_device_id} ({sd.query_devices(chosen_output_device_id)['name']})")
                    else:
                        chosen_output_device_id = None
                except Exception as e_enter_default:
                    print(f"Ошибка при выборе системного устройства по умолчанию (Enter): {e_enter_default}")
                    chosen_output_device_id = None
                return chosen_output_device_id

            device_index_in_list = int(choice_str)
            if 0 <= device_index_in_list < len(output_devices):
                chosen_device_info = output_devices[device_index_in_list]
                chosen_output_device_id = chosen_device_info['index']
                print(f"Выбрано и установлено глобально: {chosen_device_info['name']} (ID: {chosen_output_device_id})")
                return chosen_output_device_id
            else:
                print("Неверный номер. Попробуйте снова.")
        except ValueError:
            print("Неверный ввод. Введите число.")
        except Exception as e:
            print(f"Ошибка выбора устройства: {e}")
            return None


def get_default_microphone():
    """Получение микрофона по умолчанию"""
    default_mic, mic_name = None, "N/A"
    try:
        dev_info = sd.query_devices(kind='input')
        if isinstance(dev_info, dict) and 'index' in dev_info:
            default_mic, mic_name = dev_info['index'], dev_info.get('name', 'N/A')
        elif hasattr(sd.default, 'device'):
            default_device_indices = sd.default.device
            input_device_index = default_device_indices[0] if isinstance(default_device_indices, (list, tuple)) and len(
                default_device_indices) > 0 else default_device_indices
            if input_device_index != -1:
                default_mic = input_device_index
                mic_info = sd.query_devices(default_mic)
                if isinstance(mic_info, dict):
                    mic_name = mic_info.get('name', 'N/A')
    except Exception as e_mic:
        print(f"Ошибка определения микрофона по умолчанию: {e_mic}.", file=sys.stderr)

    return default_mic, mic_name