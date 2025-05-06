import os
import google.auth
import google.auth.exceptions
from dotenv import load_dotenv

load_dotenv()

SERVICE_ACCOUNT_PATH = os.getenv('VERTEXAI_SERVICE_ACCOUNT_PATH')
PROJECT_ID = os.getenv('VERTEXAI_PROJECT_ID')

print("Проверка аутентификации Google Cloud...")

found_credentials = False

# 1. Проверка ключа сервис-аккаунта
if SERVICE_ACCOUNT_PATH:
    if os.path.exists(SERVICE_ACCOUNT_PATH):
        print(f"Найден ключ сервис-аккаунта: {SERVICE_ACCOUNT_PATH}")
        try:
            credentials, project = google.auth.load_credentials_from_file(SERVICE_ACCOUNT_PATH)
            print(f"Ключ успешно загружен для проекта: {project}")
            found_credentials = True
        except Exception as e:
            print(f"Ошибка загрузки ключа: {e}")
    else:
        print(f"Указан путь к ключу ({SERVICE_ACCOUNT_PATH}), но файл не найден.")

# 2. Проверка Application Default Credentials (ADC), если ключ не найден/не указан
if not found_credentials:
    print("Проверка Application Default Credentials (ADC)...")
    try:
        credentials, project = google.auth.default()
        print(f"Найдены ADC для проекта: {project}")
        if PROJECT_ID and project and PROJECT_ID != project:
            print(
                f"ВНИМАНИЕ: Проект в ADC ({project}) отличается от проекта в .env ({PROJECT_ID}). Будет использован проект из .env.")
        elif not PROJECT_ID and project:
            print(f"ИНФО: Проект в .env не указан, будет использован проект из ADC: {project}")

        found_credentials = True
    except google.auth.exceptions.DefaultCredentialsError:
        print("ADC не найдены.")
    except Exception as e:
        print(f"Ошибка при проверке ADC: {e}")

# Итог
if found_credentials:
    print("\nАутентификация настроена.")
    if not PROJECT_ID:
        print(
            "ПРЕДУПРЕЖДЕНИЕ: VERTEXAI_PROJECT_ID не указан в .env. Убедитесь, что проект из ADC корректен или укажите ID проекта в .env.")
else:
    print("\nОШИБКА: Аутентификация Google Cloud не настроена.")
    print("Рекомендации:")
    print("1. Откройте терминал и выполните: gcloud auth application-default login")
    print("2. ИЛИ создайте ключ сервис-аккаунта в Google Cloud Console, скачайте JSON-файл")
    print("   и укажите путь к нему в файле .env в переменной VERTEXAI_SERVICE_ACCOUNT_PATH.")