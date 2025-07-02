# 🤖 George-Droid

> AI-Powered Streaming Companion • ИИ-компаньон для стримов  
> Inspired by Neuro-Sama by Vedal987 • Вдохновлён Neuro-Sama от Vedal987

---

## 📜 Description | Описание

George-Droid is a multifunctional streaming assistant built in Python. It listens to your voice, responds with humor,
and interacts with your Twitch chat like a true co-host.  
Powered by Together AI (with models like Meta's Llama 4 Scout Instruct), real-time speech recognition (Faster-Whisper),
TTS (Piper), and an **advanced contextual memory system (RAG)** with intelligent fact extraction and consolidation.

George-Droid — многофункциональный стриминговый ИИ-компаньон на Python. Он распознаёт речь, остроумно отвечает и
общается с чатом Twitch.  
Работает через Together AI (с моделями типа Meta Llama 4 Scout), STT через Faster-Whisper, озвучка через Piper, и 
**продвинутую систему контекстной памяти (RAG)** с умным извлечением и консолидацией фактов.

---

## 🚀 Features | Возможности

- 🧠 **Multi-Model AI System** • **Мульти-модельная ИИ система**
    - Main LLM: `meta-llama/Llama-4-Scout-17B-16E-Instruct` for responses
    - Context Analysis: `meta-llama/Llama-3.2-3B-Instruct-Turbo` for screenshot detection & memory query context
    - Memory Processing: `meta-llama/Llama-3.3-70B-Instruct-Turbo-Free` for fact extraction & memory storage

- 🎙️ **Advanced Voice Processing** • **Продвинутая обработка голоса**
    - Faster-Whisper STT with VAD (Voice Activity Detection) | Faster-Whisper STT с VAD (детекция голосовой активности)
    - Configurable models (medium/large) with CUDA acceleration | Настраиваемые модели (medium/large) с ускорением CUDA
    - Real-time speech recognition with noise filtering | Распознавание речи в реальном времени с фильтрацией шума

- 🗣️ **High-Quality TTS** • **Качественная озвучка**
    - Piper TTS with custom voice models | Piper TTS с пользовательскими моделями голосов
    - Async audio processing with device selection | Асинхронная обработка аудио с выбором устройства
    - Russian voice support (ru_RU-ruslan-medium) | Поддержка русских голосов (ru_RU-ruslan-medium)

- 💬 **Intelligent Chat Integration** • **Умная интеграция с чатом**
    - Twitch chat bot with smart triggers | Twitch чат-бот с умными триггерами
    - Context-aware responses based on chat history | Контекстные ответы на основе истории чата
    - Highlighted message prioritization | Приоритизация выделенных сообщений

- 🧠 **Advanced Memory System (RAG)** • **Продвинутая система памяти (RAG)**
    - FAISS vector indexing with Sentence Transformers | FAISS векторная индексация с Sentence Transformers
    - Intelligent fact extraction from conversations | Умное извлечение фактов из разговоров
    - Semantic deduplication and memory consolidation | Семантическая дедупликация и консолидация памяти
    - Context-aware retrieval with smart filtering | Контекстный поиск с умной фильтрацией
    - Automatic categorization (personal_info, preferences, events) | Автоматическая категоризация (личная_инфо,
      предпочтения, события)

- 📸 **Visual Context Analysis** • **Анализ визуального контекста**
    - Screenshot capture with intent detection | Захват скриншотов с определением намерений
    - Multi-monitor support with target resolution scaling | Поддержка нескольких мониторов с масштабированием
      разрешения
    - Visual reference understanding for enhanced responses | Понимание визуальных ссылок для улучшенных ответов

- ⚡ **Smart Automation** • **Умная автоматизация**
    - Idle monologues during silence periods | Монологи в периоды тишины
    - Hotkey controls for real-time management | Управление горячими клавишами в реальном времени
    - Automatic activity detection and response prioritization | Автоматическое определение активности и приоритизация
      ответов

- 🔧 **Advanced Configuration** • **Расширенные настройки**
    - CUDNN optimization for GPU acceleration | CUDNN оптимизация для ускорения GPU
    - Configurable audio devices and parameters | Настраиваемые аудио устройства и параметры
    - Customizable VAD thresholds and timing | Настраиваемые пороги VAD и тайминги
    - Multiple model configurations | Конфигурации нескольких моделей

---

## 🛠️ Setup | Установка

### Requirements | Требования

- Python 3.10+ | Python 3.10+
- **NVIDIA GPU with CUDA 12.8 + CUDNN v9.8 (REQUIRED)** | **NVIDIA GPU с CUDA 12.8 + CUDNN v9.8 (ОБЯЗАТЕЛЬНО)**
    - Download CUDNN: https://developer.nvidia.com/cudnn | Скачать CUDNN: https://developer.nvidia.com/cudnn
- `piper.exe` + .onnx voice models | `piper.exe` + .onnx модели голосов
- Together AI API access | Доступ к Together AI API

### Installation | Установка

```bash
git clone https://github.com/firexrwt/George-Droid.git
cd George-Droid
pip install -r requirements.txt
```

### Environment Configuration | Настройка окружения

Create a `.env` file with the following variables: | Создайте файл `.env` со следующими переменными:

```env
# Twitch Configuration | Конфигурация Twitch
TWITCH_ACCESS_TOKEN=your_twitch_oauth_token
TWITCH_BOT_NICK=your_twitch_bot_nick
TWITCH_CHANNEL=your_twitch_channel
TWITCH_REFRESH_TOKEN=your_refresh_token
TWITCH_CLIENT_ID=your_client_id
TWITCH_CLIENT_SECRET=your_client_secret

# Together AI Configuration | Конфигурация Together AI
TOGETHER_API_KEY=your_together_ai_api_key
TOGETHER_MODEL_ID=meta-llama/Llama-4-Scout-17B-16E-Instruct

# TTS Configuration (Piper) | Конфигурация TTS (Piper)
PIPER_EXE_PATH=piper_tts_bin/piper.exe
PIPER_VOICE_MODEL_PATH=voices/ru_RU-ruslan-medium.onnx
PIPER_VOICE_CONFIG_PATH=voices/ru_RU-ruslan-medium.onnx.json

# CUDA/CUDNN Configuration (Necessary) | Конфигурация CUDA/CUDNN (Обязательно)
CUDNN_PATH=C:\Program Files\NVIDIA\CUDNN\v9.8\bin\12.8
```

### Download Required Files | Скачайте необходимые файлы

1. **Piper TTS Binary**: Download `piper.exe` and place in `piper_tts_bin/` | **Piper TTS Бинарный файл**: Скачайте
   `piper.exe` и поместите в `piper_tts_bin/`
2. **Voice Models**: Download `.onnx` and `.onnx.json` voice files and place in `voices/` | **Модели голосов**: Скачайте
   файлы `.onnx` и `.onnx.json` голосов и поместите в `voices/`

---

## 🎛️ Customization | Настройка

### Bot Personality | Личность бота

- Edit `SYSTEM_PROMPT` in `config/settings.py` to change the bot's personality | Отредактируйте `SYSTEM_PROMPT` в `config/settings.py` для
  изменения личности бота
- Modify `BOT_NAME_FOR_CHECK` to change trigger name in chat | Измените `BOT_NAME_FOR_CHECK` для изменения триггерного
  имени в чате

### Voice Configuration | Настройка голоса

- Add custom `.onnx` voice models to `voices/` directory | Добавьте пользовательские `.onnx` модели голосов в папку
  `voices/`
- Update `PIPER_VOICE_MODEL_PATH` and `PIPER_VOICE_CONFIG_PATH` in `.env` | Обновите `PIPER_VOICE_MODEL_PATH` и
  `PIPER_VOICE_CONFIG_PATH` в `.env`

### Model Configuration | Настройка моделей

- Change `TOGETHER_MODEL_ID` for different response quality/speed | Измените `TOGETHER_MODEL_ID` для изменения
  качества/скорости ответов
- Modify `INTENT_ANALYSIS_MODEL_ID` and `CHUNKING_MODEL_ID` for specialized tasks | Измените `INTENT_ANALYSIS_MODEL_ID`
  и `CHUNKING_MODEL_ID` для специализированных задач

### Hotkey Controls | Управление горячими клавишами

- `Ctrl+;` → Toggle speech recognition (STT) | `Ctrl+;` → Переключить распознавание речи (STT)
- `Ctrl+'` → Toggle Twitch chat interaction | `Ctrl+'` → Переключить взаимодействие с чатом Twitch

### Audio Configuration | Настройка аудио

- The bot will prompt you to select audio output device on startup | Бот предложит выбрать устройство вывода аудио при
  запуске
- Monitor selection for screenshots is also configurable on startup | Выбор монитора для скриншотов также настраивается
  при запуске

---

## 📁 Project Structure | Структура проекта

```
George-Droid/
├── main.py                     # Main application logic | Основная логика приложения
├── backend/
│   └── memory_store.py        # RAG memory system implementation | Реализация системы памяти RAG
├── requirements.txt           # Python dependencies | Зависимости Python
├── .env                       # Environment configuration | Конфигурация окружения
├── piper_tts_bin/            # Piper TTS binary | Бинарный файл Piper TTS
├── voices/                   # Voice model files (.onnx) | Файлы моделей голосов (.onnx)
├── data_george_memory/       # Persistent memory storage | Постоянное хранение памяти
│   ├── george_memory.index   # FAISS vector index | FAISS векторный индекс
│   └── george_memory_meta.jsonl # Memory metadata | Метаданные памяти
├── screenshots_temp/         # Temporary screenshot storage | Временное хранение скриншотов
└── obs_ai_response.txt       # Output for OBS text overlays | Вывод для текстовых оверлеев OBS
```

---

## 🧠 Memory System Details | Детали системы памяти

### Fact Extraction | Извлечение фактов

- Automatic extraction of personal info, preferences, and events | Автоматическое извлечение личной информации,
  предпочтений и событий
- Smart categorization with confidence scoring | Умная категоризация с оценкой уверенности
- Entity recognition (dates, names, games, etc.) | Распознавание сущностей (даты, имена, игры и т.д.)
- Duplicate prevention with semantic similarity checking | Предотвращение дубликатов с проверкой семантического сходства

### Memory Categories | Категории памяти

- `personal_info` - Personal facts about users | `personal_info` - Личные факты о пользователях
- `preference` - User preferences and likes/dislikes | `preference` - Предпочтения пользователей и симпатии/антипатии
- `event` - Important events and milestones | `event` - Важные события и вехи
- `statement` - General statements and observations | `statement` - Общие утверждения и наблюдения
- `bot_response` - Bot's own responses for consistency | `bot_response` - Собственные ответы бота для консистентности

### Retrieval System | Система поиска

- Context-aware query processing | Контекстная обработка запросов
- Smart filtering by relevance and recency | Умная фильтрация по релевантности и свежести
- Multi-factor scoring (similarity, author, type) | Многофакторная оценка (сходство, автор, тип)
- Automatic consolidation of related memories | Автоматическая консолидация связанных воспоминаний

---

## 🔧 Performance Optimization | Оптимизация производительности

### GPU Acceleration | Ускорение на GPU

- CUDA support for Faster-Whisper STT | Поддержка CUDA для Faster-Whisper STT
- CUDNN optimization for neural networks | CUDNN оптимизация для нейронных сетей
- Configurable compute types (int8, float16, float32) | Настраиваемые типы вычислений (int8, float16, float32)

### Memory Management | Управление памятью

- Efficient FAISS indexing with embedding dimension validation | Эффективная FAISS индексация с валидацией размерности
  эмбеддингов
- Automatic index rebuilding on dimension mismatches | Автоматическое пересоздание индекса при несоответствии
  размерностей
- Lazy loading and caching for optimal performance | Ленивая загрузка и кэширование для оптимальной производительности

### Audio Processing | Обработка аудио

- Configurable VAD thresholds for noise environments | Настраиваемые пороги VAD для шумных сред
- Real-time resampling and channel conversion | Пересэмплинг и конвертация каналов в реальном времени
- Async processing to prevent blocking | Асинхронная обработка для предотвращения блокировок

---

## 🧠 Tech Stack | Технологии

- **LLM API**: [Together AI](https://www.together.ai/) with Meta Llama models
- **STT**: [faster-whisper](https://github.com/SYSTRAN/faster-whisper) with CUDA acceleration
- **TTS**: [Piper TTS](https://github.com/rhasspy/piper) for natural speech synthesis
- **Chat**: [twitchio](https://github.com/TwitchIO/TwitchIO) for Twitch integration
- **Memory**: [FAISS](https://faiss.ai/) + [Sentence-Transformers](https://www.sbert.net/) for RAG
- **Vision**: [PIL](https://pillow.readthedocs.io/) + [mss](https://python-mss.readthedocs.io/) for screenshots
- **Audio**: [sounddevice](https://python-sounddevice.readthedocs.io/) + [scipy](https://scipy.org/) for processing

---

## 🚀 Advanced Usage | Расширенное использование

### Custom Model Integration | Интеграция пользовательских моделей

You can easily switch between different Together AI models by modifying the model IDs in `main.py`: | Вы можете легко
переключаться между различными моделями Together AI, изменив ID моделей в `main.py`:

```python
TOGETHER_MODEL_ID = "meta-llama/Llama-3.1-70B-Instruct-Turbo"  # Main responses | Основные ответы
INTENT_ANALYSIS_MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct-Turbo"  # Intent analysis | Анализ намерений
CHUNKING_MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"  # Text processing | Обработка текста
```

### Memory System Tuning | Настройка системы памяти

Adjust memory parameters for different use cases: | Настройте параметры памяти для различных случаев использования:

```python
# In memory_store.py | В memory_store.py
VAD_ENERGY_THRESHOLD = 0.005        # Voice activity sensitivity | Чувствительность голосовой активности
VAD_SILENCE_TIMEOUT_MS = 1200       # Silence detection timeout | Таймаут детекции тишины
MAX_HISTORY_LENGTH = 10             # Conversation history size | Размер истории разговора
semantic_similarity_threshold = 0.97 # Duplicate detection sensitivity | Чувствительность детекции дубликатов
```

---

## 📜 License

MIT License

---

## ✨ Credits

- **Inspiration**: Neuro-Sama by Vedal987 — the inspiration behind it all
- **TTS**: Piper by Rhasspy team
- **Models**: Meta AI for Llama model family
- **Platform**: Together AI for model hosting
- Made with ❤️ by [FIREX (Stepan)](https://firexrwt.github.io)

---

## 📞 Support | Поддержка

For issues, questions, or contributions, please visit the [GitHub repository](https://github.com/firexrwt/George-Droid)
or contact the maintainer at stepanveremeev@gmail.com | По вопросам, проблемам или предложениям
посетите [GitHub репозиторий](https://github.com/firexrwt/George-Droid) или свяжитесь с разработчиком по адресу
stepanveremeev@gmail.com

---

*Built for streamers, by a streamer. Happy streaming! 🎮* | *Сделано стримером для стримеров. Удачных стримов! 🎮*
