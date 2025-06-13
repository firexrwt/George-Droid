# 🤖 George-Droid

> AI-Powered Streaming Companion • ИИ-компаньон для стримов  
> Inspired by Neuro-Sama by Vedal987 • Вдохновлён Neuro-Sama от Vedal987

---

## 📜 Description | Описание

George-Droid is a multifunctional streaming assistant built in Python. It listens to your voice, responds with humor,
and interacts with your Twitch chat like a true co-host.  
Powered by Together AI (with models like Meta's Llama 4 Scout Instruct or any compatible Together AI model), real-time
speech recognition (Faster-Whisper), TTS (Piper), and a **contextual memory system (RAG)**. Intelligence of the bot
scales with the chosen model (by default
llama-4-scout-17b is used as the main LLM, faster-whisper medium with compute type int8 for STT). Change the
voice by swapping `.onnx` and `.onnx.json` files in `voices/` and updating `VOICE_MODEL_PATH` & `VOICE_CONFIG_PATH`.

George-Droid — многофункциональный стриминговый ИИ-компаньон на Python. Он распознаёт речь, остроумно отвечает и
общается с чатом Twitch.  
Работает через Together AI (с моделями типа Meta Llama 4 Scout Instruct или любой совместимой моделью), STT через
Faster-Whisper, озвучка через Piper, и **система контекстной памяти (RAG)**. Интеллект бота зависит от выбранной
модели (по умолчанию llama-4-scout-17b для
LLM, faster-whisper medium с compute type int8 для STT). Голос можно поменять, заменив файлы `.onnx` и
`.onnx.json` в `voices/` и обновив `VOICE_MODEL_PATH` и `VOICE_CONFIG_PATH`.

---

## 🚀 Features | Возможности

- 🧠 Together AI LLM (e.g., Llama 4 Scout Instruct) for smart replies • Ответы от LLM через Together AI (например, Llama
  4 Scout Instruct)
- 🎙️ Faster-Whisper STT + VAD • Распознавание речи с VAD
- 🗣️ Piper TTS • Голосовая озвучка
- 💬 Twitch chat bot (triggered by name/highlight) • Бот в чате Twitch
- 🔁 Idle monologues, hotkey toggles • Монологи во время тишины, управление горячими клавишами
- 🧠 **Contextual Memory (RAG)** powered by Sentence Transformers and FAISS for enhanced recall • **Контекстная Память (
  RAG)** на базе Sentence Transformers и FAISS для улучшенного вспоминания
- 📸 **Visual Context** with Screenshots for LLM analysis • **Визуальный Контекст** со скриншотами для анализа LLM

---

## 🛠️ Setup | Установка

### Requirements | Требования

- Python 3.10+
- `piper.exe` + .onnx voice models (download separately)
- LLM API: Together AI
- NVidia CUDNN v9.8 & CUDA v12.8

### Installation | Установка

```bash
git clone https://github.com/firexrwt/George-Droid.git
cd George-Droid
pip install -r requirements.txt
```

Create a `.env` file:

```env
TWITCH_ACCESS_TOKEN=...
TWITCH_BOT_NICK=*your_twitch_nick*
TWITCH_CHANNEL=*your_twitch_nick*
TWITCH_REFRESH_TOKEN=...
TWITCH_CLIENT_ID=...
TWITCH_CLIENT_SECRET=...
TOGETHERAI_API_KEY=*your-togetherai-api-key*
TOGETHERAI_MODEL_NAME=*model-id-or-name*
``` 

Make sure you’ve downloaded:

- Voice model `.onnx` → `voices/`

---

## 🎛️ Customization | Настройка

- Change the **system prompt** and **bot name** in `main.py` (search for `SYSTEM_PROMPT`)
- Use hotkeys:
    - `Ctrl+;` → toggle speech recognition (STT)
    - `Ctrl+'` → toggle Twitch chat reaction
- You can add your own `.onnx` voice models in `voices/`

---

## 📁 Project Structure | Структура

```
George-Droid/
├── main.py                    # Main assistant logic
├── requirements.txt
├── .env
├── piper_tts_bin/             # Piper TTS binary
├── voices/                    # .onnx voice models
├── data_george_memory         # memory folder
└── obs_ai_response.txt        # Output text for OBS overlays
```

---

## 🧠 Tech Stack | Технологии

- LLM API: Together AI
- STT: [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- TTS: [Piper TTS](https://github.com/rhasspy/piper)
- Chat: [twitchio](https://github.com/TwitchIO/TwitchIO)
- Memory: [FAISS](https://faiss.ai/), [Sentence-Transformers](https://www.sbert.net/)

---

## 📜 License

MIT License

---

## ✨ Credits

- Neuro-Sama by Vedal987 — the inspiration behind it all
- Piper by Rhasspy
- Made with ❤️ by [FIREX (Stepan)](https://firexrwt.github.io)
