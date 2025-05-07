
# 🤖 George-Droid

> AI-Powered Streaming Companion • ИИ-компаньон для стримов  
> Inspired by Neuro-Sama by Vedal987 • Вдохновлён Neuro-Sama от Vedal987

---

## 📜 Description | Описание

George-Droid is a multifunctional streaming assistant built in Python. It listens to your voice, responds with humor, and interacts with your Twitch chat like a true co-host.  
Powered by AI Platform Vertex AI (with models like Llama 4 Scout Instruct, or any other compatible MaaS model), real-time speech recognition (Faster-Whisper) and TTS (Piper). The intelligence of the bot depends on what model is used in it (by default llama-4-scout-17b-16e-instruct-maas is used as the main model, faster whisper medium with compute type int8_float16 is used as a model for speech recognition. The voice can be changed by downloading .onnx and .onnx.json files and replacing the file names in VOICE_MODEL_PATH and VOICE_CONFIG_PATH variables).

George-Droid — многофункциональный стриминговый ИИ-компаньон на Python. Он распознаёт речь, остроумно отвечает и общается с чатом Twitch.  
Работает через Vertex AI (с моделями как Llama 4 Scout Instruct, или любой другой совместимой моделью как услуга (MaaS)), STT через Faster-Whisper и голосит с помощью Piper. Интеллект бота зависит от того какая модель используется в нём(по умолчанию в качестве главной модели используется llama-4-scout-17b-16e-instruct-maas, в качестве модели для распознавания речи используется faster whisper medium с compute type int8_float16. Голос можно поменять скачав .onnx и .onnx.json файлы и замены названий файлов в переменных VOICE_MODEL_PATH и VOICE_CONFIG_PATH)

---

## 🚀 Features | Возможности

- 🧠 Vertex AI LLM (e.g., Llama 4 Scout Instruct) for smart replies • Ответы от LLM через Vertex AI (например, Llama 4 Scout Instruct)
- 🎙️ Faster-Whisper STT + VAD • Распознавание речи с VAD
- 🗣️ Piper TTS • Голосовая озвучка
- 💬 Twitch chat bot (triggered by name/highlight) • Бот в чате Twitch
- 🔁 Idle monologues, hotkey toggles • Монологи во время тишины, управление горячими клавишами

---

## 🛠️ Setup | Установка

### Requirements | Требования

- Python 3.10+
- `piper.exe` + .onnx voice models (download separately)
- LLM API: Google Cloud Vertex AI
- NVidia CUDNN v9.8 & CUDA v12.8

### Installation | Установка

```bash
git clone https://github.com/firexrwt/George-Droid.git
cd George-Droid
pip install -r requirements.txt
```

Create a `.env` file:

```
TWITCH_ACCESS_TOKEN=...
TWITCH_BOT_NICK=*your_twitch_nick*
TWITCH_CHANNEL=*your_twitch_nick*
TWITCH_REFRESH_TOKEN=...
TWITCH_CLIENT_ID=...
TWITCH_CLIENT_SECRET=...
VERTEXAI_PROJECT_ID=*project-id*
VERTEXAI_LOCATION=*location for model*
VERTEXAI_MODEL_NAME=*full model name with publishers*
VERTEXAI_SERVICE_ACCOUNT_PATH=*path to your account json key*
```

Make sure you’ve downloaded:
  
- Voice model `.onnx` → `voices/`
- Vertex AI account key 

---

## 🎛️ Customization | Настройка

- Change the **system prompt** and **bot name** in `main.py` (look for `SYSTEM_PROMPT`)
- Use hotkeys:
  - `Ctrl+;` → toggle speech recognition (STT)
  - `Ctrl+'` → toggle Twitch chat reaction
- You can add your own `.onnx` voice models

---

## 📁 Project Structure | Структура

```
George-Droid/
├── main.py                    # Main assistant logic
├── requirements.txt
├── .env
├── piper_tts_bin/             # Piper TTS binary
├── voices/                    # .onnx voice models
└── obs_ai_response.txt        # Output text for OBS overlays
```

---

## 🧠 Tech Stack | Технологии

- LLM API: Google Cloud Vertex AI
- STT: [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- TTS: [Piper TTS](https://github.com/rhasspy/piper)
- Chat: [twitchio](https://github.com/TwitchIO/TwitchIO)

---

## 📜 License

MIT License

---

## ✨ Credits

- Neuro-Sama by Vedal987 — the inspiration behind it all  
- Piper by Rhasspy   
- Made with ❤️ by [FIREX (Stepan)](https://firexrwt.github.io)
