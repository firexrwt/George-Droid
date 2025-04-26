
# 🤖 George-Droid

> AI-Powered Streaming Companion • ИИ-компаньон для стримов  
> Inspired by Neuro-Sama by Vedal987 • Вдохновлён Neuro-Sama от Vedal987

---

## 📜 Description | Описание

George-Droid is a multifunctional streaming assistant built in Python. It listens to your voice, responds with humor, and interacts with your Twitch chat like a true co-host.  
Powered by local LLMs (via Ollama), real-time speech recognition (Faster-Whisper), and TTS (Piper), it can even play chess with you on stream. The intelligence of the bot depends on what model is used in it (by default llama3.1-8b-instruct-q5_K_S is used as the main model, faster whisper medium with compute type int8_float16 is used as a model for speech recognition. The voice can be changed by downloading .onnx and .onnx.json files and replacing the file names in VOICE_MODEL_PATH and VOICE_CONFIG_PATH variables).

George-Droid — многофункциональный стриминговый ИИ-компаньон на Python. Он распознаёт речь, остроумно отвечает, общается с чатом Twitch и даже может сыграть с вами в шахматы.  
Работает на локальных LLM через Ollama, STT через Faster-Whisper и голосит с помощью Piper. Интеллект бота зависит от того какая модель используется в нём(по умолчанию в качестве главной модели используется llama3.1-8b-instruct-q5_K_S, в качестве модели для распознавания речи используется faster whisper medium с compute type int8_float16. Голос можно поменять скачав .onnx и .onnx.json файлы и замены названий файлов в переменных VOICE_MODEL_PATH и VOICE_CONFIG_PATH)

---

## 🚀 Features | Возможности

- 🧠 Local LLM (Ollama) for smart replies • Ответы от локального LLM (через Ollama)
- 🎙️ Faster-Whisper STT + VAD • Распознавание речи с VAD
- 🗣️ Piper TTS • Голосовая озвучка
- 💬 Twitch chat bot (triggered by name/highlight) • Бот в чате Twitch
- 🕹️ Voice-triggered Chess with Stockfish + GUI • Шахматы с голосовым управлением и GUI
- 🔁 Idle monologues, hotkey toggles • Монологи во время тишины, управление горячими клавишами

---

## 🛠️ Setup | Установка

### Requirements | Требования

- Python 3.10+
- `piper.exe` + .onnx voice models (download separately)
- Stockfish UCI engine (Windows binary)
- Local LLM via [Ollama](https://ollama.com)
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
```

Make sure you’ve downloaded:
  
- Voice model `.onnx` → `voices/`
- Stockfish binary → `STOCKFISH_PATH`

---

## 🎛️ Customization | Настройка

- Change the **system prompt** and **bot name** in `main.py` (look for `SYSTEM_PROMPT`)
- Use hotkeys:
  - `Ctrl+;` → toggle speech recognition (STT)
  - `Ctrl+'` → toggle Twitch chat reaction
- You can add your own `.onnx` voice models
- Chess GUI supports Unicode or image-based pieces (see `USE_IMAGES`)

---

## ♟️ Chess Mode | Режим шахмат

George-Droid has a built-in chess mode powered by **Stockfish** and a GUI (`pysimplegui_chess_gui.py`).  
You can start/stop the game via voice commands like:

- "Джордж, давай в шахматы" → Start  
- "Останови шахматы" → Stop  

---

## 📁 Project Structure | Структура

```
George-Droid/
├── main.py                    # Main assistant logic
├── pysimplegui_chess_gui.py   # Chess GUI
├── requirements.txt
├── .env
├── piper_tts_bin/             # Piper TTS binary
├── voices/                    # .onnx voice models
├── images_chess/              # (Optional) piece images
└── obs_ai_response.txt        # Output text for OBS overlays
```

---

## 🧠 Tech Stack | Технологии

- LLM API: [Ollama](https://ollama.com)
- STT: [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- TTS: [Piper TTS](https://github.com/rhasspy/piper)
- Chat: [twitchio](https://github.com/TwitchIO/TwitchIO)
- Chess GUI: [PySimpleGUI](https://github.com/PySimpleGUI/PySimpleGUI)

---

## 📜 License

MIT License

---

## ✨ Credits

- Neuro-Sama by Vedal987 — the inspiration behind it all  
- Piper by Rhasspy   
- Made with ❤️ by [FIREX (Stepan)](https://github.com/firexrwt)
