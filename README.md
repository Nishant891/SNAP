# 🤖 SNAP — Smart Neural Assistant Program

SNAP is a **voice-interactive chatbot** powered by deep learning. It acts as a personal assistant, capable of understanding spoken input and responding using natural-sounding speech. SNAP uses NLP techniques and a trained neural network to classify intents and generate meaningful responses.

---

## ✨ Features

- 🧠 Deep learning–based intent recognition
- 🗣️ Voice input via microphone (using `SpeechRecognition`)
- 🔊 Speech output using `pyttsx3` text-to-speech
- 🗃️ Trainable model using `intents.json` file
- 🔁 Persistent training data via `pickle` for faster reuse

---

## 🛠️ Tech Stack

| Component         | Library              |
|------------------|----------------------|
| NLP               | `nltk`               |
| Model Building    | `tflearn`, `TensorFlow` |
| Voice Input       | `SpeechRecognition`  |
| Speech Output     | `pyttsx3`            |
| Serialization     | `pickle`             |

---

## 📂 Folder Structure

```
📁 project-root/
├── intents.json       # Your predefined intents and responses
├── xyz.pickle         # Serialized training data (generated)
├── model.tflearn      # Trained model (saved)
└── snap.py            # Main chatbot script
```

---

## 🚀 How to Run

### 1. 🔧 Install Dependencies

```bash
pip install nltk tflearn tensorflow pyttsx3 SpeechRecognition pyaudio
```

> **Note:** If you face issues with `pyaudio`, install it via:
> ```bash
> pip install pipwin
> pipwin install pyaudio
> ```

---

### 2. 📁 Prepare the `intents.json`

Create a file named `intents.json` in the same directory. It should contain structured data like:

```json
{
  "intents": [
    {
      "tag": "greeting",
      "patterns": ["Hi", "Hello", "Is anyone there?"],
      "responses": ["Hello!", "Hi, how can I help you?"]
    }
  ]
}
```

---

### 3. 🧠 Train and Run the Bot

```bash
python snap.py
```

On first run:
- It will train the model from `intents.json`
- Save the model in `model.tflearn` and `xyz.pickle`

On subsequent runs:
- It loads from saved model and skips training

---

## 🎙️ Usage

Once running, SNAP will:
1. Prompt you with voice: **"Hey there! I am SNAP. How can I help you?"**
2. Listen to your microphone input
3. Predict intent from speech
4. Respond back via audio and text

Say `"quit"`, `"done"` or `"thank you"` to exit.

---

## 📌 Notes

- Ensure your mic is connected and working
- Works best in quiet environments
- You can expand intents in `intents.json` to teach SNAP new skills

---

## 📜 License

This project is open-source and free to use under the MIT License.

---

## 🙋‍♂️ Author

Built with ❤️ by [Your Name Here]
