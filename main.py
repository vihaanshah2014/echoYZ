# Install required packages first:
# pip install sounddevice numpy pydub requests python-dotenv

import os
import base64
import requests
from dotenv import load_dotenv
import sounddevice as sd
import numpy as np
import wave
from io import BytesIO

# Load environment variables from .env file
load_dotenv()

# Configuration
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
SARVAM_TTS_URL = "https://api.sarvam.ai/text-to-speech"
DEEPSEEK_CHAT_URL = "https://api.deepseek.com/v1/chat/completions"

def deepseek_chat(user_input, conversation_history=[]):
    """Get response from DeepSeek chatbot using direct API call"""
    print("\n[DeepSeek] Sending request...")

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }

    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        *conversation_history,
        {"role": "user", "content": user_input}
    ]

    payload = {
        "model": "deepseek-reasoner",
        "messages": messages,
        "stream": False
    }

    try:
        response = requests.post(DEEPSEEK_CHAT_URL, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        print("[DeepSeek] Response received successfully")
        return data['choices'][0]['message']['content']
    except Exception as e:
        print(f"[DeepSeek] Error occurred: {str(e)}")
        return f"Error in chatbot: {str(e)}"


def sarvam_tts(text, language="hi-IN", speaker="meera"):
    """
    Convert text to speech using Sarvam.ai API and return base64-encoded audio (WAV).
    The returned data is expected to be a 16 kHz, mono, 16-bit PCM WAV file.
    """
    print("\n[Sarvam TTS] Sending request...")

    headers = {
        "Content-Type": "application/json",
        "API-Subscription-Key": SARVAM_API_KEY
    }

    payload = {
        "inputs": [text],
        "target_language_code": language,
        "speaker": speaker,
        "pace": 1.26,
        "speech_sample_rate": 16000,
        "enable_preprocessing": True,
        "model": "bulbul:v1"
    }

    try:
        response = requests.post(SARVAM_TTS_URL, json=payload, headers=headers)
        response.raise_for_status()
        print("[Sarvam TTS] Audio generated successfully")
        return response.json()["audios"][0]
    except Exception as e:
        print(f"[Sarvam TTS] Error occurred: {str(e)}")
        return f"Error in TTS API: {str(e)}"


def play_audio(base64_audio):
    """
    Decode and play 16kHz mono 16-bit PCM audio in a WAV container.
    """
    print("[Audio] Starting playback...")

    try:
        audio_bytes = base64.b64decode(base64_audio)
        with BytesIO(audio_bytes) as audio_file:
            with wave.open(audio_file, 'rb') as wav:
                assert wav.getnchannels() == 1, "Audio must be mono"
                assert wav.getsampwidth() == 2, "16-bit PCM expected"
                assert wav.getframerate() == 16000, "16kHz sample rate expected"

                frames = wav.readframes(wav.getnframes())
                audio_array = np.frombuffer(frames, dtype=np.int16)
                audio_array = audio_array.astype(np.float32) / 32768.0

                sd.play(audio_array, 16000)
                sd.wait()

        print("[Audio] Playback completed successfully")
    except Exception as e:
        print(f"[Audio] Error during playback: {str(e)}")
        with open("debug_audio.wav", "wb") as f:
            f.write(audio_bytes)
        print("[Audio] Debug audio saved as debug_audio.wav")


def main():
    conversation_history = []
    print("\n=== Starting Conversation Interface ===")
    print("Type 'exit' or 'quit' to end the conversation")

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ['exit', 'quit']:
            break

        print(f"\n[Main] Processing user input: {user_input}")
        chat_response = deepseek_chat(user_input, conversation_history)
        print(f"\nAssistant: {chat_response}")

        conversation_history.extend([
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": chat_response}
        ])

        if chat_response.strip():
            tts_response = sarvam_tts(chat_response)
            if isinstance(tts_response, str) and tts_response.startswith("Error"):
                print("\nTTS Error:", tts_response)
            else:
                print("\nPlaying audio...")
                play_audio(tts_response)
        else:
            print("\nNo content to synthesize.")

    print("\n=== Conversation Ended ===")


if __name__ == "__main__":
    main()