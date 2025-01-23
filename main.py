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
import json
from datetime import datetime, timezone, timedelta
from pydantic import BaseModel
from openai import OpenAI
from typing import Optional, Dict, List
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
import pickle
import os.path

# Load environment variables from .env file
load_dotenv()

# Configuration
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SARVAM_TTS_URL = "https://api.sarvam.ai/text-to-speech"
DEEPSEEK_CHAT_URL = "https://api.deepseek.com/v1/chat/completions"

# Add to configuration section
SCOPES = ['https://www.googleapis.com/auth/calendar.readonly']

class UserInfo(BaseModel):
    emotion: Optional[str] = None
    preferences: Optional[Dict[str, str]] = None
    facts: Optional[List[str]] = None

class UserProfile:
    def __init__(self):
        if not OPENAI_API_KEY:
            raise ValueError("OpenAI API key not found in environment variables")
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.profile = {
            "emotions": {},
            "preferences": {},
            "facts": [],
            "last_interaction": None,
            "mood_history": []
        }
    
    def update_from_interaction(self, user_message, assistant_response):
        """Analyze interaction to update user profile"""
        self.profile["last_interaction"] = datetime.now().isoformat()
        
        try:
            completion = self.client.beta.chat.completions.parse(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": """Extract information about the user from this conversation.
                    Include:
                    - Current emotion/mood
                    - Any preferences or likes/dislikes mentioned
                    - Personal facts or details shared"""},
                    {"role": "user", "content": f"User message: {user_message}\nAssistant response: {assistant_response}"},
                ],
                response_format=UserInfo,
            )
            
            info = completion.choices[0].message.parsed
            
            # Update emotions if detected
            if info.emotion:
                self.profile["emotions"][datetime.now().isoformat()] = info.emotion
                self.profile["mood_history"].append({
                    "timestamp": datetime.now().isoformat(),
                    "mood": info.emotion
                })
            
            # Update preferences
            if info.preferences:
                self.profile["preferences"].update(info.preferences)
            
            # Add new facts
            if info.facts:
                for fact in info.facts:
                    if fact not in self.profile["facts"]:
                        self.profile["facts"].append(fact)
                        
        except Exception as e:
            print(f"[Profile] Error analyzing interaction: {str(e)}")
    
    def save(self, filename="user_profile.json"):
        """Save user profile to file"""
        with open(filename, 'w') as f:
            json.dump(self.profile, f, indent=2)
    
    def load(self, filename="user_profile.json"):
        """Load user profile from file"""
        try:
            with open(filename, 'r') as f:
                self.profile = json.load(f)
        except FileNotFoundError:
            print("[Profile] No existing profile found, starting fresh")

def get_calendar_events():
    """Get calendar events for the next week from all calendars except pmready"""
    creds = None
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)

    try:
        service = build('calendar', 'v3', credentials=creds)
        
        # Get current time and time week ahead
        now = datetime.now(timezone.utc)
        week_later = now + timedelta(days=7)
        
        calendar_list = service.calendarList().list().execute()
        all_events = []
        
        for calendar_list_entry in calendar_list['items']:
            calendar_id = calendar_list_entry['id']
            calendar_name = calendar_list_entry.get('summary', 'Unnamed Calendar')
            
            # Skip the pmready calendar
            if calendar_id == 'pmready.official@gmail.com':
                continue
            
            events_result = service.events().list(
                calendarId=calendar_id,
                timeMin=now.isoformat(),
                timeMax=week_later.isoformat(),
                singleEvents=True,
                orderBy='startTime',
                maxResults=100
            ).execute()
            
            events = events_result.get('items', [])
            for event in events:
                event['calendar_name'] = calendar_name
                all_events.append(event)
        
        # Sort all events by start time
        all_events.sort(key=lambda x: x['start'].get('dateTime', x['start'].get('date', '')))
        return all_events
        
    except Exception as e:
        print(f"[Calendar] Error fetching events: {str(e)}")
        return []

def deepseek_chat(user_input, conversation_history=[]):
    """Get response from DeepSeek chatbot using direct API call"""
    print("\n[Bhaskar] Processing...")

    # Get calendar events
    calendar_events = get_calendar_events()
    events_context = "Upcoming calendar events:\n"
    if calendar_events:
        for event in calendar_events:
            start = event['start'].get('dateTime', event['start'].get('date'))
            summary = event.get('summary', 'No title')
            calendar_name = event.get('calendar_name', 'Unknown Calendar')
            events_context += f"- [{calendar_name}] {summary} at {start}\n"
    else:
        events_context += "No upcoming events scheduled\n"

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Include user profile context in the system message
    profile_context = f"""Previous knowledge about the user:
    - Current mood history: {user_profile.profile['mood_history'][-3:] if user_profile.profile['mood_history'] else 'Unknown'}
    - Known preferences: {user_profile.profile['preferences']}
    - Important facts: {user_profile.profile['facts']}"""

    bhaskar_persona = """You are Bhaskar, a highly efficient AI assistant focused on brevity and precision.
    Your key characteristics:
    - Extremely concise and direct in responses
    - Always address the user as 'sir'
    - Minimal small talk
    - Purely factual and straight to the point
    - Focus only on essential information
    - Professional and formal tone
    
    Keep all responses brief and precise, regardless of the question type."""

    messages = [
        {"role": "system", "content": f"{bhaskar_persona}\n\nRegarding the user: {profile_context}\n\n{events_context}"},
        *conversation_history[-10:],
        {"role": "user", "content": user_input}
    ]

    payload = {
        "model": "deepseek-chat",
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


def sarvam_tts(text, language="hi-IN", speaker="amartya"):
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
        "pace": 1.16,
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
    global user_profile
    user_profile = UserProfile()
    user_profile.load()
    
    conversation_history = []
    print("\n=== Connecting to Bhaskar Interface ===")
    print("Enable speaker mode? (yes/no)")
    speaker_mode = input().lower().strip() in ['yes', 'y']
    print("Type 'exit' or 'quit' to end")
    print(f"Speaker mode: {'enabled' if speaker_mode else 'disabled'}")
    print("\nBhaskar: Good day, sir. How may I assist?")

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ['exit', 'quit']:
            break

        print(f"\n[Main] Processing user input: {user_input}")
        chat_response = deepseek_chat(user_input, conversation_history)
        print(f"\nAssistant: {chat_response}")

        # Update user profile
        user_profile.update_from_interaction(user_input, chat_response)
        user_profile.save()

        conversation_history.extend([
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": chat_response}
        ])

        # Only process TTS if speaker mode is enabled
        if speaker_mode and chat_response.strip():
            tts_response = sarvam_tts(chat_response)
            if isinstance(tts_response, str) and tts_response.startswith("Error"):
                print("\nTTS Error:", tts_response)
            else:
                print("\nPlaying audio...")
                play_audio(tts_response)

    print("\n=== Conversation Ended ===")


if __name__ == "__main__":
    main()