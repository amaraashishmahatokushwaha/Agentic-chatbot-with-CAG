#for general setup of GUI , sql db, vector db, and general
import os
import sys
import sqlite3
import numpy as np
import cv2
import time
import pyaudio
import threading
import queue
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import PIL.Image
import PIL.ImageTk
import json
import pygame
import uuid
import wave
from gtts import gTTS
import ollama
import speech_recognition as sr
import logging
import traceback
import requests
import hashlib
import spacy
from deepface import DeepFace
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
    logger.info("Successfully loaded spaCy model: en_core_web_sm")
except Exception as e:
    logger.error(f"Failed to load spaCy model: {e}")
    sys.exit(1)

class LlamaAssistantGUI:
    def __init__(self, root):
        """Initialize the GUI and assistant components"""
        logger.info("Initializing LlamaAssistantGUI...")
        self.root = root
        self.root.title("Llama3 Offline Assistant")
        self.root.geometry("1200x800")
        self.root.configure(bg="#f0f0f0")

        # Create directories
        os.makedirs("data", exist_ok=True)
        logger.debug("Created data directory if it didn't exist")

        # Initialize DB connections
        logger.info("Setting up databases...")
        self.init_user_db()
        self.init_vector_db()

        # Clean up redundant users and preferences
        logger.info("Cleaning up user database...")
        self.clean_users_db()

        # Check Ollama server and model availability
        self.ollama_available = self._check_ollama_server()

        # Initialize speech recognition
        logger.info("Setting up speech components...")
        try:
            self.recognizer = sr.Recognizer()
            logger.debug("Speech recognizer initialized")
        except Exception as e:
            logger.error(f"Error setting up speech recognition: {e}")
            self.recognizer = None

        # Initialize Pygame for audio playback
        logger.info("Initializing Pygame mixer...")
        pygame.mixer.init()

        # Audio processing parameters
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.recording_thread = None
        self.recording_lock = threading.Lock()
        self.stop_event = threading.Event()

        # Current user
        self.current_user_id = None
        self.current_user_name = "Guest"

        # Video capture for face detection
        self.cap = None
        self.video_thread = None
        self.video_running = False
        self.current_frame = None

        # Initialize sentence transformer for embeddings
        logger.info("Loading sentence transformer model...")
        try:
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
            logger.debug("Sentence transformer loaded")
        except Exception as e:
            logger.error(f"Failed to load sentence transformer: {e}")
            sys.exit(1)

        # Setup the UI
        logger.info("Setting up user interface...")
        self._setup_ui()

        # Get users from database for dropdown
        logger.info("Loading users from database...")
        self._load_users()

        # Start video capture
        logger.info("Starting video capture...")
        self._start_video_capture()

        logger.info("LlamaAssistant initialization complete!")

    def init_user_db(self):
        """Initialize SQLite database for user profiles"""
        logger.debug("Initializing user database...")
        self.user_conn = sqlite3.connect('data/users.db', check_same_thread=False)
        cursor = self.user_conn.cursor()
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            name TEXT,
            preferences TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        self.user_conn.commit()
        logger.debug("User database initialized")

    def init_vector_db(self):
        """Initialize Qdrant vector database for interaction themes"""
        logger.debug("Initializing Qdrant vector database...")
        try:
            self.qdrant_client = QdrantClient(path="data/qdrant_db")
            collections = self.qdrant_client.get_collections().collections
            collection_names = [c.name for c in collections]
            if 'interactions' not in collection_names:
                self.qdrant_client.create_collection(
                    collection_name="interactions",
                    vectors_config=VectorParams(size=384, distance=Distance.COSINE)
                )
            logger.debug("Qdrant vector database initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant: {e}")
            sys.exit(1)

    def store_interaction(self, user_id, query, themes):
        """Store interaction themes in Qdrant with embeddings"""
        logger.debug(f"Storing interaction for user_id {user_id}: {query[:50]}...")
        try:
            # Generate embedding for the query
            embedding = self.embedder.encode(query).tolist()
            point_id = str(uuid.uuid4())
            payload = {
                "user_id": user_id,
                "query": query,
                "themes": themes,
                "timestamp": time.time()
            }
            point = PointStruct(id=point_id, vector=embedding, payload=payload)
            self.qdrant_client.upsert(
                collection_name="interactions",
                points=[point]
            )
            logger.info(f"Stored interaction in Qdrant: {point_id}")
        except Exception as e:
            logger.error(f"Failed to store interaction in Qdrant: {e}")

    def retrieve_relevant_interactions(self, query, user_id, limit=3):
        """Retrieve relevant past interactions from Qdrant"""
        logger.debug(f"Retrieving relevant interactions for query: {query[:50]}...")
        try:
            embedding = self.embedder.encode(query).tolist()
            search_result = self.qdrant_client.search(
                collection_name="interactions",
                query_vector=embedding,
                query_filter={"must": [{"key": "user_id", "match": {"value": user_id}}]},
                limit=limit,
                with_payload=True
            )
            interactions = [
                {
                    "query": hit.payload["query"],
                    "themes": hit.payload["themes"],
                    "timestamp": hit.payload["timestamp"]
                }
                for hit in search_result
            ]
            logger.debug(f"Retrieved {len(interactions)} relevant interactions")
            return interactions
        except Exception as e:
            logger.error(f"Failed to retrieve interactions: {e}")
            return []

    def _check_ollama_server(self):
        """Check if Ollama server is running and models are available"""
        logger.debug("Checking Ollama server and model availability...")
        try:
            response = requests.get('http://localhost:11434', timeout=5)
            if response.status_code != 200:
                logger.error(f"Ollama server returned status code: {response.status_code}")
                return False

            models_response = ollama.list()
            models = []
            for model in models_response.get('models', []):
                model_name = model.get('name') or model.get('model') or None
                if model_name:
                    models.append(model_name)
                else:
                    logger.warning(f"Model entry missing name: {model}")
            logger.debug(f"Available models: {models}")

            if not models:
                logger.error("No models found in Ollama response")
                return False

            if 'llama3.2:3b' not in models:
                logger.error("llama3.2:3b model not found in Ollama")
                return False

            logger.info("Ollama server and required models are available")
            return True
        except requests.ConnectionError:
            logger.error("Failed to connect to Ollama server at http://localhost:11434")
            return False
        except Exception as e:
            logger.error(f"Error checking Ollama server: {e}\n{traceback.format_exc()}")
            return False

    def clean_users_db(self):
        """Consolidate redundant users and clean preferences"""
        logger.info("Starting database cleanup...")
        cursor = self.user_conn.cursor()

        # Get all users
        cursor.execute("SELECT user_id, name, preferences FROM users")
        users = cursor.fetchall()

        if not users:
            logger.info("No users found in the database.")
            return

        # Group users by name
        user_groups = {}
        for user_id, name, preferences in users:
            if name not in user_groups:
                user_groups[name] = []
            user_groups[name].append({"user_id": user_id, "preferences": preferences})

        # Consolidate preferences for each name
        for name, user_list in user_groups.items():
            if len(user_list) == 1:
                logger.info(f"No duplicates for user '{name}', skipping consolidation")
                # Still clean preferences for single users
                user = user_list[0]
                try:
                    prefs = json.loads(user["preferences"]) if user["preferences"] else {}
                    if "other_nouns" in prefs:
                        del prefs["other_nouns"]
                        updated_prefs_json = json.dumps(prefs)
                        cursor.execute(
                            "UPDATE users SET preferences = ? WHERE user_id = ?",
                            (updated_prefs_json, user["user_id"])
                        )
                        logger.info(f"Removed 'other_nouns' for user '{name}' (user_id: {user['user_id']}): {prefs}")
                except json.JSONDecodeError:
                    logger.warning(f"Invalid preferences JSON for user_id {user['user_id']}")
                continue

            logger.info(f"Consolidating {len(user_list)} entries for user '{name}'...")
            # Keep the first user_id with non-empty preferences, or the first one
            main_user = None
            for user in user_list:
                try:
                    prefs = json.loads(user["preferences"]) if user["preferences"] else {}
                    if prefs:
                        main_user = user
                        break
                except json.JSONDecodeError:
                    logger.warning(f"Invalid preferences JSON for user_id {user['user_id']}")
            if not main_user:
                main_user = user_list[0]

            # Merge preferences
            merged_prefs = {}
            for user in user_list:
                try:
                    prefs = json.loads(user["preferences"]) if user["preferences"] else {}
                    for category, items in prefs.items():
                        if category == "other_nouns":
                            continue  # Skip other_nouns
                        if category not in merged_prefs:
                            merged_prefs[category] = []
                        for item in items:
                            if item not in merged_prefs[category]:
                                merged_prefs[category].append(item)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid preferences JSON for user_id {user['user_id']}")

            # Update main user
            main_user_id = main_user["user_id"]
            updated_prefs_json = json.dumps(merged_prefs)
            cursor.execute(
                "UPDATE users SET preferences = ? WHERE user_id = ?",
                (updated_prefs_json, main_user_id)
            )
            logger.info(f"Updated preferences for user '{name}' (user_id: {main_user_id}): {merged_prefs}")

            # Delete redundant users
            for user in user_list:
                if user["user_id"] != main_user_id:
                    cursor.execute("DELETE FROM users WHERE user_id = ?", (user["user_id"],))
                    logger.info(f"Deleted redundant user_id: {user['user_id']} for name '{name}'")

        self.user_conn.commit()
        logger.info("Database cleanup completed")

    def get_user(self, user_id):
        """Get user data from SQLite database"""
        logger.debug(f"Retrieving user data for user_id: {user_id}")
        cursor = self.user_conn.cursor()
        cursor.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
        user = cursor.fetchone()

        if user:
            logger.debug(f"User found: {user[1]}")
            return {
                'user_id': user[0],
                'name': user[1],
                'preferences': user[2],
                'created_at': user[3]
            }
        logger.debug("No user found")
        return None

    def get_user_by_name(self, name):
        """Get user data by name from SQLite database"""
        logger.debug(f"Retrieving user data for name: {name}")
        cursor = self.user_conn.cursor()
        cursor.execute("SELECT * FROM users WHERE name = ?", (name,))
        user = cursor.fetchone()

        if user:
            logger.debug(f"User found: {user[1]}")
            return {
                'user_id': user[0],
                'name': user[1],
                'preferences': user[2],
                'created_at': user[3]
            }
        logger.debug(f"No user found with name: {name}")
        return None

    def create_user(self, name, preferences=None):
        """Create a new user in the database or return existing user_id"""
        logger.debug(f"Attempting to create or find user: {name}")
        existing_user = self.get_user_by_name(name)
        if existing_user:
            logger.info(f"User '{name}' already exists with user_id: {existing_user['user_id']}")
            return existing_user['user_id']

        user_id = str(uuid.uuid4())
        preferences = preferences or "{}"
        cursor = self.user_conn.cursor()
        cursor.execute(
            "INSERT INTO users (user_id, name, preferences) VALUES (?, ?, ?)",
            (user_id, name, preferences)
        )
        self.user_conn.commit()
        logger.info(f"Created new user '{name}' with user_id: {user_id}")
        return user_id

    def update_user_preferences(self, user_id, new_preferences):
        """Update user preferences in database"""
        logger.debug(f"Updating preferences for user_id: {user_id} with: {new_preferences}")
        if not user_id:
            logger.warning("No user_id provided, skipping preferences update")
            return False

        cursor = self.user_conn.cursor()
        cursor.execute("SELECT preferences FROM users WHERE user_id = ?", (user_id,))
        result = cursor.fetchone()

        if not result:
            logger.warning(f"No user found with user_id: {user_id}")
            return False

        try:
            current_prefs = json.loads(result[0]) if result[0] else {}
            if not isinstance(current_prefs, dict):
                logger.warning(f"Existing preferences not a dict, resetting: {current_prefs}")
                current_prefs = {}

            # Merge new preferences, avoiding duplicates
            for category, items in new_preferences.items():
                if category not in current_prefs:
                    current_prefs[category] = []
                for item in items:
                    if item not in current_prefs[category]:
                        current_prefs[category].append(item)

            updated_prefs_json = json.dumps(current_prefs)
            cursor.execute(
                "UPDATE users SET preferences = ? WHERE user_id = ?",
                (updated_prefs_json, user_id)
            )
            self.user_conn.commit()
            logger.info(f"Preferences updated successfully for user_id: {user_id}")
            return True
        except Exception as e:
            logger.error(f"Error updating preferences for user_id {user_id}: {e}")
            return False

    def detect_emotion_from_face(self, frame):
        """Detect emotion from facial expressions using DeepFace"""
        logger.debug("Detecting emotion from face...")
        if frame is None:
            logger.debug("No frame provided")
            return "unknown"

        try:
            # Convert BGR (OpenCV) to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Analyze emotions using DeepFace
            result = DeepFace.analyze(
                img_path=rgb_frame,
                actions=['emotion'],
                enforce_detection=False,
                detector_backend='opencv',
                silent=True
            )
            if not result or not isinstance(result, list) or len(result) == 0:
                logger.debug("No faces detected in frame")
                return "no_face_detected"

            # Get dominant emotion
            dominant_emotion = result[0]['dominant_emotion']
            emotion_scores = result[0]['emotion']
            confidence = emotion_scores[dominant_emotion]
            logger.debug(f"Dominant emotion: {dominant_emotion} (confidence: {confidence:.2f})")

            if confidence < 30:  # Threshold to avoid low-confidence predictions
                logger.debug("Emotion confidence too low")
                return "unknown"

            return dominant_emotion
        except Exception as e:
            logger.error(f"Error detecting emotion with DeepFace: {e}")
            return "unknown"

    def start_voice_recording(self):
        """Start recording audio for voice input using PyAudio"""
        logger.debug("Starting voice recording...")
        with self.recording_lock:
            if self.is_recording:
                logger.debug("Already recording")
                return False

        self.audio_queue = queue.Queue()
        self.stop_event.clear()
        p = pyaudio.PyAudio()
        format = pyaudio.paInt16
        channels = 1
        rate = 16000
        chunk = 1024

        # List available input devices
        try:
            device_index = None
            for i in range(p.get_device_count()):
                dev = p.get_device_info_by_index(i)
                if dev['maxInputChannels'] > 0 and 'Logi USB Headset H340' in dev['name']:
                    device_index = i
                    break
            if device_index is None:
                device_index = p.get_default_input_device_info()['index']
            logger.debug(f"Using input device index: {device_index}")
        except Exception as e:
            logger.error(f"Error accessing audio devices: {e}")
            p.terminate()
            self.root.after(0, lambda: messagebox.showerror(
                "Voice Input Error",
                "No microphone detected. Please connect a microphone and try again."
            ))
            return False

        def record_audio():
            try:
                stream = p.open(
                    format=format,
                    channels=channels,
                    rate=rate,
                    input=True,
                    input_device_index=device_index,
                    frames_per_buffer=chunk
                )
                with self.recording_lock:
                    self.is_recording = True
                logger.debug("Audio stream started")

                frames = []
                timeout = 10  # seconds
                start_time = time.time()

                while not self.stop_event.is_set() and (time.time() - start_time) < timeout:
                    try:
                        data = stream.read(chunk, exception_on_overflow=False)
                        frames.append(np.frombuffer(data, dtype=np.int16))
                    except Exception as e:
                        logger.error(f"Error reading audio stream: {e}")
                        break

                with self.recording_lock:
                    self.is_recording = False
                stream.stop_stream()
                stream.close()
                p.terminate()

                if frames:
                    self.audio_queue.put(np.concatenate(frames))
                    logger.debug("Audio frames recorded")
                else:
                    logger.warning("No audio frames recorded")
            except Exception as e:
                logger.error(f"Error in audio recording: {e}")
                with self.recording_lock:
                    self.is_recording = False
                self.root.after(0, lambda: messagebox.showerror(
                    "Voice Input Error",
                    f"Failed to record audio: {str(e)}"
                ))

        self.recording_thread = threading.Thread(target=record_audio)
        self.recording_thread.daemon = True
        self.recording_thread.start()
        time.sleep(0.2)  # Ensure thread starts
        with self.recording_lock:
            if self.is_recording:
                logger.info("Voice recording started successfully")
                return True
            else:
                logger.warning("Failed to start voice recording")
                return False

    def stop_voice_recording(self):
        """Stop recording and process the audio"""
        logger.debug("Stopping voice recording...")
        with self.recording_lock:
            if not self.is_recording:
                logger.debug("Not recording")
                return None
            self.is_recording = False

        self.stop_event.set()
        if self.recording_thread:
            self.recording_thread.join(timeout=1.0)
            self.recording_thread = None

        audio_data = None
        while not self.audio_queue.empty():
            audio_data = self.audio_queue.get()

        logger.debug(f"Collected audio data: {len(audio_data) if audio_data is not None else 0} samples")
        if audio_data is None:
            logger.warning("No audio data recorded")
            return None

        temp_audio_file = "temp_recording.wav"
        try:
            wf = wave.open(temp_audio_file, 'wb')
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(16000)
            wf.writeframes(audio_data.tobytes())
            wf.close()
            logger.debug(f"Saved audio to {temp_audio_file}")
        except Exception as e:
            logger.error(f"Error saving audio file: {e}")
            return None

        if self.recognizer:
            try:
                with sr.AudioFile(temp_audio_file) as source:
                    audio = self.recognizer.record(source)
                    text = self.recognizer.recognize_google(audio)
                    logger.info(f"Transcribed audio: {text}")
                    return text
            except sr.UnknownValueError:
                logger.error("Speech recognition could not understand audio")
                self.root.after(0, lambda: messagebox.showwarning(
                    "Voice Input Warning",
                    "Could not understand audio. Please speak clearly and try again."
                ))
            except sr.RequestError as e:
                logger.error(f"Speech recognition request failed: {e}")
                self.root.after(0, lambda: messagebox.showerror(
                    "Voice Input Error",
                    "Failed to process voice input. Please check your internet connection."
                ))
            except Exception as e:
                logger.error(f"Error transcribing audio: {e}")
                self.root.after(0, lambda: messagebox.showerror(
                    "Voice Input Error",
                    f"Error processing voice input: {str(e)}"
                ))
            finally:
                try:
                    if os.path.exists(temp_audio_file):
                        os.remove(temp_audio_file)
                        logger.debug(f"Deleted temporary audio file: {temp_audio_file}")
                except Exception as e:
                    logger.warning(f"Failed to delete temporary audio file: {e}")

        logger.debug("No transcription available")
        return None

    def text_to_speech(self, text):
        """Convert response text to speech with caching"""
        logger.debug(f"Generating speech for text: {text[:50]}...")
        try:
            text_hash = hashlib.md5(text.encode()).hexdigest()
            audio_file = f"data/response_audio_{text_hash}.mp3"
            
            if os.path.exists(audio_file):
                logger.debug(f"Using cached audio file: {audio_file}")
                return audio_file

            tts = gTTS(text=text, lang='en')
            tts.save(audio_file)
            logger.debug(f"Speech saved to {audio_file}")
            return audio_file
        except Exception as e:
            logger.error(f"Error generating speech: {e}")
            return None

    def extract_preferences_from_text(self, text):
        """Extract preferences (places, likes, dislikes, also_talks_on) from text"""
        logger.debug(f"Extracting preferences from text: {text[:50]}...")
        preferences = {
            "places": [],
            "likes": [],
            "dislikes": [],
            "also_talks_on": []
        }
        text_lower = text.lower()
        doc = nlp(text)

        # Extract named entities (places)
        for ent in doc.ents:
            if ent.label_ in ["GPE", "LOC"]:
                if ent.text.lower() not in preferences["places"]:
                    preferences["places"].append(ent.text.lower())
                    logger.debug(f"Extracted place: {ent.text.lower()}")

        # Extract topics for also_talks_on (noun chunks and specific entities)
        generic_terms = {"place", "places", "thing", "things", "good"}
        for chunk in doc.noun_chunks:
            topic = chunk.text.lower()
            if topic not in generic_terms and len(topic) > 2 and topic not in preferences["also_talks_on"]:
                preferences["also_talks_on"].append(topic)
                logger.debug(f"Extracted also_talks_on: {topic}")

        # Extract likes with keywords
        like_keywords = ["like", "enjoy", "love", "prefer"]
        for keyword in like_keywords:
            if keyword in text_lower:
                words = text_lower.split()
                for i, word in enumerate(words):
                    if word == keyword and i < len(words) - 1:
                        like_item = " ".join(words[i+1:min(i+4, len(words))])
                        if like_item not in preferences["likes"]:
                            preferences["likes"].append(like_item)
                            logger.debug(f"Extracted like: {like_item}")
                        break
                break

        # Extract dislikes with keywords
        dislike_keywords = ["hate", "dislike"]
        for keyword in dislike_keywords:
            if keyword in text_lower or "don't like" in text_lower:
                words = text_lower.split()
                for i, word in enumerate(words):
                    if (word == keyword) or (word == "like" and i > 0 and words[i-1] == "don't"):
                        dislike_item = " ".join(words[i+1:min(i+4, len(words))])
                        if dislike_item not in preferences["dislikes"]:
                            preferences["dislikes"].append(dislike_item)
                            logger.debug(f"Extracted dislike: {dislike_item}")
                        break
                break

        # Remove empty categories
        preferences = {k: v for k, v in preferences.items() if v}
        if preferences:
            logger.debug(f"Extracted preferences: {preferences}")
        else:
            logger.debug("No preferences extracted")
        return preferences

    def extract_themes(self, text):
        """Extract themes from text for vector storage"""
        logger.debug(f"Extracting themes from text: {text[:50]}...")
        doc = nlp(text)
        themes = []
        # Extract entities and noun chunks
        for ent in doc.ents:
            if ent.label_ in ["GPE", "LOC", "PERSON", "ORG", "EVENT"]:
                themes.append(ent.text.lower())
        for chunk in doc.noun_chunks:
            topic = chunk.text.lower()
            if len(topic) > 2 and topic not in themes:
                themes.append(topic)
        logger.debug(f"Extracted themes: {themes}")
        return themes

    def process_query(self, user_id, query, face_frame=None):
        """Process a user query with face-based mood detection and vector context"""
        logger.info(f"Processing query for user_id {user_id}: {query[:50]}...")
        user_data = self.get_user(user_id)
        user_preferences = "{}"
        if user_data:
            user_preferences = user_data['preferences']
            try:
                user_preferences = json.loads(user_preferences)
                logger.debug(f"User preferences: {user_preferences}")
            except Exception as e:
                logger.error(f"Error parsing user preferences: {e}")
                user_preferences = {}

        # Retrieve relevant past interactions
        past_interactions = self.retrieve_relevant_interactions(query, user_id)
        context = ""
        if past_interactions:
            context = "Previous interactions:\n"
            for interaction in past_interactions:
                context += f"- Query: {interaction['query']}, Themes: {', '.join(interaction['themes'])}\n"
            logger.debug(f"Added context from past interactions: {context[:100]}...")

        face_emotion = self.detect_emotion_from_face(face_frame) if face_frame is not None else "unknown"
        logger.debug(f"Detected face emotion: {face_emotion}")

        prompt = self._build_prompt(query, user_preferences, face_emotion, context)
        logger.debug(f"Generated prompt: {prompt[:100]}...")

        response_text = None
        if not self.ollama_available:
            logger.warning("Ollama server or models unavailable, using fallback response")
            response_text = "Sorry, I'm unable to connect to the AI model. Please ensure the Ollama server is running with 'llama3.2:3b' model installed, then try again."
        else:
            for attempt in range(3):
                try:
                    logger.debug(f"Attempting to generate response (attempt {attempt + 1})...")
                    response = ollama.generate(model='llama3.2:3b', prompt=prompt, options={'max_tokens': 500, 'temperature': 0.7})
                    response_text = response['response'].strip()
                    logger.info(f"Response generated successfully: {response_text[:50]}...")
                    break
                except Exception as e:
                    logger.error(f"Error generating response (attempt {attempt + 1}): {e}\n{traceback.format_exc()}")
                    if attempt < 2:
                        logger.debug("Retrying...")
                        time.sleep(1)
                    else:
                        logger.error("All attempts failed")
                        response_text = "I encountered an issue while processing your request. Please try again or contact support."

        # Extract and store preferences
        new_preferences = self.extract_preferences_from_text(query)
        if new_preferences and user_id and self.current_user_name != "Guest":
            if self.update_user_preferences(user_id, new_preferences):
                logger.info(f"Updated preferences for user_id {user_id}: {new_preferences}")
            else:
                logger.warning(f"Failed to update preferences for user_id {user_id}")

        # Extract and store interaction themes
        themes = self.extract_themes(query)
        if themes and user_id:
            self.store_interaction(user_id, query, themes)

        audio_file = self.text_to_speech(response_text)

        logger.info("Query processing complete")
        return {
            'text': response_text,
            'audio_file': audio_file,
            'detected_emotion': {
                'face': face_emotion
            }
        }

    def _build_prompt(self, query, user_preferences, face_emotion, context):
        """Build a prompt for the LLM with vector context"""
        logger.debug("Building prompt...")
        prompt = "You are a helpful AI assistant. "

        if context:
            prompt += f"{context}\n"

        if user_preferences:
            prompt += f"The user has the following preferences: {user_preferences}. "

        if face_emotion != "unknown":
            prompt += f"The user's facial expression indicates they're {face_emotion}. "

        prompt += f"Now, please respond to the following: {query}"
        logger.debug(f"Prompt built: {prompt[:100]}...")
        return prompt

    def _setup_ui(self):
        """Set up the user interface"""
        logger.debug("Setting up UI...")
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        top_frame = ttk.Frame(main_frame)
        top_frame.pack(fill=tk.X, pady=5)

        ttk.Label(top_frame, text="User:").pack(side=tk.LEFT, padx=5)
        self.user_var = tk.StringVar()
        self.user_dropdown = ttk.Combobox(top_frame, textvariable=self.user_var, state="readonly", width=20)
        self.user_dropdown.pack(side=tk.LEFT, padx=5)
        self.user_dropdown.bind("<<ComboboxSelected>>", self._on_user_selected)

        ttk.Button(top_frame, text="Add User", command=self._show_add_user_dialog).pack(side=tk.LEFT, padx=5)

        ttk.Label(top_frame, text="Detected Mood:").pack(side=tk.LEFT, padx=(20, 5))
        self.mood_var = tk.StringVar(value="Unknown")
        ttk.Label(top_frame, textvariable=self.mood_var).pack(side=tk.LEFT, padx=5)

        middle_frame = ttk.Frame(main_frame)
        middle_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        middle_frame.columnconfigure(0, weight=3)
        middle_frame.columnconfigure(1, weight=1)

        chat_frame = ttk.LabelFrame(middle_frame, text="Conversation")
        chat_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5))

        self.chat_display = scrolledtext.ScrolledText(chat_frame, wrap=tk.WORD, state=tk.DISABLED)
        self.chat_display.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        video_frame = ttk.LabelFrame(middle_frame, text="Face Detection")
        video_frame.grid(row=0, column=1, sticky="nsew")

        self.video_label = ttk.Label(video_frame)
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        bottom_frame = ttk.Frame(main_frame)
        bottom_frame.pack(fill=tk.X, pady=5)

        self.input_box = scrolledtext.ScrolledText(bottom_frame, height=3, wrap=tk.WORD)
        self.input_box.pack(fill=tk.X, expand=True, side=tk.LEFT, padx=(0, 5))
        self.input_box.bind("<Return>", self._on_enter_pressed)

        self.voice_button_var = tk.StringVar(value="🎤 Voice")
        self.voice_button = ttk.Button(
            bottom_frame,
            textvariable=self.voice_button_var,
            command=self._toggle_voice_input
        )
        self.voice_button.pack(side=tk.LEFT, padx=5)
        self.is_recording = False

        ttk.Button(bottom_frame, text="Send", command=self._process_input).pack(side=tk.LEFT, padx=5)

        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        logger.debug("UI setup complete")

    def _load_users(self):
        """Load users from database to dropdown"""
        logger.debug("Loading users...")
        try:
            cursor = self.user_conn.cursor()
            cursor.execute("SELECT user_id, name FROM users")
            users = cursor.fetchall()

            user_dict = {"Guest": None}
            for user_id, name in users:
                user_dict[name] = user_id

            self.user_dropdown['values'] = list(user_dict.keys())
            self.user_dropdown.current(0)
            self.user_dict = user_dict
            logger.info("Users loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load users: {e}")
            messagebox.showerror("Error", f"Failed to load users: {e}")

    def _on_user_selected(self, event):
        """Handle user selection from dropdown"""
        selected_name = self.user_var.get()
        self.current_user_id = self.user_dict.get(selected_name)
        self.current_user_name = selected_name
        logger.info(f"Switched to user: {selected_name}")
        self._update_chat("System", f"Switched to user: {selected_name}")

    def _show_add_user_dialog(self):
        """Show dialog to add a new user or switch to existing"""
        logger.debug("Showing add user dialog...")
        dialog = tk.Toplevel(self.root)
        dialog.title("Add New User")
        dialog.geometry("300x150")
        dialog.transient(self.root)
        dialog.grab_set()

        ttk.Label(dialog, text="Name:").pack(pady=(10, 0))
        name_entry = ttk.Entry(dialog, width=30)
        name_entry.pack(pady=5)
        name_entry.focus_set()

        def add_user():
            name = name_entry.get().strip()
            if not name:
                logger.warning("Attempted to create user with empty name")
                messagebox.showerror("Error", "Name cannot be empty")
                return

            try:
                existing_user = self.get_user_by_name(name)
                if existing_user:
                    logger.info(f"User '{name}' already exists, switching to it")
                    dialog.destroy()
                    self.user_var.set(name)
                    self._on_user_selected(None)
                    messagebox.showinfo("Info", f"User '{name}' already exists. Switched to existing user.")
                    return

                user_id = self.create_user(name)
                dialog.destroy()
                self._load_users()
                self.user_var.set(name)
                self._on_user_selected(None)
                logger.info(f"User '{name}' created successfully")
                messagebox.showinfo("Success", f"User '{name}' created successfully")
            except Exception as e:
                logger.error(f"Failed to create or switch user: {e}")
                messagebox.showerror("Error", f"Failed to process user: {e}")

        ttk.Button(dialog, text="Add", command=add_user).pack(pady=10)
        ttk.Button(dialog, text="Cancel", command=dialog.destroy).pack()

    def _start_video_capture(self):
        """Start the video capture in a separate thread"""
        logger.debug("Starting video capture...")
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                logger.warning("Could not open webcam")
                self.root.after(0, lambda: messagebox.showwarning(
                    "Warning",
                    "Could not open webcam. Face detection will be disabled."
                ))
                self.cap = None
                return

            self.video_running = True
            self.video_thread = threading.Thread(target=self._update_video)
            self.video_thread.daemon = True
            self.video_thread.start()
            logger.info("Video capture started")
        except Exception as e:
            logger.error(f"Error starting video: {e}")
            self.root.after(0, lambda: messagebox.showwarning(
                "Warning",
                f"Error starting video: {e}. Face detection will be disabled."
            ))
            self.cap = None

    def _stop_video_capture(self):
        """Stop the video capture"""
        logger.debug("Stopping video capture...")
        self.video_running = False
        if self.video_thread:
            self.video_thread.join(timeout=1.0)
            self.video_thread = None
        if self.cap and self.cap.isOpened():
            self.cap.release()
            self.cap = None
        logger.debug("Video capture stopped")

    def _update_video(self):
        """Update the video frame for face detection"""
        logger.debug("Starting video update loop...")
        try:
            while self.video_running and self.cap:
                ret, frame = self.cap.read()
                if ret:
                    display_frame = cv2.resize(frame, (320, 240))
                    self.current_frame = frame.copy()
                    display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                    image = PIL.Image.fromarray(display_frame)
                    photo = PIL.ImageTk.PhotoImage(image=image)
                    self.video_label.configure(image=photo)
                    self.video_label.image = photo
                else:
                    logger.warning("Failed to capture video frame")
                    break
                time.sleep(0.03)
        except Exception as e:
            logger.error(f"Error in video thread: {e}")
        finally:
            logger.debug("Video update loop stopped")

    def _toggle_voice_input(self):
        """Toggle voice input recording"""
        logger.debug("Toggling voice input...")
        with self.recording_lock:
            if self.is_recording:
                self.is_recording = False
                self.voice_button_var.set("🎤 Voice")
                self.status_var.set("Processing voice input...")
                self.root.update_idletasks()

                text = self.stop_voice_recording()
                if text:
                    self.input_box.delete(1.0, tk.END)
                    self.input_box.insert(tk.END, text)
                    self.status_var.set("Voice input: " + text)
                    logger.info(f"Voice input processed: {text}")
                else:
                    self.status_var.set("Voice input failed")
                    logger.warning("Voice input failed")
            else:
                self.status_var.set("Starting voice recording...")
                self.root.update_idletasks()
                success = self.start_voice_recording()
                if success:
                    self.voice_button_var.set("⏹ Stop")
                    self.status_var.set("Recording voice input...")
                    logger.info("Started voice recording")
                else:
                    self.status_var.set("Failed to start voice recording")
                    logger.warning("Failed to start voice recording")
                    self.root.after(0, lambda: messagebox.showerror(
                        "Voice Input Error",
                        "Failed to start recording. Please check your microphone and try again."
                    ))

    def _on_enter_pressed(self, event):
        """Handle Enter key in input box"""
        logger.debug("Enter key pressed in input box")
        if event.state & 0x1:
            return
        self._process_input()
        return "break"

    def _process_input(self):
        """Process the user input"""
        query = self.input_box.get(1.0, tk.END).strip()
        if not query:
            logger.debug("Empty input, ignoring")
            return

        self.input_box.delete(1.0, tk.END)
        self._update_chat(self.current_user_name, query)
        logger.info(f"Processing user input: {query[:50]}...")
        threading.Thread(target=self._process_query_thread, args=(query,)).start()

    def _process_query_thread(self, query):
        """Process query in a background thread"""
        logger.debug("Starting query processing thread...")
        self.status_var.set("Processing...")
        self.root.update_idletasks()

        try:
            frame = self.current_frame
            self._stop_video_capture()
            response = self.process_query(self.current_user_id, query, frame)

            mood_text = f"Face: {response['detected_emotion']['face']}"
            self.mood_var.set(mood_text)
            self._update_chat("Assistant", response['text'])
            logger.info(f"Assistant response: {response['text'][:50]}...")

            if response['audio_file'] and os.path.exists(response['audio_file']):
                try:
                    pygame.mixer.music.load(response['audio_file'])
                    pygame.mixer.music.play()
                    logger.debug("Playing audio response")
                except Exception as e:
                    logger.error(f"Error playing audio: {e}")

            self.status_var.set("Ready")
            logger.debug("Query processing thread completed")
        except Exception as e:
            logger.error(f"Error in query processing thread: {e}\n{traceback.format_exc()}")
            self._update_chat("System", f"Error: {str(e)}")
            self.status_var.set("Error occurred")
        finally:
            self._start_video_capture()

    def _update_chat(self, sender, message):
        """Update the chat display with a new message"""
        logger.debug(f"Updating chat - Sender: {sender}, Message: {message[:50]}...")
        self.chat_display.config(state=tk.NORMAL)
        timestamp = time.strftime("%H:%M:%S")
        formatted_msg = f"[{timestamp}] {sender}:\n{message}\n\n"

        if sender == "Assistant":
            self.chat_display.insert(tk.END, formatted_msg, "assistant_msg")
            self.chat_display.tag_config("assistant_msg", foreground="blue")
        elif sender == "System":
            self.chat_display.insert(tk.END, formatted_msg, "system_msg")
            self.chat_display.tag_config("system_msg", foreground="red")
        else:
            self.chat_display.insert(tk.END, formatted_msg, "user_msg")
            self.chat_display.tag_config("user_msg", foreground="green")

        self.chat_display.see(tk.END)
        self.chat_display.config(state=tk.DISABLED)
        logger.debug("Chat updated")

    def cleanup(self):
        """Clean up resources when shutting down"""
        logger.info("Cleaning up resources...")
        self._stop_video_capture()
        with self.recording_lock:
            if self.is_recording:
                self.is_recording = False
                self.stop_event.set()
                if self.recording_thread:
                    self.recording_thread.join(timeout=1.0)
        if self.user_conn:
            self.user_conn.close()
            logger.debug("Closed user database connection")
        if hasattr(self, 'qdrant_client'):
            self.qdrant_client.close()
            logger.debug("Closed Qdrant connection")
        # Clean up temporary audio files
        for file in os.listdir("data"):
            if file.startswith("response_audio_") or file == "temp_recording.wav":
                try:
                    os.remove(os.path.join("data", file))
                    logger.debug(f"Deleted temporary file: {file}")
                except Exception as e:
                    logger.warning(f"Failed to delete temporary file {file}: {e}")

    def run(self):
        """Run the main application loop"""
        logger.info("Starting main application loop...")
        try:
            self.root.mainloop()
        finally:
            self.cleanup()
            logger.info("Application shutdown complete")


if __name__ == "__main__":
    try:
        logger.info("Starting application...")
        root = tk.Tk()
        style = ttk.Style()
        style.theme_use('clam')
        app = LlamaAssistantGUI(root)
        app.run()
    except Exception as e:
        logger.error(f"Fatal error: {e}\n{traceback.format_exc()}")
        if 'root' in locals():
            messagebox.showerror("Error", f"Application error: {str(e)}")
        else:
            print(f"Fatal error: {str(e)}")
        sys.exit(1)