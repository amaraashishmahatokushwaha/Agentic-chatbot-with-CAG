import os
import sys
import logging
import requests
import spacy
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
import traceback
import json
import time
from datetime import datetime
import ollama
from main import LlamaAssistantGUI
import PyPDF2
from docx import Document
import textract
import pickle
import hashlib
import threading
import speech_recognition as sr
from gtts import gTTS
import pygame
import tempfile
import queue
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchValue

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# List of known cities for fallback
KNOWN_CITIES = {
    "bengaluru": "Bengaluru",
    "bangalore": "Bengaluru",
    "mumbai": "Mumbai",
    "delhi": "Delhi",
    "chennai": "Chennai",
    "kolkata": "Kolkata"
}

# Known institutions for location extraction
KNOWN_INSTITUTIONS = {
    "iast software solutions": "IAST Software Solutions, Bengaluru",
    "bmsit & m college": "BMSIT & M College, Bengaluru",
    "bmsit": "BMSIT & M College, Bengaluru"
}

class WeatherMapAssistantGUI(LlamaAssistantGUI):
    def __init__(self, root):
        """Initialize the WeatherMapAssistantGUI with weather, map, CAG, and voice configuration"""
        logger.debug("Initializing WeatherMapAssistantGUI...")
        super().__init__(root)
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("Successfully loaded spaCy model: en_core_web_sm")
        except Exception as e:
            logger.error(f"Failed to load spaCy model: {e}")
            raise

        # Weather API configuration
        self.weather_api_key = "e719c82a8155a5b93b0b386b1ccc4e5e"
        self.weather_api_url = "http://api.openweathermap.org/data/2.5/weather"
        self.onecall_api_url = "https://api.openweathermap.org/data/3.0/onecall"
        logger.info("Weather API configured")

        # OpenRouteService API for routing and geocoding
        self.ors_api_key = "5b3ce3597851110001cf6248a5f5c2b25ac5416f9cb56f877a261e3f"
        self.ors_api_url = "https://api.openrouteservice.org/v2/directions"
        self.ors_geocode_url = "https://api.openrouteservice.org/geocode/search"
        logger.info("OpenRouteService API configured for routing and geocoding")

        # CAG configuration
        self.kv_cache_dir = "data/kv_cache"  # Directory to store KV Cache
        self.max_tokens = 8000  # Llama 3.2:3b context window (~8k tokens)
        self.chunk_timeout = 30  # Configurable timeout for chunk processing (seconds)
        self.kv_cache = {}  # In-memory KV Cache for current session
        self.document_chunks = {}  # Store document chunks per user for CAG
        self.query_tokens = {}  # Track query tokens for cache reset
        os.makedirs(self.kv_cache_dir, exist_ok=True)
        self.executor = ThreadPoolExecutor(max_workers=4)  # Increased workers for parallel processing
        logger.info("CAG configured with KV Cache directory")

        # Voice configuration
        self.voice_enabled = True
        self.listening = False
        self.audio_queue = queue.Queue()
        self.query_lock = threading.Lock()
        self.last_response = ""
        self.is_speaking = False
        self.last_input_time = 0
        self.debounce_interval = 0.5  # 500ms debounce
        try:
            pygame.mixer.init()
            logger.info("Initialized pygame mixer for voice output")
        except Exception as e:
            logger.error(f"Failed to initialize pygame mixer: {e}")
            self.voice_enabled = False
            messagebox.showwarning("Warning", "Voice output disabled due to initialization error.")

        # Check Ollama server status
        if not self.check_ollama_health():
            logger.error("Ollama server is not responsive. CAG preloading may fail.")
            messagebox.showwarning("Warning", "Ollama server connection failed. Ensure it is running with 'llama3.2:3b'.")

        # Warn about SentenceTransformer if loaded
        if hasattr(self, 'embedder') or 'sentence_transformers' in sys.modules:
            logger.warning("SentenceTransformer detected. This may indicate residual RAG code in main.py or LlamaAssistantGUI. CAG should not use SentenceTransformer. Please remove from main.py.")

        # Update the window title
        self.root.title("Llama3 Offline Assistant with Weather, Map, CAG, and Voice")
        logger.debug("WeatherMapAssistantGUI initialization complete")

    def check_ollama_health(self):
        """Check if Ollama server is responsive"""
        try:
            response = requests.get("http://127.0.0.1:11434", timeout=5)
            if response.status_code == 200:
                logger.info("Ollama server is responsive")
                return True
            else:
                logger.warning(f"Ollama server returned status code: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Failed to connect to Ollama server: {e}")
            return False

    def _setup_ui(self):
        """Set up the optimized user interface with CAG mode"""
        logger.debug("Setting up optimized UI...")
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Top frame for user selection and mood display
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

        # Button frame for document upload and voice mode
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=5)

        self.upload_button = ttk.Button(button_frame, text="Upload Document (CAG)", command=self._show_upload_dialog)
        self.upload_button.pack(side=tk.LEFT, padx=5)

        self.voice_mode_button = ttk.Button(button_frame, text="Enable Voice Mode", command=self.toggle_voice_mode)
        self.voice_mode_button.pack(side=tk.LEFT, padx=5)

        # Middle frame for chat and video
        middle_frame = ttk.Frame(main_frame)
        middle_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        middle_frame.columnconfigure(0, weight=3)
        middle_frame.columnconfigure(1, weight=1)

        chat_frame = ttk.LabelFrame(middle_frame, text="Conversation")
        chat_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5))

        self.chat_display = scrolledtext.ScrolledText(chat_frame, wrap=tk.WORD, state=tk.DISABLED, height=20)
        self.chat_display.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        video_frame = ttk.LabelFrame(middle_frame, text="Face Detection")
        video_frame.grid(row=0, column=1, sticky="nsew")

        self.video_label = ttk.Label(video_frame)
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Bottom frame for input
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

        ttk.Button(bottom_frame, text="Send", command=self._process_input).pack(side=tk.LEFT, padx=5)

        # Status bar
        self.status_var = tk.StringVar(value="Ready | Query Mode: CAG")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        logger.debug("Optimized UI setup complete")

    def _show_upload_dialog(self):
        """Show dialog to upload document for CAG processing"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Upload Document for CAG")
        dialog.geometry("300x150")
        dialog.transient(self.root)
        dialog.grab_set()

        ttk.Label(dialog, text="Select a document to preload for CAG:").pack(pady=10)

        def on_upload():
            file_path = filedialog.askopenfilename(
                filetypes=[("Documents", "*.pdf *.docx *.txt"), ("All Files", "*.*")]
            )
            if file_path:
                threading.Thread(target=self.upload_document, args=(file_path,), daemon=True).start()
                dialog.destroy()

        ttk.Button(dialog, text="Browse", command=on_upload).pack(pady=10)
        ttk.Button(dialog, text="Cancel", command=dialog.destroy).pack(pady=5)

    def _update_chat(self, sender, message, color="black"):
        """Update the chat display with a new message"""
        logger.debug(f"Updating chat display: sender={sender}, message={message[:50]}..., color={color}")
        self.chat_display.config(state=tk.NORMAL)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted_message = f"[{timestamp}] {sender}: {message}\n"
        self.chat_display.insert(tk.END, formatted_message, (sender, color))
        self.chat_display.tag_config(sender, foreground=color)
        self.chat_display.see(tk.END)
        self.chat_display.config(state=tk.DISABLED)

    def _process_input(self, event=None):
        """Process text input from the input box with debounce"""
        current_time = time.time()
        if current_time - self.last_input_time < self.debounce_interval:
            logger.debug("Input debounced, ignoring rapid successive call")
            return
        self.last_input_time = current_time

        query = self.input_box.get("1.0", tk.END).strip()
        if not query:
            return
        self.input_box.delete("1.0", tk.END)
        self._update_chat("User", query, "black")
        threading.Thread(target=self._process_query_thread, args=(query,), daemon=True).start()

    def _on_enter_pressed(self, event):
        """Handle Return key press with proper event handling"""
        self._process_input()
        return "break"  # Prevent default newline insertion

    def _process_query_thread(self, query):
        """Process query in a separate thread to avoid GUI freeze"""
        with self.query_lock:
            logger.debug(f"Acquired query lock for query: {query}")
            response = self.process_query(self.current_user_id, query)
            if response and 'text' in response:
                self.last_response = response['text']
                self.root.after(0, lambda: self._update_chat("Assistant", response['text'], "green"))
            logger.debug(f"Released query lock for query: {query}")

    def toggle_voice_mode(self):
        """Toggle continuous voice input mode"""
        if not self.voice_enabled:
            messagebox.showwarning("Warning", "Voice input is disabled due to initialization errors.")
            return

        self.listening = not self.listening
        if self.listening:
            self.voice_mode_button.config(text="Disable Voice Mode")
            threading.Thread(target=self.continuous_voice_input, daemon=True).start()
        else:
            self.voice_mode_button.config(text="Enable Voice Mode")

    def continuous_voice_input(self):
        """Handle continuous voice input with wake word 'Jarvis'"""
        recognizer = sr.Recognizer()
        recognizer.dynamic_energy_threshold = True
        recognizer.energy_threshold = 6000
        recognizer.pause_threshold = 1.0
        last_valid_input = time.time()
        timeout_seconds = 30

        while self.listening:
            try:
                with sr.Microphone() as source:
                    logger.debug("Adjusting for ambient noise...")
                    recognizer.adjust_for_ambient_noise(source, duration=1)
                    logger.debug("Listening for wake word 'Jarvis'...")
                    self.root.after(0, lambda: self._update_chat("System", "Listening for 'Jarvis'...", "blue"))
                    audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)

                logger.debug("Processing audio for wake word...")
                query = recognizer.recognize_google(audio)
                query_lower = query.lower().strip()

                if not query_lower.startswith("jarvis"):
                    logger.debug(f"No wake word 'Jarvis' detected in: {query}")
                    continue

                if pygame.mixer.get_init() and pygame.mixer.music.get_busy():
                    pygame.mixer.music.stop()
                    self.audio_queue.queue.clear()
                    logger.debug("TTS interrupted due to wake word detection")
                    time.sleep(0.1)
                    self.is_speaking = False

                query = query[6:].strip() if query_lower.startswith("jarvis") else query
                if query:
                    logger.debug(f"Initial query after wake word: {query}")
                    if len(query) >= 3 and query != self.last_response:
                        logger.info(f"Recognized voice input: {query}")
                        self.root.after(0, lambda: self._update_chat("User", query, "black"))
                        last_valid_input = time.time()
                        with self.query_lock:
                            logger.debug(f"Acquired query lock for voice query: {query}")
                            response = self.process_query(self.current_user_id, query)
                            if response and 'text' in response:
                                self.last_response = response['text']
                                self.root.after(0, lambda: self._update_chat("Assistant", response['text'], "green"))
                            logger.debug(f"Released query lock for voice query: {query}")
                    continue

                logger.debug("Listening for 2 seconds after wake word...")
                try:
                    with sr.Microphone() as source:
                        audio = recognizer.listen(source, timeout=2, phrase_time_limit=10)
                    query = recognizer.recognize_google(audio)
                    query_lower = query.lower().strip()

                    if query == self.last_response:
                        logger.debug("Query matches last response, ignoring...")
                        continue

                    logger.info(f"Recognized voice input: {query}")
                    self.root.after(0, lambda: self._update_chat("User", query, "black"))
                    last_valid_input = time.time()

                    with self.query_lock:
                        logger.debug(f"Acquired query lock for voice query: {query}")
                        response = self.process_query(self.current_user_id, query)
                        if response and 'text' in response:
                            self.last_response = response['text']
                            self.root.after(0, lambda: self._update_chat("Assistant", response['text'], "green"))
                        logger.debug(f"Released query lock for voice query: {query}")

                except sr.WaitTimeoutError:
                    logger.debug("No speech detected in 2-second window, continuing...")
                    continue

            except sr.WaitTimeoutError:
                logger.debug("No speech detected, continuing...")
                if time.time() - last_valid_input > timeout_seconds:
                    logger.debug("No valid input for 30 seconds, pausing voice mode...")
                    self.listening = False
                    self.root.after(0, lambda: self._update_chat("System", "Voice mode paused due to inactivity.", "blue"))
                    self.root.after(0, lambda: self.voice_mode_button.config(text="Enable Voice Mode"))
                    break
            except sr.UnknownValueError:
                logger.warning("Could not understand audio")
                self.root.after(0, lambda: self._update_chat("System", "Could not understand speech. Please try again.", "red"))
            except Exception as e:
                logger.error(f"Voice input error: {e}\n{traceback.format_exc()}")
                self.root.after(0, lambda: self._update_chat("System", f"Voice input error: {str(e)}", "red"))

    def text_to_speech(self, text):
        """Convert text to speech using gTTS and play asynchronously"""
        if not self.voice_enabled or not text:
            logger.debug("Voice output skipped: Disabled or no text")
            return None

        try:
            self.is_speaking = True
            tts = gTTS(text=text, lang='en')
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
            tts.save(temp_file.name)
            temp_file.close()
            logger.debug(f"Generated TTS audio file: {temp_file.name}")
            self.audio_queue.put(temp_file.name)
            self.play_audio_queue()
            return temp_file.name
        except Exception as e:
            logger.error(f"Error generating TTS: {e}\n{traceback.format_exc()}")
            self.root.after(0, lambda: messagebox.showerror("Error", f"Voice output failed: {str(e)}"))
            return None
        finally:
            self.is_speaking = False

    def play_audio_queue(self):
        """Play audio files from the queue"""
        def play_next():
            try:
                if not self.audio_queue.empty() and self.listening:
                    self.is_speaking = True
                    audio_file = self.audio_queue.get()
                    pygame.mixer.music.load(audio_file)
                    pygame.mixer.music.play()
                    while pygame.mixer.music.get_busy() and self.listening:
                        pygame.time.Clock().tick(10)
                    pygame.mixer.music.unload()
                    os.unlink(audio_file)
                    logger.debug(f"Played and removed TTS audio file: {audio_file}")
                    play_next()
            except Exception as e:
                logger.error(f"Error playing TTS audio: {e}\n{traceback.format_exc()}")
                self.root.after(0, lambda: messagebox.showerror("Error", f"Voice output failed: {str(e)}"))
            finally:
                self.is_speaking = False

        if not pygame.mixer.get_init() or not pygame.mixer.music.get_busy():
            threading.Thread(target=play_next, daemon=True).start()

    def _chunk_text(self, text, chunk_size=500, overlap=25):
        """Split text into chunks for CAG"""
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
        return chunks

    def preload_documents(self, user_id, chunks, file_name):
        """Preload document chunks into LLM and generate KV Cache for CAG"""
        logger.info(f"Preloading documents for user_id {user_id}, file: {file_name}")
        start_time = time.time()
        try:
            cache_key = hashlib.md5(file_name.encode()).hexdigest()
            cache_path = os.path.join(self.kv_cache_dir, f"{user_id}_{cache_key}.pkl")

            # Check if cache already exists
            if os.path.exists(cache_path):
                logger.info(f"KV Cache already exists for {file_name}, loading...")
                self.load_kv_cache(user_id, cache_path)
                elapsed_time = time.time() - start_time
                logger.info(f"CAG preload time: {elapsed_time:.2f} seconds")
                return True, elapsed_time

            # Estimate token count (1 word ≈ 1.33 tokens)
            total_tokens = sum(len(chunk.split()) * 1.33 for chunk in chunks)
            if total_tokens > self.max_tokens:
                logger.warning(f"Document size ({total_tokens} tokens) exceeds context window ({self.max_tokens} tokens). Truncating...")
                chunks = chunks[:int(self.max_tokens / (1.33 * 500))]

            # Generate KV Cache in parallel
            kv_cache = []
            skipped_chunks = []

            def process_chunk(i, chunk):
                token_count = len(chunk.split()) * 1.33
                logger.debug(f"Processing chunk {i+1}/{len(chunks)} ({token_count:.0f} tokens): {chunk[:100]}...")
                for attempt in range(3):  # Retry up to 3 times
                    try:
                        response = ollama.generate(
                            model='llama3.2:3b',
                            prompt=f"Store this document chunk for context:\n{chunk}",
                            options={'max_tokens': 0, 'return_kv_cache': True}
                        )
                        kv_state = response.get('kv_cache', None)
                        if kv_state is None:
                            logger.warning(f"No KV cache returned for chunk {i+1}. Using chunk text as fallback.")
                            kv_state = {'text_fallback': chunk}  # Fallback for Ollama limitation
                        return {
                            'chunk_id': i,
                            'text': chunk,
                            'file_name': file_name,
                            'kv_state': kv_state,
                            'processed_time': time.time()
                        }
                    except Exception as e:
                        logger.error(f"Error processing chunk {i+1} (attempt {attempt+1}/3): {e}\n{traceback.format_exc()}")
                        if attempt < 2:
                            time.sleep(1)
                        else:
                            logger.error(f"Skipping chunk {i+1} after {attempt+1} failed attempts: {chunk[:200]}...")
                            return None

            futures = [self.executor.submit(process_chunk, i, chunk) for i, chunk in enumerate(chunks)]
            for future in futures:
                try:
                    result = future.result(timeout=self.chunk_timeout)
                    if result:
                        kv_cache.append(result)
                    else:
                        skipped_chunks.append(result['chunk_id'] + 1 if result else len(skipped_chunks) + 1)
                except TimeoutError:
                    logger.error(f"Timeout processing chunk after {self.chunk_timeout} seconds")
                    skipped_chunks.append(len(skipped_chunks) + 1)

            if skipped_chunks:
                logger.warning(f"Skipped chunks: {skipped_chunks}. Proceeding with {len(kv_cache)}/{len(chunks)} chunks.")

            if not kv_cache:
                logger.error("No chunks processed successfully. Aborting preloading.")
                elapsed_time = time.time() - start_time
                return False, elapsed_time

            # Save KV Cache to disk
            self.save_kv_cache(user_id, cache_path, kv_cache)
            self.kv_cache[user_id] = kv_cache
            self.document_chunks[user_id] = [item['text'] for item in kv_cache]
            elapsed_time = time.time() - start_time
            logger.info(f"Preloaded {len(kv_cache)}/{len(chunks)} chunks and saved KV Cache for {file_name}")
            if skipped_chunks:
                logger.info(f"Note: {len(skipped_chunks)} chunks were skipped due to processing errors.")
            logger.info(f"CAG preload time: {elapsed_time:.2f} seconds")
            return True, elapsed_time
        except Exception as e:
            logger.error(f"Error preloading documents: {e}\n{traceback.format_exc()}")
            elapsed_time = time.time() - start_time
            return False, elapsed_time

    def save_kv_cache(self, user_id, cache_path, kv_cache):
        """Save KV Cache to disk"""
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(kv_cache, f)
            logger.debug(f"Saved KV Cache to {cache_path}")
        except Exception as e:
            logger.error(f"Error saving KV Cache: {e}\n{traceback.format_exc()}")
            raise

    def load_kv_cache(self, user_id, cache_path):
        """Load KV Cache from disk"""
        try:
            with open(cache_path, 'rb') as f:
                kv_cache = pickle.load(f)
            self.kv_cache[user_id] = kv_cache
            self.document_chunks[user_id] = [item['text'] for item in kv_cache]
            logger.debug(f"Loaded KV Cache from {cache_path}")
        except Exception as e:
            logger.error(f"Error loading KV Cache: {e}\n{traceback.format_exc()}")
            raise

    def reset_kv_cache(self, user_id):
        """Reset KV Cache by truncating query-specific tokens"""
        if user_id in self.query_tokens:
            logger.debug(f"Resetting KV Cache for user_id {user_id}")
            del self.query_tokens[user_id]
            logger.debug("Query tokens cleared for cache reset")

    def upload_document(self, file_path):
        """Handle document upload and process with CAG"""
        if not file_path:
            logger.debug("No file selected for upload")
            return

        try:
            logger.info(f"Uploading document for CAG: {file_path}")
            text = self._extract_text_from_file(file_path)
            if not text:
                self.root.after(0, lambda: messagebox.showerror("Error", "Could not extract text from the document"))
                return

            chunks = self._chunk_text(text)
            logger.debug(f"Extracted {len(chunks)} chunks from document")

            user_id = self.current_user_id if hasattr(self, 'current_user_id') else "default"
            file_name = os.path.basename(file_path)

            success, cag_time = self.preload_documents(user_id, chunks, file_name)
            if success:
                timing_message = f"Document processed successfully with CAG.\n- CAG processing time: {cag_time:.2f} seconds"
                self.root.after(0, lambda: messagebox.showinfo("Success", timing_message))
            else:
                self.root.after(0, lambda: messagebox.showerror("Error", "CAG processing failed. Check logs for details."))
        except Exception as e:
            logger.error(f"Error uploading document: {e}\n{traceback.format_exc()}")
            self.root.after(0, lambda: messagebox.showerror("Error", f"Failed to process document: {str(e)}"))

    def _extract_text_from_file(self, file_path):
        """Extract text from PDF, DOCX, or TXT files"""
        try:
            extension = os.path.splitext(file_path)[1].lower()
            if extension == '.pdf':
                with open(file_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    text = ""
                    for page in reader.pages:
                        extracted = page.extract_text()
                        if extracted:
                            text += extracted
            elif extension == '.docx':
                doc = Document(file_path)
                text = "\n".join([para.text for para in doc.paragraphs if para.text])
            elif extension == '.txt':
                with open(file_path, 'r', encoding='utf-8') as file:
                    text = file.read()
            else:
                text = textract.process(file_path).decode('utf-8')
            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting text from {file_path}: {e}")
            return None

    def _is_document_query(self, query):
        """Determine if the query is related to uploaded documents or requires CAG"""
        query_lower = self._normalize_query(query)
        document_keywords = [
            "document", "file", "policy", "manual", "guide", "terms", "conditions", "faq",
            "bar", "pub", "restaurant", "rooftop", "dining", "nightlife", "menu"
        ]
        doc = self.nlp(query_lower)

        # Check for explicit document-related keywords
        has_document_keyword = any(keyword in query_lower for keyword in document_keywords)

        # Check for file name references
        has_file_reference = False
        if self.current_user_id in self.document_chunks:
            for file_name in [item['file_name'] for item in self.kv_cache.get(self.current_user_id, [])]:
                if file_name.lower() in query_lower:
                    has_file_reference = True
                    break

        # Check for content overlap with document chunks
        has_content_overlap = False
        if self.current_user_id in self.document_chunks:
            query_tokens = set(query_lower.split())
            for chunk in self.document_chunks[self.current_user_id]:
                chunk_tokens = set(chunk.lower().split())
                if query_tokens & chunk_tokens:  # Non-empty intersection
                    has_content_overlap = True
                    break

        # Exclude weather or map queries to avoid false positives
        is_weather = self._is_weather_query(query)
        is_map = self._is_map_query(query)
        is_bengaluru_specific = "bengaluru" in query_lower or "bangalore" in query_lower

        # CAG is triggered only if:
        # 1. Explicit document keywords or file reference is present, OR
        # 2. Content overlap exists AND the query isn't primarily weather or map-related
        result = (
            (has_document_keyword or has_file_reference or has_content_overlap) and
            not (is_weather or is_map) and
            self.current_user_id in self.document_chunks
        ) or (is_bengaluru_specific and has_document_keyword and self.current_user_id in self.document_chunks)

        logger.debug(
            f"Document query check: "
            f"keywords={has_document_keyword}, file_reference={has_file_reference}, "
            f"content_overlap={has_content_overlap}, is_weather={is_weather}, is_map={is_map}, "
            f"bengaluru_specific={is_bengaluru_specific}, result={result}"
        )
        return result

    def _normalize_query(self, query):
        """Normalize query phrasing for better spaCy performance"""
        query = query.lower().strip()
        query = query.replace("weather of", "weather in")
        logger.debug(f"Normalized query: {query}")
        return query

    def _is_weather_query(self, query):
        """Determine if the query is weather-related"""
        query_lower = self._normalize_query(query)
        weather_keywords = ["weather", "temperature", "humidity", "forecast", "rain", "sunny", "cloudy", "today", "hourly", "week", "7", "weekly"]
        doc = self.nlp(query_lower)
        has_location = any(ent.label_ in ["GPE", "LOC"] for ent in doc.ents)
        has_weather_keyword = any(keyword in query_lower for keyword in weather_keywords)
        logger.debug(f"Weather keywords: {has_weather_keyword}, Location: {has_location}")
        return has_weather_keyword or has_location

    def _is_map_query(self, query):
        """Determine if the query is map/distance/routing related"""
        query_lower = self._normalize_query(query)
        map_keywords = ["map", "distance", "route", "travel", "directions", "navigate", "journey", "time", "far", "kilometers", "miles", "path", "drive", "walk", "commute"]
        doc = self.nlp(query_lower)
        locations = [ent.text for ent in doc.ents if ent.label_ in ["GPE", "LOC", "ORG"]]
        has_two_locations = len(locations) >= 2
        has_from_to = "from" in query_lower and "to" in query_lower
        has_source_destination = bool(re.search(r'source:.*destination:', query_lower, re.IGNORECASE))
        has_map_keyword = any(keyword in query_lower for keyword in map_keywords)
        logger.debug(f"Map keywords: {has_map_keyword}, From-To: {has_from_to}, Source-Destination: {has_source_destination}, Two locations: {has_two_locations}")
        return has_map_keyword or has_from_to or has_two_locations or has_source_destination

    def _fetch_weather_data(self, query):
        """Fetch weather data from OpenWeatherMap APIs"""
        if not self.weather_api_key:
            logger.error("No API key available")
            return None

        try:
            query_lower = self._normalize_query(query)
            doc = self.nlp(query_lower)
            city = None
            for ent in doc.ents:
                if ent.label_ in ["GPE", "LOC"]:
                    city = ent.text
                    break

            if not city:
                for key, city_name in KNOWN_CITIES.items():
                    if key in query_lower:
                        city = city_name
                        break

            if not city:
                logger.warning(f"No city found in query: {query_lower}")
                return None

            params = {
                "q": city,
                "appid": self.weather_api_key,
                "units": "metric"
            }
            response = requests.get(self.weather_api_url, params=params, timeout=5)
            if response.status_code != 200:
                logger.error(f"Weather API error: {response.status_code}, {response.text}")
                return None

            data = response.json()
            city_name = data["name"]
            current_date = datetime.fromtimestamp(data["dt"]).strftime("%Y-%m-%d")
            temp = data["main"]["temp"]
            humidity = data["main"]["humidity"]
            description = data["weather"][0]["description"].capitalize()
            lat = data["coord"]["lat"]
            lon = data["coord"]["lon"]

            onecall_params = {
                "lat": lat,
                "lon": lon,
                "appid": self.weather_api_key,
                "units": "metric",
                "exclude": "minutely,alerts"
            }
            onecall_response = requests.get(self.onecall_api_url, params=onecall_params, timeout=5)
            if onecall_response.status_code != 200:
                weather_data = {
                    "city": city_name,
                    "current": {
                        "date": current_date,
                        "description": description,
                        "temperature": temp,
                        "humidity": humidity
                    },
                    "note": "Hourly and weekly forecasts unavailable.",
                    "timestamp": time.time()
                }
                return weather_data

            onecall_data = onecall_response.json()
            rain_probability = onecall_data["hourly"][0].get("pop", 0) * 100
            will_rain = "likely" if rain_probability > 50 else "unlikely"

            hourly_forecast = []
            for hour in onecall_data["hourly"][:8]:
                hour_time = datetime.fromtimestamp(hour["dt"]).strftime("%Y-%m-%d %H:%M")
                hour_temp = hour["temp"]
                hour_humidity = hour["humidity"]
                hour_desc = hour["weather"][0]["description"].capitalize()
                hour_pop = hour.get("pop", 0) * 100
                hourly_forecast.append({
                    "time": hour_time,
                    "temperature": hour_temp,
                    "humidity": hour_humidity,
                    "description": hour_desc,
                    "rain_probability": hour_pop
                })

            daily_forecast = []
            for day in onecall_data["daily"][1:8]:
                day_date = datetime.fromtimestamp(day["dt"]).strftime("%Y-%m-%d")
                day_temp = day["temp"]["day"]
                day_humidity = day["humidity"]
                day_desc = day["weather"][0]["description"].capitalize()
                day_pop = day.get("pop", 0) * 100
                daily_forecast.append({
                    "date": day_date,
                    "temperature": day_temp,
                    "humidity": day_humidity,
                    "description": day_desc,
                    "rain_probability": day_pop
                })

            weather_data = {
                "city": city_name,
                "current": {
                    "date": current_date,
                    "description": description,
                    "temperature": temp,
                    "humidity": humidity,
                    "rain_probability": rain_probability,
                    "will_rain": will_rain
                },
                "hourly_forecast": hourly_forecast,
                "daily_forecast": daily_forecast,
                "timestamp": time.time()
            }
            logger.info(f"Weather data fetched for {city_name}")
            return weather_data
        except Exception as e:
            logger.error(f"Error fetching weather data: {e}\n{traceback.format_exc()}")
            return None

    def _extract_locations(self, query):
        """Extract source and destination locations from a query"""
        query_lower = self._normalize_query(query)
        doc = self.nlp(query_lower)
        locations = [ent.text for ent in doc.ents if ent.label_ in ["GPE", "LOC", "ORG"]]

        source = None
        destination = None
        source_match = re.search(r'source:\s*([^\n]+)', query_lower, re.IGNORECASE)
        dest_match = re.search(r'destination:\s*([^\n]+)', query_lower, re.IGNORECASE)
        if source_match and dest_match:
            source = source_match.group(1).strip()
            destination = dest_match.group(1).strip()
        elif "from" in query_lower and "to" in query_lower:
            try:
                parts = query_lower.split("from")[1].split("to")
                source = parts[0].strip()
                destination = parts[1].strip().split()[0]
            except:
                pass

        if not (source and destination) and len(locations) >= 2:
            source = locations[0]
            destination = locations[1]

        if source and destination:
            for key, full_name in KNOWN_INSTITUTIONS.items():
                if key in source.lower():
                    source = full_name
                if key in destination.lower():
                    destination = full_name
            return source, destination

        found_cities = []
        for key, city_name in KNOWN_CITIES.items():
            if key in query_lower:
                found_cities.append(city_name)
        if len(found_cities) >= 2:
            return found_cities[0], found_cities[1]

        logger.warning(f"Could not extract two locations from: {query}")
        return None, None

    def _get_coordinates(self, location):
        """Get coordinates for a location using OpenRouteService geocoding API"""
        try:
            headers = {
                'Authorization': self.ors_api_key,
                'Content-Type': 'application/json; charset=utf-8'
            }
            params = {
                "api_key": self.ors_api_key,
                "text": location,
                "size": 1,
                "boundary.country": "IN"
            }
            response = requests.get(self.ors_geocode_url, params=params, headers=headers, timeout=5)
            if response.status_code != 200:
                logger.error(f"Geocoding API error: {response.status_code}, {response.text}")
                return None
            data = response.json()
            if not data.get("features"):
                logger.warning(f"No coordinates found for: {location}")
                return None
            coords = data["features"][0]["geometry"]["coordinates"]
            return coords[0], coords[1]
        except Exception as e:
            logger.error(f"Error getting coordinates: {e}\n{traceback.format_exc()}")
            return None

    def _fetch_route_data(self, source, destination):
        """Fetch route data between source and destination"""
        if not self.ors_api_key:
            logger.error("No OpenRouteService API key")
            return None
        try:
            source_coords = self._get_coordinates(source)
            destination_coords = self._get_coordinates(destination)
            if not source_coords or not destination_coords:
                return None
            headers = {
                'Authorization': self.ors_api_key,
                'Content-Type': 'application/json; charset=utf-8'
            }
            payload = {
                "coordinates": [source_coords, destination_coords],
                "profile": "driving-car",
                "format": "json",
                "units": "km",
                "language": "en",
                "instructions": True
            }
            response = requests.post(
                f"{self.ors_api_url}/driving-car",
                headers=headers,
                json=payload,
                timeout=10
            )
            if response.status_code != 200:
                logger.error(f"Routing API error: {response.status_code}, {response.text}")
                return None
            data = response.json()
            if not data.get("routes"):
                logger.error("No routes found")
                return None
            route_data = {
                "source": source,
                "destination": destination,
                "distance": data["routes"][0]["summary"]["distance"],
                "duration": data["routes"][0]["summary"]["duration"],
                "bbox": data["routes"][0]["bbox"],
                "timestamp": time.time(),
                'steps': []
            }
            steps = []
            for segment in data["routes"][0]["segments"]:
                for step in segment["steps"]:
                    steps.append({
                        "instruction": step["instruction"],
                        "distance": step["distance"],
                        "duration": step["duration"]
                    })
            route_data['steps'] = steps
            map_context = (
                f"Route from {route_data['source']} to {route_data['destination']}:\n"
                f"- Distance: {route_data['distance']:.2f} kilometers\n"
            )
            duration_seconds = route_data["duration"]
            hours = int(duration_seconds // 3600)
            minutes = int((duration_seconds % 3600) // 60)
            if hours > 0:
                map_context += f"- Travel time: {hours} hour{'s' if hours > 1 else ''} and {minutes} minute{'s' if minutes > 1 else ''}\n"
            else:
                map_context += f"- Travel time: {minutes} minute{'s' if minutes > 1 else ''}\n"
            if len(route_data['steps']) > 0:
                map_context += "\nDirections:\n"
                step_count = min(5, len(route_data['steps']))
                for i, step in enumerate(route_data['steps'][:step_count]):
                    map_context += f"- Step {i+1}: {step['instruction']} ({step['distance']:.2f} km)\n"
                if len(route_data['steps']) > step_count:
                    map_context += f"- ... and {len(route_data['steps']) - step_count} more steps\n"
            route_data['map_context'] = map_context + "Use this data to provide a detailed routing response."
            logger.info(f"Route data fetched")
            return route_data
        except Exception as e:
            logger.error(f"Error fetching route data: {e}\n{traceback.format_exc()}")
            return None

    def retrieve_relevant_interactions(self, query, user_id):
        """Retrieve relevant past interactions from Qdrant"""
        try:
            if not hasattr(self, 'qdrant_client'):
                logger.warning("Qdrant client not initialized. Skipping interaction retrieval.")
                return []

            # Use updated Qdrant search API
            search_result = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=[0] * 384,  # Dummy vector if embedder is removed
                query_filter=Filter(
                    must=[
                        FieldCondition(
                            key="user_id",
                            match=MatchValue(value=user_id)
                        )
                    ]
                ),
                limit=5
            )
            interactions = [
                {
                    'query': point.payload.get('query', ''),
                    'themes': point.payload.get('themes', [])
                }
                for point in search_result
            ]
            logger.debug(f"Retrieved {len(interactions)} relevant interactions")
            return interactions
        except Exception as e:
            logger.error(f"Failed to retrieve interactions: {e}\n{traceback.format_exc()}")
            return []

    def _retrieve_relevant_chunks(self, query, user_id, top_k=10):
        """Retrieve the most relevant document chunks based on token overlap"""
        if user_id not in self.document_chunks or not self.document_chunks[user_id]:
            return []

        query_tokens = set(self._normalize_query(query).split())
        chunk_scores = []

        for i, chunk in enumerate(self.document_chunks[user_id]):
            chunk_tokens = set(chunk.lower().split())
            common_tokens = query_tokens & chunk_tokens
            score = len(common_tokens) / max(len(chunk_tokens), 1)  # Normalize by chunk length
            chunk_scores.append((i, chunk, score))

        # Sort by score in descending order and select top_k
        chunk_scores.sort(key=lambda x: x[2], reverse=True)
        relevant_chunks = [(i, chunk) for i, chunk, score in chunk_scores[:top_k] if score > 0]

        logger.debug(f"Retrieved {len(relevant_chunks)} relevant chunks for query: {query}")
        return relevant_chunks

    def process_query(self, user_id, query, face_frame=None):
        """Process a user query with weather, map, CAG, or general handling"""
        logger.info(f"Processing query for user_id {user_id}: {query}")
        start_time = time.time()
        user_data = self.get_user(user_id)
        user_preferences = {}
        if user_data:
            try:
                user_preferences = json.loads(user_data['preferences'])
            except Exception as e:
                logger.error(f"Error parsing user preferences: {e}")

        past_interactions = self.retrieve_relevant_interactions(query, user_id)
        context = ""
        if past_interactions:
            context = "Previous interactions:\n"
            for interaction in past_interactions:
                context += f"- Query: {interaction['query']}, Themes: {', '.join(interaction['themes'])}\n"

        cag_context = ""
        cag_time = 0
        prefix = ""
        query_mode = "General"

        # Determine query type
        is_document_query = self._is_document_query(query)
        is_weather_query = self._is_weather_query(query)
        is_map_query = self._is_map_query(query)

        face_emotion = self.detect_emotion_from_face(face_frame) if face_frame is not None else "unknown"

        weather_context = ""
        if is_weather_query and not is_document_query:
            query_mode = "Weather"
            weather_data = self._fetch_weather_data(query)
            if weather_data:
                current = weather_data["current"]
                weather_context = (
                    f"Current weather for {weather_data['city']} as of {current['date']}:\n"
                    f"- Description: {current['description']}\n"
                    f"- Temperature: {current['temperature']}°C\n"
                    f"- Humidity: {current['humidity']}%\n"
                )
                if "rain_probability" in current:
                    weather_context += f"- Rain Probability: {current['rain_probability']}% (rain is {current['will_rain']})\n"
                query_lower = query.lower()
                if "hourly" in query_lower or "forecast" in query_lower:
                    if "hourly_forecast" in weather_data:
                        weather_context += "\nHourly forecast for the next 24 hours:\n"
                        for hour in weather_data["hourly_forecast"]:
                            weather_context += (
                                f"- {hour['time']}: {hour['description']}, "
                                f"Temp: {hour['temperature']}°C, "
                                f"Humidity: {hour['humidity']}%, "
                                f"Rain: {hour['rain_probability']}%\n"
                            )
                    else:
                        weather_context += f"\n{weather_data['note']}\n"
                if any(k in query_lower for k in ["week", "7", "weekly", "forecast"]):
                    if "daily_forecast" in weather_data:
                        weather_context += "\n7-day forecast:\n"
                        for day in weather_data["daily_forecast"]:
                            weather_context += (
                                f"- {day['date']}: {day['description']}, "
                                f"Temp: {day['temperature']}°C, "
                                f"Humidity: {day['humidity']}%, "
                                f"Rain: {day['rain_probability']}%\n"
                            )
                    else:
                        weather_context += f"\n{weather_data['note']}\n"
                weather_context += "Use this data to provide a detailed weather response."
            else:
                weather_context = (
                    f"Unable to fetch weather data for the requested location in query: '{query}'. "
                    "Please clarify the city or check the API key and network."
                )

        map_context = ""
        if is_map_query and not is_document_query:
            query_mode = "Map"
            source, destination = self._extract_locations(query)
            if source and destination:
                route_data = self._fetch_route_data(source, destination)
                if route_data:
                    map_context = route_data['map_context']
                else:
                    map_context = (
                        f"Unable to fetch route data between {source} and {destination}. "
                        "Please check location names or try different locations."
                    )
            else:
                map_context = (
                    f"Unable to identify source and destination from query: '{query}'. "
                    "Please specify locations clearly, e.g., 'source: Mumbai, destination: Delhi'."
                )

        # Handle CAG queries
        if is_document_query:
            query_mode = "CAG"
            prefix = "[CAG]"
            start_time_cag = time.time()
            if user_id in self.document_chunks:
                relevant_chunks = self._retrieve_relevant_chunks(query, user_id, top_k=10)
                if relevant_chunks:
                    cag_context = "Preloaded document information (CAG):\n"
                    for i, (chunk_id, chunk) in enumerate(relevant_chunks):
                        cag_context += f"- Chunk {chunk_id+1}: {chunk[:300]}...\n"
                    logger.debug(f"Constructed CAG context with {len(relevant_chunks)} chunks")
                else:
                    cag_context = "No relevant document chunks found for the query."
                    prefix = "[CAG-Partial]"
            else:
                cag_context = "No preloaded documents available for CAG. Please upload a document."
                prefix = "[CAG-None]"
            cag_time = time.time() - start_time_cag
            logger.info(f"CAG context retrieval time: {cag_time:.2f} seconds")

        # Update status bar with query mode
        self.status_var.set(f"Ready | Query Mode: {query_mode}")

        prompt = self._build_prompt(
            query,
            user_preferences,
            face_emotion,
            context,
            weather_context,
            map_context,
            cag_context,
            is_document_query
        )
        inference_time = 0

        response_text = None
        if not self.ollama_available:
            response_text = f"{prefix} Sorry, unable to connect to the AI model. Ensure Ollama server is running with 'llama3.2:3b'."
        else:
            for attempt in range(3):
                try:
                    logger.debug(f"Attempting to generate response (attempt {attempt + 1})...")
                    start_time_inference = time.time()
                    response = ollama.generate(
                        model='llama3.2:3b',
                        prompt=prompt,
                        options={'max_tokens': 1000, 'temperature': 0.7}
                    )
                    inference_time = time.time() - start_time_inference
                    response_text = response['response'].strip()
                    logger.info(f"Response generated: {response_text[:50]}...")
                    response_text = f"{prefix} {response_text}" if prefix else response_text
                    # Track query tokens (simulated)
                    self.query_tokens[user_id] = {'query': query, 'timestamp': time.time()}
                    break
                except Exception as e:
                    logger.error(f"Error generating response (attempt {attempt + 1}): {e}\n{traceback.format_exc()}")
                    if attempt < 2:
                        time.sleep(1)
                    else:
                        response_text = f"{prefix} Error processing query: '{query}'. Please try again." if prefix else f"Error processing query: '{query}'. Please try again."

        self.reset_kv_cache(user_id)

        new_preferences = self.extract_preferences_from_text(query)
        if new_preferences and user_id and self.current_user_name != "Guest":
            if self.update_user_preferences(user_id, new_preferences):
                logger.info(f"Updated preferences for user_id {user_id}")
            else:
                logger.warning(f"Failed to update preferences for user_id {user_id}")

        themes = self.extract_themes(query)
        if themes and user_id:
            self.store_interaction(user_id, query, themes)

        audio_file = self.text_to_speech(response_text)

        total_time = time.time() - start_time
        logger.info(f"Total query processing time: {total_time:.2f} seconds")

        timing_message = f"\nQuery Processing Times:\n- Total: {total_time:.2f} seconds\n"
        if cag_time > 0:
            timing_message += f"- CAG Context Retrieval: {cag_time:.2f} seconds\n"
        if inference_time > 0:
            timing_message += f"- Inference: {inference_time:.2f} seconds\n"
        response_text += timing_message

        return {
            'text': response_text,
            'audio_file': audio_file,
            'detected_emotion': {
                'face': face_emotion
            }
        }

    def _build_prompt(self, query, user_preferences, face_emotion, context, weather_context="", map_context="", cag_context="", is_document_query=False):
        """Build a prompt for the LLM based on query type"""
        if is_document_query:
            prompt = (
                "You are a helpful AI assistant using Cache-Augmented Generation (CAG). "
                "Your task is to provide accurate and comprehensive responses based on preloaded document context. "
                "For document-related queries, use ALL relevant preloaded document information (CAG context) provided to generate a complete response. "
                "Synthesize information from all document chunks to avoid incomplete answers, and prioritize details that directly address the query. "
                "Ensure responses are concise, structured as bullet points for clarity, and directly address the query. "
                "If no relevant data is available, state that clearly and suggest the user provide more details or upload a document.\n"
            )
        else:
            prompt = (
                "You are a helpful AI assistant. "
                "Your task is to provide accurate and concise responses based on provided context or general knowledge. "
                "For weather-related queries, use the provided weather data to give precise answers. "
                "For map or routing queries, use the provided route data to detail distances, travel times, and directions. "
                "For general queries, rely on your knowledge and context from past interactions. "
                "Ensure responses are structured as bullet points for clarity and directly address the query.\n"
            )

        if context:
            prompt += f"{context}\n"
        if user_preferences:
            prompt += f"User preferences: {user_preferences}\n"
        if face_emotion != "unknown":
            prompt += f"User's facial expression: {face_emotion}. Adjust tone accordingly.\n"
        if weather_context:
            prompt += f"{weather_context}\n"
        if map_context:
            prompt += f"{map_context}\n"
        if cag_context and is_document_query:
            prompt += f"{cag_context}\n"
        prompt += f"Query: {query}"
        logger.debug(f"Prompt built: {prompt[:200]}...")
        return prompt

    def cleanup(self):
        """Clean up resources including KV Cache and pygame mixer"""
        super().cleanup()
        self.listening = False
        if hasattr(self, 'kv_cache'):
            self.kv_cache.clear()
            logger.debug("Cleared KV Cache")
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)
            logger.debug("Shut down ThreadPoolExecutor")
        if self.voice_enabled:
            pygame.mixer.quit()
            logger.debug("Closed pygame mixer")

if __name__ == "__main__":
    try:
        logger.info("Starting WeatherMapAssistant with CAG...")
        root = tk.Tk()
        style = ttk.Style()
        style.theme_use('clam')
        app = WeatherMapAssistantGUI(root)
        app.run()
    except Exception as e:
        logger.error(f"Fatal error: {e}\n{traceback.format_exc()}")
        if 'root' in locals():
            messagebox.showerror("Error", f"Application error: {str(e)}")
        sys.exit(1)