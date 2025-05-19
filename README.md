**AGENTIC CHATBOT WITH CAG**

A Python-based offline AI assistant with Cache-Augmented Generation (CAG), weather forecasting, route planning, and voice input/output capabilities. Built using llama3.2:3b (via Ollama), OpenWeatherMap, OpenRouteService, and spaCy, it supports document-based queries, weather updates, and navigation directions, with a Tkinter GUI.

**Features**

-Cache-Augmented Generation (CAG): Preload documents (PDF, DOCX, TXT) to answer queries using stored document chunks, with KV cache for efficiency.

-Weather Forecasts: Fetch real-time weather and forecasts using OpenWeatherMap API.

-Route Planning: Calculate driving routes and distances using OpenRouteService API.

-Voice Interaction: Supports voice input (wake word: "Jarvis") and text-to-speech output using speech_recognition and gTTS.

-Mood Detection: Integrates facial emotion detection (requires main.py with relevant code).

-Qdrant Integration: Stores past interactions for context (requires Qdrant setup).

-Local LLM: Uses llama3.2:3b via Ollama for offline query processing.



**Setup**

Clone the Repository:

[Agentic Chatbot with CAG](https://github.com/amaraashishmahatokushwaha/Agentic-chatbot-with-CAG "GitHub Repository for Agentic Chatbot with CAG")


cd llama3_assistant

Create and Activate Virtual Environment:

python3 -m venv venv

source venv/bin/activate  # Linux/Mac

venv\Scripts\activate     # Windows


**Install Dependencies:**

pip install -r requirements.txt

Install spaCy Model:

python -m spacy download en_core_web_sm

Set Up Ollama:

Install Ollama: Ollama Installation Guide.

Pull llama3.2:3b:

ollama pull llama3.2:3b

Start Ollama server:


ollama serve

Set Up Qdrant:

Install Qdrant: Qdrant Docker Setup.

Run Qdrant locally:

docker run -p 6333:6333 qdrant/qdrant

Configure API Keys:

In test.py, replace API keys (or set environment variables):

self.weather_api_key = "your_openweathermap_key"  # Get from https://openweathermap.org/

self.ors_api_key = "your_openrouteservice_key"    # Get from https://openrouteservice.org/


**Running the Application**

Activate Virtual Environment:

#source venv/bin/activate  # Linux/Mac

#venv\Scripts\activate     # Windows


Start Ollama Server (if not running):

ollama serve

Run the Script:

python3 test.py

**Interact with the GUI:**

Select/Add User: Choose or create a user from the dropdown.

Upload Document: Click "Upload Document (CAG)" to preload documents (e.g., curated_documents/Top Bars & Pubs in Bengaluru -1.pdf).

Voice Mode: Click "Enable Voice Mode" and say "Jarvis" followed by your query.

Text Input: Type queries (e.g., "weather in Bengaluru", "route from Mumbai to Delhi", "good pubs in Bengaluru").

View Responses: Responses appear in the chat window, with voice output if enabled.

**Example Queries**


Weather: "What's the weather in Bengaluru today?"

**Inspecting KV Cache**

To view KV cache contents (e.g., for uploaded documents):

python kv_cache.py

Note: Ensure kv_cache.py is updated to target the correct cache file (see previous conversation).

**Troubleshooting**

Ollama Not Responding:

Check server: curl http://127.0.0.1:11434

Restart: ollama serve

**Qdrant Errors:**

Verify Qdrant is running: docker ps

Check main.py for correct Qdrant configuration.

**Voice Input Issues:**

Ensure portaudio is installed and microphone is accessible.

Test: python -c "import speech_recognition as sr; r = sr.Recognizer(); with sr.Microphone() as source; print(r.recognize_google(r.listen(source)))"

**Document Upload Fails:**

Check logs in data/assistant.log or console.

Ensure ffmpeg and textract dependencies are installed.

**API Errors:**

Verify API keys in test.py.

Check network: ping api.openweathermap.org

**Dependencies**

If requirements.txt is missing, install:

pip install requests spacy tkinter PyPDF2 python-docx textract ollama speechrecognition gtts pygame qdrant-client

**Contributing**

Fork the repository.

-Create a feature branch: git checkout -b feature-name

-Commit changes: git commit -m "Add feature"

-Push: git push origin feature-name

-Open a pull request.
