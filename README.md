Prerequisites





Python: 3.8+



Ollama: Installed with llama3.2:3b model pulled.



Qdrant: Running locally (default: http://localhost:6333).



APIs:





OpenWeatherMap API key (free tier).



OpenRouteService API key (free tier).



System Dependencies:





portaudio for voice input (speech_recognition).



ffmpeg for textract (document processing).



On Ubuntu:

sudo apt-get install -y python3-dev portaudio19-dev ffmpeg

Setup





Clone the Repository:

git clone https://github.com/<your-username>/llama3_assistant.git
cd llama3_assistant



Create and Activate Virtual Environment:

python3 -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows



Install Dependencies:

pip install -r requirements.txt

Note: If requirements.txt is missing, install manually (see below).



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

Running the Application





Activate Virtual Environment:

source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows



Start Ollama Server (if not running):

ollama serve



Run the Script:

python test.py



Interact with the GUI:





Select/Add User: Choose or create a user from the dropdown.



Upload Document: Click "Upload Document (CAG)" to preload documents (e.g., curated_documents/Top Bars & Pubs in Bengaluru -1.pdf).



Voice Mode: Click "Enable Voice Mode" and say "Jarvis" followed by your query.



Text Input: Type queries (e.g., "weather in Bengaluru", "route from Mumbai to Delhi", "good pubs in Bengaluru").



View Responses: Responses appear in the chat window, with voice output if enabled.

Example Queries





Weather: "What's the weather in Bengaluru today?"



Routing: "Route from IAST Software Solutions to BMSIT & M College"



Document (CAG): "Tell me some good pubs in Bengaluru" (after uploading Top Bars & Pubs in Bengaluru -1.pdf)



General: "What are some fun things to do in Bengaluru?"

Inspecting KV Cache

To view KV cache contents (e.g., for uploaded documents):

python kv_cache.py

Note: Ensure kv_cache.py is updated to target the correct cache file (see previous conversation).

Troubleshooting





Ollama Not Responding:





Check server: curl http://127.0.0.1:11434



Restart: ollama serve



Qdrant Errors:





Verify Qdrant is running: docker ps



Check main.py for correct Qdrant configuration.



Voice Input Issues:





Ensure portaudio is installed and microphone is accessible.



Test: python -c "import speech_recognition as sr; r = sr.Recognizer(); with sr.Microphone() as source; print(r.recognize_google(r.listen(source)))"



Document Upload Fails:





Check logs in data/assistant.log or console.



Ensure ffmpeg and textract dependencies are installed.



API Errors:





Verify API keys in test.py.



Check network: ping api.openweathermap.org

Dependencies

If requirements.txt is missing, install:

pip install requests spacy tkinter PyPDF2 python-docx textract ollama speechrecognition gtts pygame qdrant-client


**Contributing**
Fork the repository.
-Create a feature branch: git checkout -b feature-name
-Commit changes: git commit -m "Add feature"
-Push: git push origin feature-name
-Open a pull request.