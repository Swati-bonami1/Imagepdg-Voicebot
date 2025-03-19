# Imagepdf-Voicebot

 Functionalities of This Model
 PDF Processing

- Extracts text and images from PDFs using PyMuPDF (fitz)
-  Detects images and replaces them with AI-generated descriptions
- Saves the processed text back into a new PDF

Image Processing & Recognition

- Extracts images from PDFs and saves them as PNGs
- Uses OpenAI’s GPT-4 Turbo to generate descriptions of images
- Runs image-to-text processing in parallel for speed optimization

 AI-Powered Search & Retrieval

- Uses FAISS to store & retrieve document embeddings
- Splits document text into smaller chunks for better search accuracy
- Retrieves relevant information based on user queries

Speech Recognition & AI Response

- Listens to user speech input using SpeechRecognition
- Converts speech to text using Google Speech API
- Searches the document for relevant answers
- Uses OpenAI's GPT-4o to generate responses
- Reads out the response using pyttsx3 text-to-speech

Performance Optimizations

- Multithreading for parallel image processing (ThreadPoolExecutor)
- Efficient text chunking using LangChain
- Index optimization in FAISS for fast searches

API Endpoints (Flask)


mainimage.py= uses openai for all functions
mainIVF.py= uses sentence transformer and embedding

🔹 /upload → Uploads a PDF, processes it, and returns a new version with images replaced
🔹 /ask → Takes voice input, finds relevant answers, and speaks the response

