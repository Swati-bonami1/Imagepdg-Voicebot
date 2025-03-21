import os
import numpy as np
import faiss, openai, time, pyttsx3
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader  
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from flask import Flask, request, jsonify
import speech_recognition as sr
import warnings, asyncio
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer
from bert_score import score

warnings.filterwarnings("ignore", category=DeprecationWarning)
similarity_model = SentenceTransformer("all-MiniLM-L6-v2")

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

if not api_key:
    raise ValueError("Missing OpenAI API key! Set OPENAI_KEY in environment variables.")

engine = pyttsx3.init()
engine.setProperty("rate", 150)

app = Flask(__name__)
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
retriever = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True).as_retriever() 
retriever.search_kwargs = {"k": 10}

def remove_non_latin1(text):
    return text.encode("latin-1", "ignore").decode("latin-1")

async def fetch_docs(query):
    docs = await retriever.aget_relevant_documents(query) 
    return docs

def process_document(file_path):
    """Loads a PDF file, extracts text, splits it into chunks, and stores embeddings in FAISS."""
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    print(f"Loaded {len(chunks)} document chunks.")

    chunk_texts = [doc.page_content for doc in chunks]
    all_embeddings = np.array(embedding_model.embed_documents(chunk_texts)).astype(np.float32)

    dimension = 1536 
    if len(all_embeddings) < 100:
        print("Not enough data for IVF. Using Flat Search instead.")
        index = faiss.IndexFlatL2(dimension) 
    else:
        quantizer = faiss.IndexFlatL2(dimension)
        nlist = min(len(all_embeddings) // 2, 30)
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
        index.train(all_embeddings)
    
    vector_db = FAISS.from_documents(chunks, embedding_model)
    vector_db.save_local("faiss_index")
    print("FAISS index created and stored successfully!")

    return vector_db

def generate_openai_response(query):
    try:
        vector_db = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)
        retriever = vector_db.as_retriever()

        docs = asyncio.run(fetch_docs(query))  
        context = " ".join([doc.page_content for doc in docs])

        prompt = (
            "You are an AI assistant that answers questions based on the provided document context.\n"
            "Ensure your answer is accurate and relevant to the document, recheck your answer thoroughly before responding."
            "Generate your response strictly from the document given.\n"
            "If no relevant information is found, respond with: 'Sorry, no relevant information found.'\n\n"
            f"Context:{context}"
            f"Question:\n{query}"
        )

        response = client.chat.completions.create(
            model="gpt-4o", 
            messages=[{"role": "system", "content": "You are a helpful AI assistant."},
                      {"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error retrieving data: {str(e)}"

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    file_path = os.path.join("uploads", file.filename)
    file.save(file_path)

    process_document(file_path)  
    global retriever
    vector_db = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)
    retriever = vector_db.as_retriever()

    print(f"File uploaded and FAISS index updated: {file_path}")
    return jsonify({"message": "File processed and indexed successfully!"})

@app.route('/ask', methods=['POST'])
def ask_question():
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print("Say something...")
        recognizer.adjust_for_ambient_noise(source)  
        audio = recognizer.listen(source)

        try:
            text = recognizer.recognize_google(audio)   
            print("You said:", text)
            query = text  
            docs = asyncio.run(fetch_docs(query))  
            retrieved_texts = [doc.page_content for doc in docs] if docs else []

            response = generate_openai_response(query)  

            bert_score_result = {"precision": 0, "recall": 0, "f1": 0, "message": "Not computed."}
            if retrieved_texts:
                faithfulness_result = check_faithfulness(query, response, retrieved_texts)
                bert_score_result = evaluate_bert_score(response, retrieved_texts, model_type="bert-base-uncased")
            else:
                faithfulness_result = {"faithfulness_score": 0, "message": "No relevant documents found."}

            engine.say(response)
            engine.runAndWait()

            return jsonify({
                "response": response,
                "faithfulness_score": faithfulness_result,
                "bert_score_result": bert_score_result
            })

        except sr.UnknownValueError:
            return jsonify({"response": "Could not understand the audio."})
        except sr.RequestError:
            return jsonify({"response": "Could not request results from Google Speech API."})

def check_faithfulness(query, response, docs):
    if not docs:
        return {"faithfulness_score": 0, "message": "No documents retrieved!"}
    retrieved_text = " ".join(docs)

    # Compute embeddings
    response_embedding = similarity_model.encode(response, convert_to_tensor=True)
    retrieved_embedding = similarity_model.encode(retrieved_text, convert_to_tensor=True)

    # Compute cosine similarity (range: 0-1, higher is better)
    faithfulness_score = util.pytorch_cos_sim(response_embedding, retrieved_embedding).item()

    return {
        "faithfulness_score": round(faithfulness_score, 4),
        "query": query,
        "response": response,
        "retrieved_docs": docs
    }

def evaluate_bert_score(response, retrieved_docs, model_type="bert-base-uncased"):
    if not retrieved_docs:
        return {"precision": 0, "recall": 0, "f1": 0, "message": "No retrieved documents!"}
        reference_text = " ".join(retrieved_docs).strip()
    
    if not reference_text:
        return {"precision": 0, "recall": 0, "f1": 0, "message": "Retrieved documents are empty!"}

     # Compute BERTScore
    P, R, F1 = score([response], [reference_text], model_type= "bert-base-uncased", lang="en", rescale_with_baseline=True)

    return {
        "precision": round(P.tolist()[0], 4),
        "recall": round(R.tolist()[0], 4),
        "f1": round(F1.tolist()[0], 4),
        "response": response,
        "retrieved_docs": retrieved_docs
        }

if __name__ == '__main__':
    if not os.path.exists("uploads"):
        os.makedirs("uploads")
    app.run(host="0.0.0.0", port=5000, debug=True)

