import os
import numpy as np
import faiss, fitz, io, openai,time, pyttsx3, base64
from openai import OpenAIError
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader  
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from flask import Flask, request, jsonify
import speech_recognition as sr
from PIL import Image
from fpdf import FPDF
import concurrent.futures
from openai import OpenAI
import warnings, asyncio, torch
from bert_score import score
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer


warnings.filterwarnings("ignore", category=DeprecationWarning)
similarity_model = SentenceTransformer("all-MiniLM-L6-v2")

load_dotenv()
api_key= os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


max_workers = min(32, os.cpu_count() + 4)
print(f"Default max_workers: {max_workers}")

if not api_key:
    raise ValueError("Missing OpenAI API key! Set OPENAI_KEY in environment variables.")

engine = pyttsx3.init()
engine.setProperty("rate", 150)

app = Flask(__name__)
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
retriever = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True).as_retriever() 
retriever.search_kwargs = {"k": 10}
# Function to extract text and images
def extract_text_images_from_pdf(file_path):
    doc = fitz.open(file_path)
    content = []  
    images = []   

    for page_num, page in enumerate(doc):
        blocks = page.get_text("dict")["blocks"]
        
        for block in blocks:
            if "lines" in block:
                text = "\n".join([span["text"] for line in block["lines"] for span in line["spans"]])
                content.append(text)
            elif "image" in block:
                img_index = len(images)
                images_meta = page.get_images(full=True)
                
                for img in images_meta:
                    xref = img[0]  
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image = Image.open(io.BytesIO(image_bytes))

                    image_path = f"extracted_image_{img_index}.png"
                    image.save(image_path, "PNG")  # Save image file

                    images.append(image_path)  # Store file path
                    content.append(f"[IMAGE_{img_index}]")
                    print(f"Extracted Image: {image_path}")  

    return content, images


def remove_non_latin1(text):
    return text.encode("latin-1", "ignore").decode("latin-1")


async def fetch_docs(query):
    docs = await retriever.aget_relevant_documents(query) 
    return docs



# Image to text conversion function
def image_to_text_gpt4(image_path):
    if not os.path.exists(image_path):  
        print(f"Image not found: {image_path}")
        return f"[Image {image_path} not found]"

    try:
        with open(image_path, "rb") as img_file:
            base64_image = base64.b64encode(img_file.read()).decode("utf-8")

        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "Describe the image accurately in a way that fits within a PDF document."},
                {"role": "user", "content": [
                    {"type": "text", "text": "Describe this image in detail."},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                ]}
            ],
            max_tokens=500
        )

        image_text = response.choices[0].message.content.strip()
        print(f"Image {image_path} Description: {image_text}")

    except openai.OpenAIError as e:
        return f"[Error processing {image_path}: {e}]"


# Function to save text to PDF
def save_text_to_pdf(text_list, output_pdf):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    print("\nFinal PDF Content:")
    for text in text_list:
        print(text)  # Print text content before saving
        pdf.multi_cell(0, 10, remove_non_latin1(text))
        pdf.ln()

    pdf.output(output_pdf)
    print(f"PDF saved as {output_pdf}")
    return output_pdf



def Updateimage_text(pdf_images):
    global pdf_content 

    if pdf_images:
        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:  
            image_texts = list(executor.map(image_to_text_gpt4, pdf_images))
            total_time = time.time() - start_time
            print(f"Total processing time: {total_time:.2f} seconds")
        
        for idx, image_text in enumerate(image_texts):
            pdf_content = [text.replace(f"[IMAGE_{idx}]", image_text) for text in pdf_content]  
            pdf_content = [text.replace("\uf0e3", "") for text in pdf_content]  

        print("Updated content with image descriptions:", pdf_content)
    else:
        print("No images found in the PDF.")


# Function to process and store user-uploaded document
def process_document(file_path):
    """Loads a PDF file, extracts text, splits it into chunks, and stores embeddings in FAISS."""
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    print(f"Loaded {len(chunks)} document chunks.")

    # Convert chunks to embeddings
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
    
    # Store embeddings in FAISS
    vector_db = FAISS.from_documents(chunks, embedding_model)
    vector_db.save_local("faiss_index")
    print("FAISS index created and stored successfully!")

    return vector_db


#OPenAI response
def generate_openai_response(query):
    """Retrieves relevant chunks from FAISS and generates an answer using OpenAI GPT-4."""
    try:
        vector_db = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)
        retriever = vector_db.as_retriever()

        # Retrieve relevant chunks

        docs = asyncio.run(fetch_docs(query))  
        context = " ".join([doc.page_content for doc in docs])

        # Generate response using OpenAI
        prompt = (
            "You are an AI assistant that answers questions based on the provided document context.\n"
            "Ensure your answer is accurate and relevant to the document, recheck your answer thoroughly before responding.Generate your response strictly from document given.\n"
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
    """Handles file uploads from users, processes the document, and updates FAISS."""
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    file_path = os.path.join("uploads", file.filename)
    file.save(file_path)

    # Extract text and images from PDF
    pdf_content, pdf_images = extract_text_images_from_pdf(file_path)

    if pdf_images:
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor: 
            image_texts = list(executor.map(image_to_text_gpt4, pdf_images)) 
            for idx, image_text in enumerate(image_texts):
                pdf_content = [text.replace(f"[IMAGE_{idx}]", image_text) for text in pdf_content]
    else:
        image_texts = []

    # Save updated text to PDF
    updated_document = "updated_document.pdf"
    save_text_to_pdf(pdf_content, updated_document) 
    process_document(file_path)  
    global retriever
    vector_db = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)
    retriever = vector_db.as_retriever()

    print(f"File uploaded and FAISS index updated: {file_path}")
    return jsonify({"message": "File processed and indexed successfully!", "output_pdf": updated_document})




@app.route('/ask', methods=['POST'])
def ask_question():
    """Handles user speech queries and returns AI-generated responses."""
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print("Say something...")
        recognizer.adjust_for_ambient_noise(source)  
        audio = recognizer.listen(source)

        try:
            text = recognizer.recognize_google(audio)   
            print("You said:", text)
            query = text  
            try:
                docs = asyncio.run(fetch_docs(query))  
                retrieved_texts = [doc.page_content for doc in docs] if docs else []
            except Exception as e:
                print(f"Error retrieving documents: {e}")
                retrieved_texts = []  

            response = generate_openai_response(query)  

            # Compute faithfulness 
            if retrieved_texts:
                faithfulness_result = check_faithfulness(query, response, retrieved_texts)
            else:
                faithfulness_result = {"faithfulness_score": 0, "message": "No relevant documents found."}

            print(f"Faithfulness Result: {faithfulness_result}")

            evaluate_bert_score(response, retrieved_texts, model_type= "bert-base-uncased")
            bert_score_result = evaluate_bert_score(response, retrieved_texts)
            print(bert_score_result)


            engine.say(response)
            engine.runAndWait()

            return jsonify({
                "response": response,
                "faithfulness_score": faithfulness_result["faithfulness_score"],
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


from bert_score import score

def evaluate_bert_score(response, retrieved_docs, model_type="bert-base-uncased"):
    
    if not retrieved_docs:
        return {"precision": 0, "recall": 0, "f1": 0, "message": "No retrieved documents!"}
    reference_text = " ".join(retrieved_docs).strip()
    
    if not reference_text:
        return {"precision": 0, "recall": 0, "f1": 0, "message": "Retrieved documents are empty!"}

    # Compute BERTScore
    P, R, F1 = score([response], [reference_text], model_type=model_type, lang="en", rescale_with_baseline=True)

    return {
        "precision": round(P.tolist()[0], 4),
        "recall": round(R.tolist()[0], 4),
        "f1": round(F1.tolist()[0], 4),
        "response": response,
        "retrieved_docs": retrieved_docs
    }
 


# Start Flask app
if __name__ == '__main__':
    if not os.path.exists("uploads"):
        os.makedirs("uploads")
    app.run(host="0.0.0.0", port=5000, debug=True)

