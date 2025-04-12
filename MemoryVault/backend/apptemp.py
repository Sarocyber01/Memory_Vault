import os
import uuid
import pinecone
import google.generativeai as genai
from pinecone import Pinecone, ServerlessSpec
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS

 
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

index_name = "memory-vault"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=768,  
        metric="cosine",  
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1" 
        )
    )
index = pc.Index(index_name)

app = Flask(__name__)
CORS(app)

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

def get_embeddings(text: str):
    try:
        result = genai.embed_content(
            model="models/text-embedding-004",  
            content=text,
            task_type="retrieval_document"
        )
        return result['embedding']
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return None

def add_document_to_pinecone(text: str):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    chunks = text_splitter.split_text(text)

    document_id = str(uuid.uuid4())
    for i, chunk in enumerate(chunks):
        embedding = get_embeddings(chunk)
        if embedding:
            
            index.upsert([
                {
                    'id': f"{document_id}_{i}",
                    'values': embedding,
                    'metadata': {'document_id': document_id, 'text': chunk}
                }
            ])

def generate_gemini_embedding(text: str):
    model = genai.GenerativeModel(model_name="gemini-embedding-1")
    embedding = model.embed_text(text)
    return embedding

def get_vectorstore():
    return index

def add_to_vectorstore(text: str, metadata: dict = None):
    try:
        embedding = generate_gemini_embedding(text)
        index.upsert([{
            'id': text,
            'values': embedding,
            'metadata': metadata if metadata else {}
        }])
        print("Document added successfully!")
    except Exception as e:
        print(f"Error adding document to Pinecone: {e}")

def get_embeddings_for_query(query: str):
    """Generate embeddings for the query using Gemini."""
    return get_embeddings(query) 

def get_llm_response(query: str):
    """Retrieve relevant documents from Pinecone and generate a response using Gemini."""
    
    query_embedding = get_embeddings_for_query(query)
    
    if not query_embedding:
        return "Error generating query embeddings"
    
    response = index.query(
        vector=query_embedding,
        top_k=10,  # Retrieve top 10 most similar documents
        include_metadata=True
    )

    if 'matches' not in response:
        return "No relevant documents found."

    # Step 3: Extract the retrieved text chunks and join them into context
    context = " ".join([match['metadata']['text'] for match in response['matches']])

    if not context:
        return "No context available to generate a response."

    # Step 4: Use Gemini to generate a response using the retrieved context
    model = genai.GenerativeModel(model_name="gemini-1.5-flash")
    llm_response = model.generate_content(context) 

    try:
        response_text = llm_response.text 
    except AttributeError:
        return "Error retrieving response text from LLM."

    # print(response_text)
    return response_text


@app.route("/postMemory", methods=["POST"])
def post_memory():
    print("The function started !! velaiyaa arambii....")
    data = request.get_json()
    text = data.get("text")
    if not text:
        return jsonify({"error": "No text provided"}), 400
    try:
        print("From now it ok to start function")
        add_document_to_pinecone(text)
        print("Added successfully")
        print("Velai mudujathuu")
        return jsonify({"message": "Memory saved successfully"}), 200
    except Exception as e:
        return jsonify({"error": f"Error saving memory: {e}"}), 500

@app.route("/query", methods=["GET"])
def query_memory():
    query = request.args.get("query")
    if not query:
        return jsonify({"error": "No query provided"}), 400

    llm_res = get_llm_response(query)
    return jsonify({"response": llm_res})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
