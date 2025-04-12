# import os
# from langchain_pinecone import PineconeVectorStore
# from langchain_openai import OpenAIEmbeddings
# from langchain_community.document_loaders import TextLoader
# from langchain_text_splitters import CharacterTextSplitter

# import uuid

# # load env vars with dot-env
# from dotenv import load_dotenv

# from langchain_openai import ChatOpenAI
# from langchain.chains import RetrievalQA

# load_dotenv()  # take environment variables from .env.

# from langchain.docstore.document import Document


# def get_vectorstore():
#     index_name = "memory-vault"
#     embeddings = OpenAIEmbeddings(
#         model="text-embedding-3-small", openai_api_key=os.environ["OPENAI_API_KEY"]
#     )
#     vectorstore = PineconeVectorStore(
#         index_name=index_name,
#         embedding=embeddings,
#         pinecone_api_key=os.environ["PINECONE_API_KEY"],
#     )

#     return vectorstore


# def add_document_to_pinecone(text: str):
#     new_doc = Document(page_content=text)

#     # do chunking for this new doc
#     text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
#     chunks = text_splitter.split_documents([new_doc])

#     # generate unique id for this doc that's shared across the chunks
#     document_id = str(uuid.uuid4())
#     for chunk in chunks:
#         chunk.metadata["document_id"] = document_id

#     # get the vectorstore
#     vectorstore = get_vectorstore()

#     vectorstore.add_documents(chunks)


# # the pinecone index stores "memories" = pieces of text
# # this function uses an llm to query the index about memories
# def get_llm_response(query: str):
#     # get the vectorstore
#     vectorstore = get_vectorstore()

#     # completion llm with GPT-2
#     llm = ChatOpenAI(
#         openai_api_key=os.environ["OPENAI_API_KEY"],
#         model_name="gpt-2",
#         temperature=0.2,
#     )
#     qa = RetrievalQA.from_chain_type(
#         llm=llm,
#         chain_type="stuff",
#         retriever=vectorstore.as_retriever(k=10),
#     )


#     res = qa.run(query)
#     return res


# from flask import Flask, jsonify, request
# from flask_cors import CORS

# app = Flask(__name__)
# CORS(app)


# @app.route("/postMemory")
# def post_memory():
#     text = request.args.get("text")
#     if not text:
#         return "No text provided", 400
#     try:
#         add_document_to_pinecone(text)
#         return "success"
#     except Exception as e:
#         error_message = f"Error adding document to Pinecone: {e}"
#         print(error_message)
#         if "insufficient_quota" in str(e):
#             return jsonify({"error": "Insufficient quota for OpenAI API. Please check your plan and billing details."}), 429
#         return jsonify({"error": "Error saving memory"}), 500



# DALLE3_PROMPT = "you are an ai that helps people with alzheimers visualize their past memories. visualize the given memory  with a photo. make it obviously a sketch as opposed to photorealistic, but an artistic one. capture the spirit of the memory. only use drawings, no words or text"

# from openai import OpenAI


# @app.route("/generateImage")
# def generate_dalle3_image(
#     image_dimension="1024x1024",
#     image_quality="hd",
#     model="dall-e-3",
#     nb_final_image=1,
# ):

#     prompt = DALLE3_PROMPT + request.args.get("llmResponse")

#     # Instantiate the OpenAI client
#     client = OpenAI()

#     response = client.images.generate(
#         model=model,
#         prompt=prompt,
#         size=image_dimension,
#         quality=image_quality,
#         n=nb_final_image,
#     )

#     image_url = response.data[0].url

#     return image_url
#     # return "null"


# # System prompt to give some more guidance
# PROMPT = """
# You are an AI meant to help Alzheimer's patients remember their memories. An example question thhey might ask: "Tell me about a time I felt fulfilled."

# They could ask about some more detail for a memory that they remember a little of.

# Be kind and considerate.

# USE AS MUCH DETAIL AS POSSIBLE. you want them to feel like they are living there again.

# Respond in the second person.

# Make it vivid and paraphrase. 

# REMEMBER THE INFORMATION THAT THE USER TELLS YOU TO.

# Do NOT
# - mention anything about you being an AI.
# - mention anything about context. 
# - make up ANY FALSE INFORMATION.

# If you can't find any relevant memories, tell them to go to the add memory page and have them or a family member add a memory.


# act like a human.

# """


# @app.route("/query")
# def hello_world():
#     query = request.args.get("query")

#     query = PROMPT + query

#     print(query)

#     llm_res = get_llm_response(query)
#     print(llm_res)

#     return llm_res


# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=8080)
import os
import uuid
import google.generativeai as genai
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import TextLoader
from langchain.docstore.document import Document
from langchain_text_splitters import CharacterTextSplitter
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS

# Load environment variables
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index("datavault")
# Initialize the app
app = Flask(__name__)
CORS(app)

# Set up Gemini API key
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# DALLE3 Prompt for generating images
DALLE3_PROMPT = (
    "You are an AI that helps people with Alzheimer's visualize their past memories. "
    "Visualize the given memory with a photo. Make it obviously a sketch as opposed to photorealistic, "
    "but an artistic one. Capture the spirit of the memory. Only use drawings, no words or text."
)


# # Function to generate embeddings using Gemini
# def get_embeddings(text: str): # ORIGINAL
#     try:
#         # Generate embeddings using Gemini
#         result = genai.embed_content(
#             model="models/text-embedding-004",
#             content=text,
#             task_type="retrieval_document",
#             title="Embedding of a single string"
#         )
#         # Debug: print the result structure
#         print(f"Gemini embedding result: {result}")
#         return result['embedding']
#     except Exception as e:
#         print(f"Error generating embeddings: {e}")

# Function to generate embeddings using Gemini
def get_embeddings(text: str):
    try:
        # Generate embeddings using Gemini API
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=text,
            task_type="retrieval_document",
            title="Embedding of a single string"
        )
        # Debugging: Ensure the response structure is correct
        print(f"Gemini embedding is done")
        return result.get('embedding')  # Use get to safely retrieve 'embedding'
    except Exception as e:
        # Provide detailed error log and handling
        print(f"Error generating embeddings: {e}")
        if "rate_limit_exceeded" in str(e):
            print("Rate limit exceeded for Gemini API.")
        elif "invalid_model" in str(e):
            print("Invalid model specified.")
        return None


# Function to add document to vector store
# ----------------------------------------------------------------------------------------

# def add_document_to_pinecone(text: str):      // ORIGINAL
#     chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
#     document_id = str(uuid.uuid4())

#     for chunk in chunks:
#         embedding = get_embeddings(chunk)
#         if embedding:
#             # Upsert the document into Pinecone
#             index.upsert(vectors=[(document_id, embedding)], metadata={"text": chunk})
#             print(f"Document {document_id} added to vector store.")
#         else:
#             print(f"Failed to add chunk: {chunk[:50]}...") 

def add_document_to_pinecone(text: str):
    # -----------------------------------
    # # Break text into chunks of 1000 characters for embedding
    # chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
    # document_id = str(uuid.uuid4())  # Generate a unique document ID

    # for idx, chunk in enumerate(chunks):
    #     embedding = get_embeddings(chunk)
    #     if embedding:
    #         # Upsert each chunk with metadata including its chunk index
    #         index.upsert(vectors=[(document_id + f"_{idx}", embedding)], 
    #                      metadata={"text": chunk, "chunk_index": idx})
    #         print(f"Document {document_id}_{idx} added to vector store.")
    #     else:
    #         print(f"Failed to add chunk {idx}: {chunk[:50]}...")
    # --------------------------
    new_doc = Document(page_content=text)

    # do chunking for this new doc
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    chunks = text_splitter.split_documents([new_doc])

    # generate unique id for this doc that's shared across the chunks
    document_id = str(uuid.uuid4())
    for chunk in chunks:
        chunk.metadata["document_id"] = document_id

    # get the vectorstore
    vectorstore = get_vectorstore()

    vectorstore.add_documents(chunks)

# ------------------------------------------------------------
def get_vectorstore(query: str):
    index_name = "datavault"
    embeddings = genai.embed_content(model="models/embedding-001",
                              content=query,
                              task_type="retrieval_query")
    vectorstore = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings,
        pinecone_api_key=os.environ["PINECONE_API_KEY"],
    )

    return vectorstore

# def fetch_memory_text_by_id(memory_id):
#     """Fetch memory text based on the provided memory ID."""
#     return memory_storage.get(memory_id, "No memory found.")

def get_llm_response(query: str):
    vectorstore = get_vectorstore(query)
    embeddings = genai.embed_content(
        model="models/embedding-001",
        content=query,
        task_type="retrieval_query"
    )
    results = vectorstore.query(embeddings['embedding'], top_k=10)

    # Construct context based on fetched memory texts
    context = ""
    for match in results['matches']:
        memory_text = fetch_memory_text_by_id(match['id'])
        context += memory_text + "\n"

    # Proceed to generate the response
    response = genai.chat(
        model="models/chat-bison-001",
        messages=[
            {"role": "user", "content": context},
            {"role": "user", "content": query}
        ]
    )
    return response['choices'][0]['message']['content'] if response['choices'] else "No response."



# -------------------------------------------------------

# Function to generate LLM response using Gemini
# def get_llm_response(query: str):
#     try:
#         # Replace this with appropriate Gemini model usage if available
#         # For now, this is a placeholder to show where the LLM call would go
#         response = genai.generate_text(
#             model="models/text-davinci-003",
#             prompt=query,
#             max_tokens=150
#         )
#         return response['generated_text']
#     except Exception as e:
#         print(f"Error generating LLM response: {e}")
#         return "Error generating response."


# # Route to post memory   //  ORIGINAL
# @app.route("/postMemory")
# def post_memory():
#     text = request.args.get("text")
#     if not text:
#         return "No text provided", 400

#     try:
#         add_document_to_pinecone(text)
#         return "success"
#     except Exception as e:
#         error_message = f"Error adding document to Pinecone: {e}"
#         print(error_message)
#         if "insufficient_quota" in str(e):
#             return jsonify({"error": "Insufficient quota for Gemini API. Please check your plan and billing details."}), 429
#         return jsonify({"error": "Error saving memory"}), 500

@app.route("/postMemory")
def post_memory():
    text = request.args.get("text")
    if not text:
        return "No text provided", 400

    try:
        add_document_to_pinecone(text)
        return "Memory saved successfully", 200
    except Exception as e:
        error_message = f"Error adding document to Pinecone: {e}"
        print(error_message)
        if "insufficient_quota" in str(e):
            return jsonify({"error": "Insufficient quota for Gemini API. Please check your plan and billing details."}), 429
        return jsonify({"error": "Error saving memory"}), 500

# Route to generate image
@app.route("/generateImage")
def generate_dalle3_image(image_dimension="1024x1024", image_quality="hd", model="dall-e-3", nb_final_image=1):
    prompt = DALLE3_PROMPT + request.args.get("llmResponse")

    # Instantiate the OpenAI client
    client = genai.OpenAI()

    response = client.images.generate(
        model=model,
        prompt=prompt,
        size=image_dimension,
        quality=image_quality,
        n=nb_final_image,
    )

    image_url = response.data[0].url

    return image_url


# System prompt to give some more guidance
PROMPT = """
You are an AI meant to help Alzheimer's patients remember their memories. An example question they might ask: "Tell me about a time I felt fulfilled."

They could ask about some more detail for a memory that they remember a little of.

Be kind and considerate.

USE AS MUCH DETAIL AS POSSIBLE. you want them to feel like they are living there again.

Respond in the second person.

Make it vivid and paraphrase.

REMEMBER THE INFORMATION THAT THE USER TELLS YOU TO.

Do NOT
- mention anything about you being an AI.
- mention anything about context. 
- make up ANY FALSE INFORMATION.

If you can't find any relevant memories, tell them to go to the add memory page and have them or a family member add a memory.

act like a human.
"""

@app.route("/query")
def hello_world():
    query = request.args.get("query")
    query = PROMPT + query

    print(query)

    llm_res = get_llm_response(query)
    print(llm_res)

    return llm_res


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
