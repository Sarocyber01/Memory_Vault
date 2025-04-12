import os
os.environ["GOOGLE_API_KEY"] = "AIzaSyDjJLYc0OeKDYspyyI-u7SRqe47hJsHzO0"
from pinecone import Pinecone
import google.generativeai as genai

# # Now, run the embedding function
# result = genai.embed_content(
#     model="models/text-embedding-004",
#     content="What is the meaning of life?",
#     task_type="retrieval_document",
#     title="Embedding of single string"
# )

# # 1 input > 1 vector output
# # print(str(result['embedding'])[:50], '... [TRIMMED]')
# print(result)


from pinecone import Pinecone

pc = Pinecone(api_key="4f448d11-d408-4d69-bb9d-3519de8235d7")
index = pc.Index("datavault")

response = index.query(
    namespace="ns1",
    vector=[0.1, 0.3],
    top_k=2,
    include_values=True,
    include_metadata=True,
    filter={"genre": {"$eq": "action"}}
)
    
print(response)