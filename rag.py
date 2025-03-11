import os
import numpy as np
import torch
from annoy import AnnoyIndex
import os

from llama_index.core import SimpleDirectoryReader

from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM


documents = SimpleDirectoryReader("C:\\Codes\\Langchain\\see").load_data()

texts = [doc.text for doc in documents]


embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
embed_tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
embed_model = AutoModel.from_pretrained(embedding_model_name)

def generate_embedding(text):

    inputs = embed_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = embed_model(**inputs)
    # Average pooling over the token embeddings
    embedding = outputs.last_hidden_state.mean(dim=1)
    return embedding[0].numpy()

# Compute embeddings for each document chunk
document_embeddings = [generate_embedding(text) for text in texts]
document_embeddings = np.array(document_embeddings).astype('float32')

embedding_dim = document_embeddings.shape[1]
annoy_index = AnnoyIndex(embedding_dim, metric='euclidean')  

# Add embeddings to the Annoy index
for i, vector in enumerate(document_embeddings):
    annoy_index.add_item(i, vector.tolist())


num_trees = 10  # Increasing this number improves accuracy at the cost of indexing time.
annoy_index.build(num_trees)

query = "give me some helpul email or links for contact   ?"
query_embedding = generate_embedding(query)

# Retrieve top k closest chunks using Annoy
k = 10
indices = annoy_index.get_nns_by_vector(query_embedding.tolist(), k, include_distances=False)
retrieved_texts = [texts[i] for i in indices]
context = " ".join(retrieved_texts)


from groq import Groq


client = Groq(api_key="")  # <-- Replace with your actual API key

def answer(query, document_content):

    prompt = (
        f"You are helpful assiant who can find relevant passage and answer from the text based on user query:\n{document_content}\n\n"
        f"User: {query}\nAssistant:"
    )
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile"
    )
    return chat_completion.choices[0].message.content


result = answer(query, context)



print("Query:")
print(query)
print("\nRetrieved Context:")
print(context)
print("\nGenerated Answer:")
print(result)

