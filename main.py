import os
import PyPDF2
from flask import Flask, request, jsonify
from pinecone import Pinecone, ServerlessSpec
from openai.embeddings_utils import get_embedding
import openai

app = Flask(__name__)

openai.api_key = "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"

# Specify the path to the directory containing PDF files
docs_path = "C:/Users/JatinSahni/OneDrive - inmorphis.com/Desktop/OpenAI/Pinecone/data"

text_chunks = []
for f_name in os.listdir(docs_path):
    doc_path = os.path.join(docs_path, f_name)
    if doc_path.endswith('.pdf'):  # Check if the file is a PDF file
        with open(doc_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            num_pages = len(pdf_reader.pages)
            for page_num in range(num_pages):
                page = pdf_reader.pages[page_num]
                text_chunks.append({"text": page.extract_text(), "filename": f_name})

# Remove all chunks shorter than 10 words and strip the rest
text_chunks = [{"text": string["text"].strip().strip('\n'), "filename": string["filename"]} for string in text_chunks if len(string["text"].split()) >= 10]

# Generate embeddings for text chunks
text_embeddings = [get_embedding(chunk["text"], engine='text-embedding-ada-002') for chunk in text_chunks]

# Convert embeddings into the required format for Pinecone
formatted_embeddings = [{"id": str(i), "values": embedding, "metadata": {"filename": chunk["filename"], "text": chunk["text"]}} for i, (chunk, embedding) in enumerate(zip(text_chunks, text_embeddings))]

pc = Pinecone(api_key="d1726c1a-9707-4a4a-9f7d-1d3dfcf1bb07")  # Create an instance of Pinecone

# Create or connect to index
index_name = "servicenow"

if "servicenow" not in pc.list_indexes().names():
    pc.create_index(
        name='servicenow',
        dimension=1536,
        metric='cosine',
        spec=ServerlessSpec(cloud='gcp', region='us-central1')
    )

index = pc.Index("servicenow", host="https://servicenow-1e5aki9.svc.gcp-starter.pinecone.io")  # Connect to index

# Index text embeddings into Pinecone
index.upsert(formatted_embeddings)

def search_docs(query):
    xq = openai.Embedding.create(input=query, engine="text-embedding-ada-002")['data'][0]['embedding']
    res = index.query(vector=[xq], top_k=5, include_metadata=True)
    matches = []
    for match in res['matches']:
        matches.append(match['metadata'])
    print("Matches:", matches)
    return matches

def construct_prompt(query, conversation_history):
    matches = search_docs(query)
    chosen_text = ""
    for match in matches:
        chosen_text += match['text'] + "\n"
    print("Chosen Text:", chosen_text)

    # Add conversation history to the context
    context_with_history = f"{chosen_text}\n\nPrevious Conversation:\n{conversation_history}"

    prompt = """Answer the question as truthfully as possible using the context below, and if the answer is no within the context, say 'I don't know.'"""
    prompt += "\n\n"
    prompt += "Context: " + context_with_history
    prompt += "\n\n"
    prompt += "Question: " + query
    prompt += "\n"
    prompt += "Answer: "
    print("Constructed Prompt:", prompt)
    return prompt

def answer_question(query, conversation_history):
    prompt = construct_prompt(query, conversation_history)
    try:
        res = openai.Completion.create(
            prompt=prompt,
            model="gpt-3.5-turbo-instruct",
            max_tokens=500,
            temperature=0.0,
        )
        answer = res.choices[0].text.strip()
    except openai.error.InvalidRequestError as e:
        print(f"Error: {e.message}")
        answer = None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        answer = None

    return answer

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.get_json()
    user_question = data['question']
    conversation_history = data.get('conversation_history', [])  # Get existing conversation history or initialize as empty list

    answer = answer_question(user_question, conversation_history)

    # Append the current interaction to the conversation history
    conversation_history.append({"user_question": user_question, "bot_answer": answer})

    # Keep only the last 3 conversational exchanges
    if len(conversation_history) > 3:
        conversation_history = conversation_history[-3:]

    response = {
        "answer": answer,
        "conversation_history": conversation_history
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)

