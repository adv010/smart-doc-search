import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
import cohere
import spacy
import tiktoken
from utils.text_processor import load_manpages, chunk_manpages, analyze_tokens
import pinecone
from pinecone import Pinecone, ServerlessSpec
import pdb
import gradio as gr
co = cohere.ClientV2(os.environ["COHERE_API_KEY"])
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

def embed_chunks(chunks, batch_size=96):
    embedded_chunks = []
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        texts = [chunk["text"] for chunk in batch]
        
        try:
            response = co.embed(
                texts=texts,
                model="embed-english-v3.0",
                input_type="search_document",
                embedding_types=["float"]
            )
            embeddings = response.embeddings.float_
            print(len(embeddings))
            
            for j, chunk in enumerate(batch):
                chunk["embedding"] = response.embeddings.float_[j]
                embedded_chunks.append(chunk)
                
            print(f"Processed batch {i//batch_size + 1}/{(len(chunks)//batch_size)+1}")
            
        except Exception as e:
            print(f"Error in batch {i//batch_size + 1}: {str(e)}")
            # Optionally: Save failed batch indices for retry
    
    return embedded_chunks


def retrieve_relevant_chunks(query, index, co, top_k=3):  # Embedding the query
    # Embed the query
    query_embed = co.embed(
        texts=[query],
        model="embed-english-v3.0",
        input_type="search_query",
        embedding_types=["float"]
    ).embeddings.float_[0]
    
    # Query Pinecone
    results = index.query(
        vector=query_embed,
        top_k=top_k,
        include_metadata=True
    )
    return [match.metadata for match in results.matches]

def generate_answer(user_query, context_chunks, co):
    context = "\n\n".join([chunk["text"] for chunk in context_chunks])
    response = co.chat(
        model="command-a-03-2025",
        messages=[{"role": "user", "content": f"""Answer the question based ONLY on the following manpage context:
        {context}
        Question: {user_query}"""}],
        temperature=0.6
    )
    return response


def handle_user_query(user_query):
    index_name = "manpages-rag"
    index = pc.Index(index_name)

    index_matches = retrieve_relevant_chunks(user_query, index, co)
    answer = generate_answer(user_query, index_matches, co)
    # pdb.set_trace()
    print(answer.message.content[0].text)
    raw_text = "".join([c.text for c in answer.message.content if c.type == "text"])

    sources = "\n".join(
        f"**{i+1}. From `{chunk['command']}` manpage:**\n{chunk['text'][:300]}..."
        for i, chunk in enumerate(index_matches)
    )

    return f"### Answer\n{raw_text}\n\n### Sources\n{sources}"


def main():
    nlp = spacy.load('en_core_web_sm')
    index_name = "manpages-rag"

    if index_name not in pc.list_indexes().names():
        print(f"Creating Pinecone index: {index_name}")
        pc.create_index(
            name=index_name,
            dimension=1024,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

    index = pc.Index(index_name)

    if index.describe_index_stats().total_vector_count == 0:
        manpages = load_manpages('./manpages')
        text_chunks = chunk_manpages(nlp, manpages)
        embedded_chunks = embed_chunks(text_chunks)
        vectors = [
            {
                "id": f"{chunk['command']}-{chunk['chunk_id']}",
                "values": chunk["embedding"],
                "metadata": {"text": chunk["text"], "command": chunk["command"]}
            } for chunk in embedded_chunks
        ]
        index.upsert(vectors=vectors)
    else:
        print("Using existing Pinecone vectors.")

    demo = gr.Interface(
        fn=handle_user_query,
        inputs=gr.Textbox(label="Enter your manpage-related question", lines=2),
        outputs=gr.Markdown(label="Response"),
        title="Man Page QA Chat",
        description=" Your personal assistant for Linux Man pages",
        allow_flagging=False
    )
    demo.launch()

if __name__ == "__main__":
    main()
