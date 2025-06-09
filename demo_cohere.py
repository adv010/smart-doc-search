import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
import cohere
import spacy
import tiktoken
from utils.text_processor import load_manpages, chunk_manpages, analyze_tokens
import pinecone
from pinecone import Pinecone, ServerlessSpec
import pdb
co = cohere.ClientV2("4dkVC224oH8uSqKsYCAypJ6fOwxdsfUqQux2Rf1B")
pc = Pinecone(api_key="pcsk_jgdQw_9ezxffUEfvzvvUErGckMcRD3T746MD7hmkK2UVTG8NrUvMpXFeVA887XHJ9sNrD")


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


def retrieve_relevant_chunks(query, index, co, top_k=3):
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

def generate_answer(query, context_chunks, co):
    context = "\n\n".join([chunk["text"] for chunk in context_chunks])
    response = co.chat(
        model="command-a-03-2025",
        messages=[{"role": "user", "content": f"""Answer the question based ONLY on the following manpage context:
        {context}
        Question: {query}"""}],
        temperature=0.3
    )
    return response



def query_examples(index, co):
    examples = [
        ("brew", "What is the description of the brew package?"),
        ("grep", " What is the prefix  path in which Homebrew is installed"),
    ]
    
    for cmd, query in examples:
        print(f"\n=== Query about '{cmd}': {query} ===")
        chunks = retrieve_relevant_chunks(query, index, co)
        answer = generate_answer(query, chunks, co)
        print(f"Answer: {answer}")
        print("\nSources:")
        for i, chunk in enumerate(chunks, 1):
            print(f"{i}. From {chunk['command']} manpage:\n{chunk['text'][:200]}...")


def main():
    nlp = spacy.load('en_core_web_sm')
    encoder = tiktoken.get_encoding("cl100k_base")
    manpages = load_manpages('/Users/adv8a3ya/Projects/smart-doc-search/manpages')
    text_chunks = chunk_manpages(nlp,manpages)
    print(f"Total chunks created: {len(text_chunks)}")
    analyze_tokens(encoder,text_chunks)
    embedded_chunks = embed_chunks(text_chunks)
    print(f"Total chunks with embeddings: {len(embedded_chunks)}")
    index_name = "manpages-rag"
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=1024,  # Cohere v3 embeddings are 1024-dim
            metric="cosine",
            spec=ServerlessSpec(
    cloud="aws",
    region="us-east-1",
  )
        )

    index = pc.Index(index_name)
    # Prepare vectors for upsert
    vectors = []
    for chunk in embedded_chunks:
        vectors.append({
            "id": f"{chunk['command']}-{chunk['chunk_id']}",
            "values": chunk["embedding"],
            "metadata": {
                "text": chunk["text"],
                "command": chunk["command"]
            }
        })

    # Upsert to Pinecone
    index.upsert(vectors=vectors)
    query_examples(index, co)


if __name__ == "__main__":
    main()


