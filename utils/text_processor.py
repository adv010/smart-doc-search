from pathlib import Path
import glob

# Load spaCy for sentence splitting (better than raw chunking)

def load_manpages(path):
    texts = []
    for filepath in glob.glob(f"{path}/*.txt"):
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read().strip()
            if text:
                texts.append((Path(filepath).stem, text))  # (command_name, content)
    return texts


def chunk_manpages(nlp, texts, max_chunk_size=500):
    chunks = []
    for name, text in texts:
        doc = nlp(text)
        current_chunk = []
        current_len = 0
        
        for sent in doc.sents:
            sent_len = len(sent.text)
            if current_len + sent_len > max_chunk_size and current_chunk:
                chunks.append({
                    "command": name,
                    "text": " ".join(current_chunk),
                    "chunk_id": len(chunks)
                })
                current_chunk = []
                current_len = 0
            current_chunk.append(sent.text)
            current_len += sent_len
        
        if current_chunk:  # Add remaining text
            chunks.append({
                "command": name,
                "text": " ".join(current_chunk),
                "chunk_id": len(chunks)
            })
    return chunks


def analyze_tokens(encoder,chunks):
    for chunk in chunks:
        chunk["token_count"] = len(encoder.encode(chunk["text"]))
        # print(f"{chunk['command']} - Chunk {chunk['chunk_id']}: {chunk['token_count']} tokens")


# def main():
#     nlp = spacy.load('en_core_web_sm')

# if __name__ == "__main__":
#     main()