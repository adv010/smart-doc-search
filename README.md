# smart-doc-search Chatbot 
An interactive chatbot that helps developers understand manual pages (`man` documents) by allowing them to "chat" with the documentation. This tool leverages a Cohere LLM (free-tier) and Pinecone (free-tier vector database) to parse and retrieve relevant information from manuals, making it easier to comprehend flags, options, and usage details.

## Features  

- **Document Parsing**: Extracts and indexes information from manual pages  
- **Interactive Chat**: Ask questions about commands, flags, and usage in natural language  
- **RAG-Powered**: Uses Retrieval-Augmented Generation (RAG) for accurate, context-aware responses  
- **Free-Tier Stack**: Built with Cohere's free LLM and Pinecone's free vector database  
- **Gradio UI**: Simple and user-friendly interface for seamless interaction  

## Installation  

1. Clone the repository:  
   ```sh
   git clone <repository-url>
   cd <repository-directory>

Sources:
1. https://docs.cohere.com/v2/docs/rag-with-cohere#basic-rag
2. https://wandb.ai/mostafaibrahim17/ml-articles/reports/Vector-Embeddings-in-RAG-Applications--Vmlldzo3OTk1NDA5
3. https://colab.research.google.com/drive/1W0-Sy34TkJSQWnbUCn3CGZwU3uuuSSyE?usp=sharing#scrollTo=HZmvUReEFnmz