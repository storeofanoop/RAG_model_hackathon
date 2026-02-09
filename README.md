#  RAG system using local LLMs to power semantic search over Confluence knowledge.

# What it does:

* Ingests documents (tested using Wikipedia pages for few countries)
* Converts content into embeddings
* Stores them in a vector database
* Retrieves the most relevant chunks for a user query
* Uses a local LLM  to generate grounded answers


# RAG flow diagram
<img width="950" height="577" alt="image" src="https://github.com/user-attachments/assets/98eec129-ad99-4bef-b913-3b5e5180778a" />



# Tech Stack Used

* Python for ingestion and orchestration
* Sentence-Transformers for embeddings
* FAISS as the vector database
* Ollama local LLM (Mistral) for answer generation
* Streamlit for a web UI

# Usage details

Currently i am using 3 wikipedia pages. you can replace it with any other pages. 

# How to implement

Install python and following libraries in your mac system (I have tested it in Mac only), it can also be implemented in windows as well.
* streamlit
* sentence-transformers
* faiss-cpu
* requests
* beautifulsoup4
* tqdm

Download Ollama https://ollama.com and a lightweight model
* ollama pull mistral

Create data ingestion file (rag_ingest_api_nw.py) for following operations
* It Fetches document
* Convert content into chunks
* Generate embeddings for those chunks
* Store embeddings into vector database.

Run the data indestion file (One time Job)
* python rag_ingest_api_nw.py

Create online query pipeline file (rag_app.py) for following operations
* It will present a UI
* User enters question via UI
* Question is converted into and embedding
* Relevant chunks are retrieved from the chunks
* Answer will be displayed

Run the query pipeline file
* streamlit run rag_app.py


# Some demo Output

* satisfactory results
  
<img width="848" height="669" alt="image" src="https://github.com/user-attachments/assets/99b1b471-f769-4810-8cfe-2f7dd87c834d" />

* satisfactory results
  
<img width="834" height="745" alt="image" src="https://github.com/user-attachments/assets/0a08d9c1-befc-45df-a000-408eb14dfc00" />

* satisfactory results
  
<img width="842" height="683" alt="image" src="https://github.com/user-attachments/assets/a871678a-eaf1-48bb-aa90-3d06d2feefff" />

* Not very good compare to cloud LLM

<img width="848" height="705" alt="image" src="https://github.com/user-attachments/assets/3105e184-49fa-4972-88ba-c350f1f4bcd8" />





