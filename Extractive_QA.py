import json
import torch
import ipywidgets as widgets
from IPython.display import display
from transformers import pipeline
from sentence_transformers import SentenceTransformer, CrossEncoder
from google.colab import files

# --- 1. Data Uploading ---
uploaded = files.upload()

# --- 2. Data Loading & Parsing ---
def load_squad_contexts(path):
    contexts = []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)["data"]
    
    for article in data:
        for paragraph in article["paragraphs"]:
            contexts.append(paragraph["context"])
            
    return list(set(contexts))

# Load your SQuAD dataset paths
train_contexts = load_squad_contexts("train-v1.1.json")
val_contexts = load_squad_contexts("dev-v1.1.json")

documents = train_contexts + val_contexts
print("Total documents:", len(documents))

# --- 3. Model Initialization ---
retriever = SentenceTransformer("all-MiniLM-L6-v2")

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

qa_model = pipeline(
    "question-answering",
    model="distilbert-base-uncased-distilled-squad",
    device=0 if torch.cuda.is_available() else -1
)

print("Models loaded.")

# --- 4. Document Embedding Generation ---
# Generate embeddings for all context documents
doc_embeddings = retriever.encode(documents, show_progress_bar=True, convert_to_tensor=True)

# --- 5. Extractive QA Pipeline Implementation ---
def get_answer(question, top_k_retrive=10, top_k_rerank=3):
    # Step 1: Retrieve top candidate contexts using Bi-Encoder
    query_embedding = retriever.encode(question, convert_to_tensor=True)
    cos_scores = torch.nn.functional.cosine_similarity(query_embedding, doc_embeddings)
    top_results = torch.topk(cos_scores, k=top_k_retrive)
    
    retrieved_docs = [documents[idx] for idx in top_results.indices]
    
    # Step 2: Rerank the retrieved contexts using Cross-Encoder
    rerank_pairs = [[question, doc] for doc in retrieved_docs]
    rerank_scores = reranker.predict(rerank_pairs)
    
    # Sort documents by cross-encoder score
    reranked_indices = torch.argsort(torch.tensor(rerank_scores), descending=True)
    best_contexts = [retrieved_docs[idx] for idx in reranked_indices[:top_k_rerank]]
    
    # Step 3: Extract answers using the Reader Model (Transformer)
    answers = []
    for context in best_contexts:
        res = qa_model(question=question, context=context)
        answers.append({
            "answer": res["answer"],
            "score": res["score"],
            "context": context
        })
        
    # Select the answer with the highest Reader confidence score
    best_answer = max(answers, key=lambda x: x["score"])
    return best_answer

# --- 6. Interactive User Interface ---
# Creating widgets for interactive question answering
input_text = widgets.Text(
    value='',
    placeholder='Ask anything from dataset...',
    description='Question:',
    layout=widgets.Layout(width='90%')
)

search_btn = widgets.Button(
    description='Search Answer',
    button_style='success'
)

output_area = widgets.Output()

def on_click_search(b):
    with output_area:
        output_area.clear_output()
        query = input_text.value.strip()
        if not query:
            print("Please enter a valid question.")
            return
            
        print("Searching for answer... Please wait...\n")
        try:
            result = get_answer(query)
            print(f"🥇 Predicted Answer: {result['answer']}")
            print(f"🎯 Confidence Score: {result['score']:.4f}\n")
            print(f"📖 Context Context:\n... {result['context']} ...")
        except Exception as e:
            print(f"An error occurred: {e}")

search_btn.on_click(on_click_search)

# Display User Interface Components
display(widgets.VBox([input_text, search_btn, output_area]))
