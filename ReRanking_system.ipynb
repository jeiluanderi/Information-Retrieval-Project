# --- 1. Installation ---
!pip install datasets sentence-transformers rank-bm25 numpy torch scikit-learn
!pip install transformers

# --- 2. Imports ---
import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from rank_bm25 import BM25Okapi
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report

# --- 3. Model & Data Initialization ---
# Bi-Encoder for dense keyword/semantic retrieval
bi_encoder = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
# Cross-Encoder for high-accuracy score-based re-ranking
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

print("Loading dataset...")
dataset = load_dataset("ms_marco", "v1.1", split="train[:100]")

# --- 4. Hybrid Pipeline (Sparse BM25 + Dense Bi-Encoder) ---
def hybrid_reranking_pipeline(query, docs, top_k_retrieval=50):
    # --- STAGE 1: BM25 Retrieval ---
    tokenized_docs = [d.lower().split() for d in docs]
    bm25 = BM25Okapi(tokenized_docs)
    bm25_scores = np.array(bm25.get_scores(query.lower().split()))

    # --- STAGE 2: Semantic Retrieval (Bi-Encoder) ---
    query_emb = bi_encoder.encode(query, convert_to_tensor=True)
    doc_embs = bi_encoder.encode(docs, convert_to_tensor=True)
    semantic_scores = util.cos_sim(query_emb, doc_embs).cpu().numpy().flatten()

    # --- STAGE 3: Hybrid Combination & Reciprocal Scaling ---
    scaler = MinMaxScaler()
    bm25_norm = scaler.fit_transform(bm25_scores.reshape(-1, 1)).flatten()
    semantic_norm = scaler.fit_transform(semantic_scores.reshape(-1, 1)).flatten()

    # Weighted combination: 30% lexical keyword score + 70% semantic embedding score
    hybrid_scores = (0.3 * bm25_norm) + (0.7 * semantic_norm)
    top_indices = np.argsort(hybrid_scores)[::-1][:top_k_retrieval]
    candidate_docs = [docs[i] for i in top_indices]

    # --- STAGE 4: Final Re-ranking (Cross-Encoder) ---
    pairs = [[query, doc] for doc in candidate_docs]
    cross_scores = cross_encoder.predict(pairs)

    ranked_results = sorted(zip(candidate_docs, cross_scores), key=lambda x: x[1], reverse=True)
    return ranked_results

# --- 5. Pipeline Evaluation Loop ---
ranked_outputs = []
actual_labels = []

print("\nRunning pipeline...")
for i in range(20):
    query = dataset[i]["query"]
    passages = dataset[i]["passages"]["passage_text"]
    labels = dataset[i]["passages"]["is_selected"]

    # Execute search pipeline
    final_ranked = hybrid_reranking_pipeline(query, passages)
    ranked_outputs.append(final_ranked)
    actual_labels.append(labels)

    if i < 3:  # Preview first 3 query runs
        print(f"\nQuery: {query}")
        print(f"Top Result: {final_ranked[0][0][:100]}... (Score: {final_ranked[0][1]:.4f})")

# --- 6. Metrics Calculation (MRR) ---
def calculate_mrr(outputs, labels_list):
    rr_total = 0
    for ranked_docs, labels in zip(outputs, labels_list):
        for rank, (doc_text, score) in enumerate(ranked_docs):
            if any(doc_text == dataset[i]["passages"]["passage_text"][idx]
                   for i, lbls in enumerate(labels_list)
                   for idx, val in enumerate(lbls) if val == 1):
                rr_total += 1 / (rank + 1)
                break
    return rr_total / len(outputs)

mrr_final = calculate_mrr(ranked_outputs, actual_labels)
print(f"\n======================")
print(f"MRR: {mrr_final:.4f}")
print(f"======================")

# --- 7. Validation & Confusion Matrix Visualization ---
def plot_ir_confusion_matrix(outputs, dataset, num_queries=20):
    y_true = []
    y_pred = []

    for i in range(num_queries):
        passages = dataset[i]["passages"]["passage_text"]
        labels = dataset[i]["passages"]["is_selected"]
        relevance_map = {txt: lbl for txt, lbl in zip(passages, labels)}

        top_result_text = outputs[i][0][0]  # Rank 1 prediction context

        for txt, is_relevant in relevance_map.items():
            y_true.append(is_relevant)
            y_pred.append(1 if txt == top_result_text else 0)

    cm = confusion_matrix(y_true, y_pred)

    # Plotting configuration without plt.figure()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=['Not Rank 1', 'Rank 1'],
                yticklabels=['Irrelevant', 'Relevant'], ax=ax)

    plt.title('Document-Level Confusion Matrix (Top-1 Performance)')
    plt.xlabel('Model Prediction')
    plt.ylabel('Actual Ground Truth')
    plt.savefig("confusion_matrix_preview.png", bbox_inches='tight')

    print("\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred, target_names=['Irrelevant', 'Relevant']))

# Run matrix calculation
plot_ir_confusion_matrix(ranked_outputs, dataset)
