Extractive Question Answering and Re-ranking System

Overview

This project focuses on two important Natural Language Processing (NLP) tasks:

Extractive Question Answering (QA)
Passage Re-ranking
The project explores how information retrieval and transformer-based models can be combined to retrieve relevant passages and extract accurate answers from text.

Project Components

1. Extractive Question Answering

The extractive QA component predicts an answer span directly from a given context passage.

Dataset Used

SQuAD v1.1
Features

Context preprocessing
Tokenization
Answer span prediction
Evaluation using:
Exact Match (EM)
F1 Score
2. Passage Re-ranking

The re-ranking component improves retrieval quality by ranking candidate passages according to their relevance to a query.

Dataset Used

MS MARCO Dataset
The dataset was imported directly during runtime and was not stored locally in the repository.

Features

Query-passage relevance scoring
Passage ranking
Transformer-based reranking workflow
Retrieval quality improvement
Technologies Used

Python
Google Colab
Hugging Face Transformers
PyTorch
NumPy
Pandas
Scikit-learn
Workflow

Extractive QA Pipeline Load SQuAD dataset Preprocess contexts and questions Tokenize text Train/evaluate QA model Predict answer spans Compute EM and F1 metrics Re-ranking Pipeline Load query-passage pairs from MS MARCO Generate candidate passages Compute relevance scores Re-rank retrieved passages Return top-ranked passages

Running the Project

Open the notebook using Jupyter Notebook or Google Colab and run the cells sequentially.

Future Improvements Dense retrieval models Semantic search with embeddings Hybrid retrieval systems API deployment Web interface integration Advanced reranking architectures


License

This project is for educational and research purposes.
