











import os
import numpy as np
import pandas as pd
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, normalized_mutual_info_score, fowlkes_mallows_score, homogeneity_score
import hdbscan
from transformers import BertTokenizer, BertModel, GPT2Tokenizer, GPT2Model
from sentence_transformers import SentenceTransformer
import ollama

# Path to dataset
DATASET_PATH = r"C:\Users\asus\Downloads\clustering\kaggle_data\website-classification\website_classification.csv"

# Load dataset
def load_dataset():
    print("Loading dataset...")
    df = pd.read_csv(DATASET_PATH)
    print("First few samples:")
    print(df.head())
    texts = df['cleaned_website_text'].tolist()
    categories = df['Category'].tolist()
    print(f"Dataset loaded with {len(texts)} documents.")
    return texts, categories

# Summarize each document using Llama 3.1 concurrently with retry and truncation
def summarize_text_concurrently(text, model="llama3.1:latest", retries=3):
    max_length = 512  # Limit the input length for summarization
    truncated_text = text[:max_length]
    for attempt in range(retries):
        try:
            prompt = "Summarize the following text in a concise sentence for better embedding analysis:\n\n" + truncated_text
            print("Summarizing text...")
            response = ollama.generate(prompt=prompt, model=model)
            if 'content' not in response:
                raise ValueError(f"Summarization failed for text: {text}")
            summary = response['content']
            print("Summary generated.")
            return summary
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt + 1 < retries:
                print("Retrying...")
    raise ValueError(f"Summarization failed after {retries} attempts for text: {text}")

# Summarize all texts
def summarize_texts(texts, model="llama3.1:latest"):
    summaries = []
    with ThreadPoolExecutor(max_workers=2) as executor:  # Reduce concurrency to 2 workers
        future_to_text = {executor.submit(summarize_text_concurrently, text, model): text for text in texts}
        for future in as_completed(future_to_text):
            summaries.append(future.result())  # No skipping, all texts must succeed
    return summaries

# Embedding methods
def get_tfidf_embeddings(texts):
    print("Generating embeddings for TF-IDF...")
    tfidf_vectorizer = TfidfVectorizer(max_features=10000)
    embeddings = tfidf_vectorizer.fit_transform(texts).toarray()
    print("Embeddings for TF-IDF generated.")
    return embeddings

def get_bert_embeddings(texts):
    print("Generating embeddings for BERT...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    embeddings = []
    for text in texts:
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings.append(outputs.last_hidden_state[:, 0, :].squeeze().numpy())
    print("Embeddings for BERT generated.")
    return np.array(embeddings)

def get_gpt2_embeddings(texts):
    print("Generating embeddings for GPT-2...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2Model.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    embeddings = []
    for text in texts:
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings.append(outputs.last_hidden_state[:, 0, :].squeeze().numpy())
    print("Embeddings for GPT-2 generated.")
    return np.array(embeddings)

def get_glove_embeddings(texts):
    print("Generating embeddings for GloVe...")
    glove_path = 'C:/Users/asus/Downloads/glove.6B/glove.6B.300d.txt'
    embedding_dim = 300
    glove_embeddings = {}
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.array(values[1:], dtype='float32')
            glove_embeddings[word] = vector

    def get_glove_embedding(doc):
        words = doc.split()
        valid_embeddings = [glove_embeddings[word] for word in words if word in glove_embeddings]
        return np.mean(valid_embeddings, axis=0) if valid_embeddings else np.zeros(embedding_dim)

    embeddings = np.array([get_glove_embedding(doc) for doc in texts])
    print("Embeddings for GloVe generated.")
    return embeddings

def get_minilm_embeddings(texts):
    print("Generating embeddings for Ollama MiniLM...")
    embeddings = []
    for text in texts:
        response = ollama.embeddings(model='all-minilm', prompt=text)
        embeddings.append(np.array(response['embedding']))
    print("Embeddings for Ollama MiniLM generated.")
    return np.array(embeddings)

def get_sentence_transformer_embeddings(texts):
    print("Generating embeddings for Sentence Transformer (all-mpnet-base-v2)...")
    model = SentenceTransformer("all-mpnet-base-v2")
    embeddings = model.encode(texts, convert_to_numpy=True)
    print("Embeddings for Sentence Transformer generated.")
    return embeddings

# Clustering methods
def evaluate_clustering(true_labels, predicted_labels, method_name, embeddings, param=None):
    sil_score = silhouette_score(embeddings, predicted_labels)
    nmi = normalized_mutual_info_score(true_labels, predicted_labels)
    fmi = fowlkes_mallows_score(true_labels, predicted_labels)
    homogeneity = homogeneity_score(true_labels, predicted_labels)
    param_info = f" with parameter {param}" if param else ""
    print(f"\n{method_name}{param_info} Evaluation Metrics:")
    print(f"Silhouette Score: {sil_score}")
    print(f"Fowlkes-Mallows Index (FMI): {fmi}")
    print(f"Normalized Mutual Information (NMI): {nmi}")
    print(f"Homogeneity: {homogeneity}")
    return sil_score, fmi, nmi, homogeneity

def clustering_experiments(embeddings, true_labels, embedding_name):
    results = []
    kmeans_cluster_values = [15, 20, 25]
    for n_clusters in kmeans_cluster_values:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters_kmeans = kmeans.fit_predict(embeddings)
        results.append(
            (embedding_name, 'KMeans', n_clusters) + evaluate_clustering(true_labels, clusters_kmeans, 'KMeans',
                                                                         embeddings, n_clusters))
    return results

# Main function
def main():
    print("Starting clustering experiment...")
    texts, categories = load_dataset()

    print("Starting summarization...")
    summarized_texts = summarize_texts(texts)

    embedding_methods = {
        'tfidf': get_tfidf_embeddings,
        'bert': get_bert_embeddings,
        'gpt2': get_gpt2_embeddings,
        'glove': get_glove_embeddings,
        'ollama': get_minilm_embeddings,
        'sbert': get_sentence_transformer_embeddings,
    }

    all_results = []
    for name, func in embedding_methods.items():
        print(f"\nGenerating {name} embeddings...")
        embeddings = func(summarized_texts)
        all_results.extend(clustering_experiments(embeddings, categories, name))

    results_df = pd.DataFrame(all_results, columns=["Embedding", "Clustering_Method", "Parameter", "Silhouette_Score", "FMI", "NMI", "Homogeneity"])
    results_df.to_csv("clustering_results.csv", index=False)
    print("Results saved to clustering_results.csv")

if __name__ == "__main__":
    main()
