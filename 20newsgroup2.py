import os
import numpy as np
import pandas as pd
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, normalized_mutual_info_score, fowlkes_mallows_score, homogeneity_score
import hdbscan
from transformers import BertTokenizer, BertModel, GPT2Tokenizer, GPT2Model
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups
import ollama


# Summarize each document using Llama 3.1 concurrently
def summarize_text_concurrently(text, model="llama3.1:latest"):
    prompt = "Summarize the following text in a concise sentence for better embedding analysis:\n\n" + text
    try:
        print("Summarizing text...")
        response = ollama.generate(prompt=prompt, model=model)
        summary = response.get('content', text) if 'content' in response else text
        print("Summary generated.")
        return summary
    except Exception as e:
        print(f"Error summarizing text: {e}")
        return text  # Use original text if summarization fails


def summarize_texts(texts, model="llama3.1:latest"):
    summaries = []
    with ThreadPoolExecutor(max_workers=8) as executor:  # Adjust max_workers based on your machine's capabilities
        future_to_text = {executor.submit(summarize_text_concurrently, text, model): text for text in texts}
        for future in as_completed(future_to_text):
            try:
                summaries.append(future.result())
            except Exception as e:
                print(f"An error occurred: {e}")
                summaries.append(future_to_text[future])  # Use original text if summarization fails
    return summaries


# Load the 20 Newsgroups dataset with all categories
def load_20newsgroups_dataset():
    print("Loading the 20 Newsgroups dataset...")
    dataset = fetch_20newsgroups(remove=("headers", "footers", "quotes"), subset="all", shuffle=True, random_state=42)

    print("Starting summarization of texts...")
    texts = summarize_texts(dataset.data)

    labels = dataset.target
    unique_labels, category_sizes = np.unique(labels, return_counts=True)
    print(f"{len(texts)} documents - {unique_labels.shape[0]} categories")
    return texts, labels


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
    tokenizer.pad_token = tokenizer.eos_token  # Set the padding token to the EOS token
    embeddings = []
    for text in texts:
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings.append(outputs.last_hidden_state[:, 0, :].squeeze().numpy())
    print("Embeddings for GPT-2 generated.")
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
    kmeans_cluster_values = [15, 20, 25]  # Adjusted for the number of categories in 20 Newsgroups
    for n_clusters in kmeans_cluster_values:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters_kmeans = kmeans.fit_predict(embeddings)
        results.append(
            (embedding_name, 'KMeans', n_clusters) + evaluate_clustering(true_labels, clusters_kmeans, 'KMeans',
                                                                         embeddings, n_clusters))

    hdbscan_parameters = [(5, 10), (10, 20), (15, 30)]
    for min_samples, min_cluster_size in hdbscan_parameters:
        hdbscan_cluster = hdbscan.HDBSCAN(min_samples=min_samples, min_cluster_size=min_cluster_size)
        clusters_hdbscan = hdbscan_cluster.fit_predict(embeddings)
        if len(np.unique(clusters_hdbscan)) > 1:
            results.append((embedding_name, 'HDBSCAN',
                            f"min_samples={min_samples}, min_cluster_size={min_cluster_size}") + evaluate_clustering(
                true_labels, clusters_hdbscan, 'HDBSCAN', embeddings,
                f"min_samples={min_samples}, min_cluster_size={min_cluster_size}"))

    hac_cluster_values = [15, 20, 25]
    for n_clusters in hac_cluster_values:
        hac = AgglomerativeClustering(n_clusters=n_clusters)
        clusters_hac = hac.fit_predict(embeddings)
        results.append((embedding_name, 'Agglomerative', n_clusters) + evaluate_clustering(true_labels, clusters_hac,
                                                                                           'Agglomerative Clustering',
                                                                                           embeddings, n_clusters))

    return results


# Main function
def main():
    print("Starting the 20 Newsgroups clustering experiment...")
    texts, true_labels = load_20newsgroups_dataset()

    all_results = []
    embedding_functions = {
        'tfidf': get_tfidf_embeddings,
        'bert': get_bert_embeddings,
        'gpt2': get_gpt2_embeddings,
        'sbert': get_sentence_transformer_embeddings,
    }

    for name, func in embedding_functions.items():
        print(f"\nProcessing {name} embeddings...")
        embeddings = func(texts)
        all_results.extend(clustering_experiments(embeddings, true_labels, name))

    # Convert results to a DataFrame and save to CSV
    results_df = pd.DataFrame(all_results,
                              columns=["Embedding", "Clustering_Method", "Parameter", "Silhouette_Score", "FMI", "NMI",
                                       "Homogeneity"])
    results_df.to_csv("20newsgroups_clustering_results.csv", index=False)
    print("Results saved to 20newsgroups_clustering_results.csv")


if __name__ == "__main__":
    main()
