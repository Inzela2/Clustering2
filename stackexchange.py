# import os
# import numpy as np
# import pandas as pd
# import json
# import torch
# from sklearn.cluster import KMeans, AgglomerativeClustering
# from sklearn.metrics import silhouette_score, normalized_mutual_info_score, fowlkes_mallows_score, homogeneity_score
# import hdbscan
# from transformers import BertTokenizer, BertModel, GPT2Tokenizer, GPT2Model, AutoTokenizer, AutoModel
# from sentence_transformers import SentenceTransformer
# from sklearn.feature_extraction.text import TfidfVectorizer
# import ollama
#
# # Load the StackExchange dataset
# def load_stackexchange_dataset(file_path):
#     test_texts = []
#     true_labels = []
#     with open(file_path, 'r', encoding='utf-8') as f:
#         for line in f:
#             data = json.loads(line)
#             test_texts.append(data['input'])
#             true_labels.append(data['label'])
#     print(f"Total number of true labels: {len(true_labels)}")
#     print(f"Number of unique true labels: {len(set(true_labels))}")
#     return test_texts, true_labels
#
# # Embedding methods
# def get_tfidf_embeddings(test_texts):
#     print("Generating embeddings for tfidf...")
#     tfidf_vectorizer = TfidfVectorizer(max_features=10000)
#     embeddings = tfidf_vectorizer.fit_transform(test_texts).toarray()
#     print("Embeddings for tfidf generated.")
#     return embeddings
#
# def get_bert_embeddings(test_texts):
#     print("Generating embeddings for bert...")
#     tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#     model = BertModel.from_pretrained('bert-base-uncased')
#     embeddings = []
#     for text in test_texts:
#         inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
#         with torch.no_grad():
#             outputs = model(**inputs)
#         embeddings.append(outputs.last_hidden_state[:, 0, :].squeeze().numpy())
#     print("Embeddings for bert generated.")
#     return np.array(embeddings)
#
# def get_gpt2_embeddings(test_texts):
#     print("Generating embeddings for gpt2...")
#     tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
#     tokenizer.pad_token = tokenizer.eos_token  # Add padding token
#     model = GPT2Model.from_pretrained('gpt2')
#     embeddings = []
#     for text in test_texts:
#         inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
#         with torch.no_grad():
#             outputs = model(**inputs)
#         embeddings.append(outputs.last_hidden_state[:, 0, :].squeeze().numpy())
#     print("Embeddings for gpt2 generated.")
#     return np.array(embeddings)
#
# def get_glove_embeddings(test_texts):
#     print("Generating embeddings for glove...")
#     glove_path = 'C:/Users/asus/Downloads/glove.6B/glove.6B.300d.txt'
#     embedding_dim = 300
#     glove_embeddings = {}
#     with open(glove_path, 'r', encoding='utf-8') as f:
#         for line in f:
#             values = line.split()
#             word = values[0]
#             vector = np.array(values[1:], dtype='float32')
#             glove_embeddings[word] = vector
#     def get_glove_embedding(doc):
#         words = doc.split()
#         valid_embeddings = [glove_embeddings[word] for word in words if word in glove_embeddings]
#         return np.mean(valid_embeddings, axis=0) if valid_embeddings else np.zeros(embedding_dim)
#     embeddings = np.array([get_glove_embedding(doc) for doc in test_texts])
#     print("Embeddings for glove generated.")
#     return embeddings
#
# def get_minilm_embeddings(test_texts):
#     print("Generating embeddings for ollama...")
#     embeddings = []
#     for text in test_texts:
#         response = ollama.embeddings(model='all-minilm', prompt=text)
#         embeddings.append(np.array(response['embedding']))
#     print("Embeddings for ollama generated.")
#     return np.array(embeddings)
#
# def get_sentence_transformer_embeddings(test_texts):
#     print("Generating embeddings for sbert...")
#     model = SentenceTransformer("all-mpnet-base-v2")
#     embeddings = model.encode(test_texts, convert_to_numpy=True)
#     print("Embeddings for sbert generated.")
#     return embeddings
#
# # Embedding function for Voyage AI embeddings
# def get_voyage_embeddings(test_texts):
#     print("Generating embeddings for Voyage AI...")
#     voyage_model_name = "voyage-embedding-model"  # Replace with the actual model name from the Voyage AI documentation
#     tokenizer = AutoTokenizer.from_pretrained(voyage_model_name)
#     model = AutoModel.from_pretrained(voyage_model_name)
#     embeddings = []
#     for text in test_texts:
#         inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
#         with torch.no_grad():
#             outputs = model(**inputs)
#         embeddings.append(outputs.last_hidden_state[:, 0, :].squeeze().numpy())
#     print("Embeddings for Voyage AI generated.")
#     return np.array(embeddings)
#
# def get_jina_embeddings(test_texts):
#     print("Generating embeddings for jina...")
#     tokenizer = AutoTokenizer.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True)
#     model = AutoModel.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True)
#     embeddings = []
#     for text in test_texts:
#         inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
#         with torch.no_grad():
#             outputs = model(**inputs)
#         embeddings.append(outputs.last_hidden_state[:, 0, :].squeeze().numpy())
#     print("Embeddings for jina generated.")
#     return np.array(embeddings)
#
# # Clustering methods
# def evaluate_clustering(true_labels, predicted_labels, method_name, embeddings, param=None):
#     sil_score = silhouette_score(embeddings, predicted_labels)
#     nmi = normalized_mutual_info_score(true_labels, predicted_labels)
#     fmi = fowlkes_mallows_score(true_labels, predicted_labels)
#     homogeneity = homogeneity_score(true_labels, predicted_labels)
#     param_info = f" with parameter {param}" if param else ""
#     print(f"\n{method_name}{param_info} Evaluation Metrics:")
#     print(f"Silhouette Score: {sil_score}")
#     print(f"Fowlkes-Mallows Index (FMI): {fmi}")
#     print(f"Normalized Mutual Information (NMI): {nmi}")
#     print(f"Homogeneity: {homogeneity}")
#     return sil_score, fmi, nmi, homogeneity
#
# def clustering_experiments(embeddings, true_labels):
#     results = []
#     kmeans_cluster_values = [30, 60, 90, 120]
#     for n_clusters in kmeans_cluster_values:
#         kmeans = KMeans(n_clusters=n_clusters, random_state=42)
#         clusters_kmeans = kmeans.fit_predict(embeddings)
#         results.append(evaluate_clustering(true_labels, clusters_kmeans, 'KMeans', embeddings, n_clusters))
#
#     hdbscan_parameters = [(5, 20), (10, 30)]
#     for min_samples, min_cluster_size in hdbscan_parameters:
#         hdbscan_cluster = hdbscan.HDBSCAN(min_samples=min_samples, min_cluster_size=min_cluster_size)
#         clusters_hdbscan = hdbscan_cluster.fit_predict(embeddings)
#         if len(np.unique(clusters_hdbscan)) > 1:
#             results.append(evaluate_clustering(true_labels, clusters_hdbscan, 'HDBSCAN', embeddings, f"min_samples={min_samples}, min_cluster_size={min_cluster_size}"))
#
#     hac_cluster_values = [30, 60]
#     for n_clusters in hac_cluster_values:
#         hac = AgglomerativeClustering(n_clusters=n_clusters)
#         clusters_hac = hac.fit_predict(embeddings)
#         results.append(evaluate_clustering(true_labels, clusters_hac, 'Agglomerative Clustering', embeddings, n_clusters))
#
#     return results
#
# # Main function
# def main():
#     file_path = "C:/Users/asus/Downloads/clustering/datasets/stackexchange/small.jsonl"
#     test_texts, true_labels = load_stackexchange_dataset(file_path)
#
#     all_results = []
#     embedding_functions = {
#         'tfidf': get_tfidf_embeddings,
#         'bert': get_bert_embeddings,
#         'gpt2': get_gpt2_embeddings,
#         'glove': get_glove_embeddings,
#         'ollama': get_minilm_embeddings,
#         'sbert': get_sentence_transformer_embeddings,
#         'voyage': get_voyage_embeddings,
#         'jina': get_jina_embeddings,
#     }
#
#     for name, func in embedding_functions.items():
#         print(f"\nProcessing {name} embeddings...")
#         embeddings = func(test_texts)
#         all_results.extend(clustering_experiments(embeddings, true_labels))
#
#     results_df = pd.DataFrame(all_results, columns=["Silhouette Score", "FMI", "NMI", "Homogeneity"])
#     results_df.to_csv("stackexchange_clustering_results.csv", index=False)
#     print("Results saved to stackexchange_clustering_results.csv")
#
# if __name__ == "__main__":
#     main()




import os
import numpy as np
import pandas as pd
import json
import torch
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, normalized_mutual_info_score, fowlkes_mallows_score, homogeneity_score
import hdbscan
from transformers import BertTokenizer, BertModel, GPT2Tokenizer, GPT2Model
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import ollama

# Load the StackExchange dataset
def load_stackexchange_dataset(file_path):
    test_texts = []
    true_labels = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            test_texts.append(data['input'])
            true_labels.append(data['label'])
    print(f"Total number of true labels: {len(true_labels)}")
    print(f"Number of unique true labels: {len(set(true_labels))}")
    return test_texts, true_labels

# Embedding methods
def get_tfidf_embeddings(test_texts):
    print("Generating embeddings for tfidf...")
    tfidf_vectorizer = TfidfVectorizer(max_features=10000)
    embeddings = tfidf_vectorizer.fit_transform(test_texts).toarray()
    print("Embeddings for tfidf generated.")
    return embeddings

def get_bert_embeddings(test_texts):
    print("Generating embeddings for bert...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    embeddings = []
    for text in test_texts:
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings.append(outputs.last_hidden_state[:, 0, :].squeeze().numpy())
    print("Embeddings for bert generated.")
    return np.array(embeddings)

def get_gpt2_embeddings(test_texts):
    print("Generating embeddings for gpt2...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token  # Add padding token
    model = GPT2Model.from_pretrained('gpt2')
    embeddings = []
    for text in test_texts:
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings.append(outputs.last_hidden_state[:, 0, :].squeeze().numpy())
    print("Embeddings for gpt2 generated.")
    return np.array(embeddings)

def get_glove_embeddings(test_texts):
    print("Generating embeddings for glove...")
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
    embeddings = np.array([get_glove_embedding(doc) for doc in test_texts])
    print("Embeddings for glove generated.")
    return embeddings

def get_minilm_embeddings(test_texts):
    print("Generating embeddings for ollama...")
    embeddings = []
    for text in test_texts:
        response = ollama.embeddings(model='all-minilm', prompt=text)
        embeddings.append(np.array(response['embedding']))
    print("Embeddings for ollama generated.")
    return np.array(embeddings)

def get_sentence_transformer_embeddings(test_texts):
    print("Generating embeddings for sbert...")
    model = SentenceTransformer("all-mpnet-base-v2")
    embeddings = model.encode(test_texts, convert_to_numpy=True)
    print("Embeddings for sbert generated.")
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

def clustering_experiments(embeddings, true_labels):
    results = []
    kmeans_cluster_values = [100, 121, 150]
    for n_clusters in kmeans_cluster_values:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters_kmeans = kmeans.fit_predict(embeddings)
        results.append(evaluate_clustering(true_labels, clusters_kmeans, 'KMeans', embeddings, n_clusters))

    # Updated HDBSCAN parameters to target around 121 clusters
    hdbscan_parameters = [(10, 80), (15, 100), (20, 120)]
    for min_samples, min_cluster_size in hdbscan_parameters:
        hdbscan_cluster = hdbscan.HDBSCAN(min_samples=min_samples, min_cluster_size=min_cluster_size)
        clusters_hdbscan = hdbscan_cluster.fit_predict(embeddings)
        if len(np.unique(clusters_hdbscan)) > 1:
            results.append(evaluate_clustering(true_labels, clusters_hdbscan, 'HDBSCAN', embeddings, f"min_samples={min_samples}, min_cluster_size={min_cluster_size}"))

    # Updated HAC values to test around the target number of clusters
    hac_cluster_values = [115, 121, 130]
    for n_clusters in hac_cluster_values:
        hac = AgglomerativeClustering(n_clusters=n_clusters)
        clusters_hac = hac.fit_predict(embeddings)
        results.append(evaluate_clustering(true_labels, clusters_hac, 'Agglomerative Clustering', embeddings, n_clusters))

    return results


# Main function
def main():
    file_path = "C:/Users/asus/Downloads/clustering/datasets/stackexchange/small.jsonl"
    test_texts, true_labels = load_stackexchange_dataset(file_path)

    all_results = []
    embedding_functions = {
        'tfidf': get_tfidf_embeddings,
        'bert': get_bert_embeddings,
        'gpt2': get_gpt2_embeddings,
        'glove': get_glove_embeddings,
        'ollama': get_minilm_embeddings,
        'sbert': get_sentence_transformer_embeddings,  # Running this first
    }

    for name, func in embedding_functions.items():
        print(f"\nProcessing {name} embeddings...")
        embeddings = func(test_texts)
        all_results.extend(clustering_experiments(embeddings, true_labels))

    results_df = pd.DataFrame(all_results, columns=["Silhouette Score", "FMI", "NMI", "Homogeneity"])
    results_df.to_csv("stackexchange_clustering_results.csv", index=False)
    print("Results saved to stackexchange_clustering_results.csv")

if __name__ == "__main__":
    main()
