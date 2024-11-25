import os
import pandas as pd
import ollama
from concurrent.futures import ThreadPoolExecutor, as_completed

# Path to the dataset
DATASET_PATH = r"C:\Users\asus\Downloads\clustering\kaggle_data\website-classification\website_classification.csv"
OUTPUT_PATH = r"C:\Users\asus\Downloads\clustering\kaggle_data\website-classification\summarized_website_classification.csv"

# Load dataset
def load_dataset():
    print("Loading dataset...")
    df = pd.read_csv(DATASET_PATH)
    print("First few samples:")
    print(df.head())
    texts = df['cleaned_website_text'].tolist()
    categories = df['Category'].tolist()
    print(f"Dataset loaded with {len(texts)} documents.")
    return df, texts, categories

# Summarize each document using Mistral concurrently with retry and truncation
def summarize_text_concurrently(text, model="avr/sfr-embedding-mistral:latest", retries=3):
    max_length = 512  # Limit the input length for summarization
    truncated_text = text[:max_length]
    for attempt in range(retries):
        try:
            prompt = "Summarize the following text in a concise sentence for better analysis:\n\n" + truncated_text
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
def summarize_texts(texts, model="avr/sfr-embedding-mistral:latest"):
    summaries = []
    with ThreadPoolExecutor(max_workers=4) as executor:  # Adjust concurrency based on system capability
        future_to_text = {executor.submit(summarize_text_concurrently, text, model): text for text in texts}
        for future in as_completed(future_to_text):
            try:
                summaries.append(future.result())
            except Exception as e:
                print(f"Error summarizing text: {e}")
    return summaries

# Main function
def main():
    print("Starting summarization...")
    df, texts, categories = load_dataset()

    print("Generating summaries...")
    summaries = summarize_texts(texts)

    # Save the summarized texts with their labels to a CSV
    df['summarized_text'] = summaries
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Summarized texts saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
