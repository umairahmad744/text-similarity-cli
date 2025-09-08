import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

documents = [
    "Machine learning is great for pattern recognition.",
    "I love playing football with my friends.",
    "Transformers are a powerful architecture for NLP.",
    "Deep learning allows us to model complex functions.",
    "Soccer and football are popular sports worldwide."
]

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

query = input("Enter your search query: ")
query_vec = vectorizer.transform([query])

similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()

results_df = pd.DataFrame({
    "Document": documents,
    "SimilarityScore": similarities
})


results_df = results_df.sort_values(by="SimilarityScore", ascending=False)
results_df.to_csv("similarity_results.csv", index=False)

best_doc = results_df.iloc[0]
print("\n Query:", query)
print("Most similar doc:", best_doc['Document'])
print("Similarity scores saved to similarity_results.csv ")
