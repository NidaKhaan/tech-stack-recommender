from src.data_loader import load_skills_data
from src.recommender import build_vectorizer, recommend

# 1. load the dataset
df = load_skills_data()

# 2. build the vectorizer + tfidf matrix
vectorizer, tfidf_matrix = build_vectorizer(df)

# 3. define a sample user_skills list
user_skills = ["Python", "Docker", "Astrology","Python"]

# 4. call recommend() and print the result
results, unknown = recommend(user_skills, df, vectorizer, tfidf_matrix)
print(results)
if unknown:
    print(f"⚠️  Unknown skills (ignored): {unknown}")