"""
recommender.py
Core recommendation logic: TF-IDF vectorization + cosine similarity
to match user skills against job role skill profiles.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd


def build_vectorizer(df: pd.DataFrame):
    """
    Fit a TF-IDF vectorizer on the job roles' skills corpus.

    Args:
        df: DataFrame with a 'skills' column.

    Returns:
        Tuple of (fitted TfidfVectorizer, TF-IDF matrix for df['skills']).
    """
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df["skills"])
    return vectorizer, tfidf_matrix


def recommend(
    user_skills: list[str],
    df: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    tfidf_matrix,
    top_n: int = 3,
) -> pd.DataFrame:
    """
    Recommend top_n job roles based on cosine similarity between
    user skills and each job role's skill profile.

    Args:
        user_skills: list of skill strings, e.g. ["Python", "Cloud", "Automation"].
        df: original job roles DataFrame (for looking up role names).
        vectorizer: fitted TfidfVectorizer (from build_vectorizer).
        tfidf_matrix: TF-IDF matrix of job roles (from build_vectorizer).
        top_n: number of recommendations to return.

    Returns:
        DataFrame with columns ['job_role', 'similarity_score'],
        sorted descending by score.
    """
    if not user_skills:
        raise ValueError("user_skills cannot be empty")

    # Join user skills into a single space-separated string,
    # same format the vectorizer was trained on
    user_text = ", ".join(user_skills)

    # Transform (not fit_transform!) — must use the SAME vocabulary
    # learned from the job roles corpus, otherwise vectors aren't comparable
    user_vector = vectorizer.transform([user_text])

    # Compute similarity between user vector and every job role vector
    similarities = cosine_similarity(user_vector, tfidf_matrix).flatten()

    results = df.copy()
    results["similarity_score"] = similarities
    results = results.sort_values("similarity_score", ascending=False)

    return results[["job_role", "similarity_score"]].head(top_n).reset_index(drop=True)