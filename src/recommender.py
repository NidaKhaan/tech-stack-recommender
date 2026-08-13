"""
recommender.py
Core recommendation logic: TF-IDF vectorization + cosine similarity
to match user skills against job role skill profiles.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd


def build_vectorizer(df: pd.DataFrame):
    """Fit a TF-IDF vectorizer on the job roles' skills corpus."""
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df["skills"])
    return vectorizer, tfidf_matrix


def clean_user_skills(user_skills: list[str], vectorizer: TfidfVectorizer) -> tuple[list[str], list[str]]:
    """
    Normalize user skills and flag which ones are unknown to the vocabulary.

    Args:
        user_skills: raw list of skill strings from user input.
        vectorizer: fitted TfidfVectorizer (to check vocabulary).

    Returns:
        Tuple of (cleaned_skills, unknown_skills).
        cleaned_skills: deduplicated, stripped, non-empty skills.
        unknown_skills: skills not present in the trained vocabulary
                        (these contribute nothing to the score).
    """
    seen = set()
    cleaned = []
    for skill in user_skills:
        s = skill.strip()
        if not s:
            continue
        key = s.lower()
        if key not in seen:
            seen.add(key)
            cleaned.append(s)

    vocab = vectorizer.vocabulary_
    unknown = [s for s in cleaned if s.lower() not in vocab]

    return cleaned, unknown


def recommend(
    user_skills: list[str],
    df: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    tfidf_matrix,
    top_n: int = 3,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Recommend top_n job roles based on cosine similarity between
    user skills and each job role's skill profile.

    Returns:
        Tuple of (results_df, unknown_skills).
        results_df: ['job_role', 'similarity_score'], sorted descending.
        unknown_skills: skills the vectorizer's vocabulary didn't recognize
                        (useful for warning the user in the UI).

    Raises:
        ValueError: if user_skills is empty or all skills are blank/duplicate-only.
    """
    if not user_skills:
        raise ValueError("Please enter at least one skill.")

    cleaned_skills, unknown_skills = clean_user_skills(user_skills, vectorizer)

    if not cleaned_skills:
        raise ValueError("No valid skills provided after cleaning input.")

    user_text = ", ".join(cleaned_skills)
    user_vector = vectorizer.transform([user_text])

    similarities = cosine_similarity(user_vector, tfidf_matrix).flatten()

    results = df.copy()
    results["similarity_score"] = similarities
    results = results.sort_values("similarity_score", ascending=False)

    top_results = results[["job_role", "similarity_score"]].head(top_n).reset_index(drop=True)

    return top_results, unknown_skills