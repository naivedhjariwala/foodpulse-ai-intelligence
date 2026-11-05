"""
NLP Sentiment Analysis & Service Complaint Extraction for FoodPulse.
Processes customer review text using TF-IDF, N-Gram frequency analysis, sentiment scoring,
and rules-based service failure categorizers to extract actionable operational feedback.
"""

import os
import re
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

COMPLAINT_PATTERNS = {
    "Late Delivery": [
        r"\blate\b", r"\bdelay", r"\bdelayed\b", r"\bwaited\b", r"\bwaiting\b",
        r"\bminutes\b", r"\bhours?\b", r"\bslow\b", r"\beta\b", r"\bdetour\b"
    ],
    "Cold Food": [
        r"\bcold\b", r"\blukewarm\b", r"\bsoggy\b", r"\brubber\b", r"\bfreeze\b",
        r"\bfreezing\b", r"\bcongealed\b", r"\bmicrowave\b", r"\bthermal\b"
    ],
    "Missing Items": [
        r"\bmissing\b", r"\bforgot\b", r"\bleft out\b", r"\bhalf\b", r"\bno drinks?\b",
        r"\bwhere is\b", r"\bincomplete\b"
    ],
    "Rude Courier": [
        r"\brude\b", r"\bimpatient\b", r"\bargumentative\b", r"\bshoved\b",
        r"\bthrew\b", r"\brefused\b", r"\byelled\b"
    ],
    "Damaged Packaging / Spills": [
        r"\bleak", r"\bspill", r"\bcrushed\b", r"\bmess\b", r"\bdamaged\b",
        r"\bloose lid\b", r"\bupside down\b"
    ]
}

POSITIVE_WORDS = {
    "blazing", "fast", "piping", "hot", "delicious", "perfect", "ahead", "fresh",
    "crispy", "super", "polite", "seamless", "best", "generous", "securely",
    "speedy", "great", "excellent", "amazing", "quick", "warm", "tasty", "courteous"
}

NEGATIVE_WORDS = {
    "unacceptable", "delay", "cold", "late", "ruined", "bizarre", "detours", "terrible",
    "solidified", "lukewarm", "soggy", "freezing", "congealed", "disappointed",
    "missing", "forgot", "left", "half", "rude", "refused", "threw", "impatient",
    "argumentative", "shoved", "leaked", "mess", "damaged", "crushed", "loose", "spilled"
}


def clean_text(text: str) -> str:
    """Sanitizes text, lowers case, and removes non-alphanumeric symbols."""
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class NLPSentimentAnalyzer:
    """
    NLP Engine for review sentiment classification and operational complaint extraction.
    """
    def __init__(self, max_features: int = 600):
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words="english",
            max_features=max_features
        )
        self.is_fitted = False
        self.top_positive_ngrams = []
        self.top_negative_ngrams = []
        self.complaint_distribution = {}

    def fit(self, df_reviews: pd.DataFrame) -> "NLPSentimentAnalyzer":
        """
        Fits TF-IDF vectorizer and computes top positive and negative n-grams.
        """
        cleaned_texts = [clean_text(t) for t in df_reviews["review_text"]]
        self.vectorizer.fit(cleaned_texts)

        # Categorize by sentiment
        pos_texts = [clean_text(t) for t in df_reviews[df_reviews["sentiment_label"] == "Positive"]["review_text"]]
        neg_texts = [clean_text(t) for t in df_reviews[df_reviews["sentiment_label"] == "Negative"]["review_text"]]

        self.top_positive_ngrams = self._extract_top_ngrams(pos_texts, top_n=15)
        self.top_negative_ngrams = self._extract_top_ngrams(neg_texts, top_n=15)

        # Complaint counts
        self.complaint_distribution = df_reviews["complaint_category"].value_counts().to_dict()
        self.is_fitted = True
        return self

    def _extract_top_ngrams(self, texts: List[str], top_n: int = 15) -> List[Tuple[str, float]]:
        """Extracts top frequent unigrams and bigrams from a list of texts."""
        if not texts:
            return []
        try:
            cv = CountVectorizer(ngram_range=(1, 2), stop_words="english", max_features=top_n * 2)
            matrix = cv.fit_transform(texts)
            counts = matrix.sum(axis=0).A1
            vocab = cv.get_feature_names_out()
            freq_pairs = sorted(zip(vocab, counts), key=lambda x: x[1], reverse=True)
            return [(word, int(count)) for word, count in freq_pairs[:top_n]]
        except Exception:
            return []

    def analyze_single_review(self, review_text: str) -> Dict[str, Any]:
        """
        Analyzes a single user-provided review:
        returns polarity score, sentiment label, and detected complaints.
        """
        cleaned = clean_text(review_text)
        words = set(cleaned.split())

        pos_matches = words.intersection(POSITIVE_WORDS)
        neg_matches = words.intersection(NEGATIVE_WORDS)

        pos_count = len(pos_matches)
        neg_count = len(neg_matches)

        # Polarity score in [-1.0, 1.0]
        total_matched = pos_count + neg_count
        if total_matched > 0:
            polarity_score = round((pos_count - neg_count) / total_matched, 2)
        else:
            polarity_score = 0.0

        if polarity_score > 0.15:
            sentiment = "Positive"
        elif polarity_score < -0.15:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"

        # Complaint detection using regex patterns
        detected_complaints = []
        for category, patterns in COMPLAINT_PATTERNS.items():
            for pat in patterns:
                if re.search(pat, cleaned):
                    detected_complaints.append(category)
                    break

        if not detected_complaints:
            if sentiment == "Positive":
                detected_complaints = ["None (Complimentary / Satisfied)"]
            else:
                detected_complaints = ["General / Unspecified"]

        return {
            "cleaned_text": cleaned,
            "sentiment_label": sentiment,
            "polarity_score": polarity_score,
            "detected_complaints": detected_complaints,
            "key_positive_signals": list(pos_matches),
            "key_negative_signals": list(neg_matches)
        }

    def save(self, filepath: str) -> None:
        """Saves model to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)

    @staticmethod
    def load(filepath: str) -> "NLPSentimentAnalyzer":
        """Loads serialized model."""
        return joblib.load(filepath)
