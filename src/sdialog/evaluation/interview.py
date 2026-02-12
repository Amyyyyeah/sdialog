"""
# Mock turn and dialog classes for demonstration
class Turn:
    def __init__(self, speaker, text):
        self.speaker = speaker
        self.text = text

class Dialog:
    def __init__(self, turns):
        self.turns = turns


# Create example dialogue
dialog1 = Dialog([
    Turn("Child", "uh I want to go to the park."),
    Turn("Adult", "That sounds great. What do you want to do there?"),
    Turn("Child", "um maybe play soccer.")
])

dialog2 = Dialog([
    Turn("Child", "I like reading books."),
    Turn("Adult", "What kind of books do you enjoy?")
])


# Run evaluation
evaluator = LinguisticFeaturesDatasetEvaluator()
summary = evaluator.run([dialog1, dialog2], dataset_name="example_dataset")

print("Evaluation Summary:")
for k, v in summary.items():
    print(f"{k}: {v}")

"""


import re
import os
import uuid
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import syllables
from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class EvalConfig:
    name: str = "linguistic_features"
    features: List[str] = None


class LinguisticFeaturesDatasetEvaluator:
    """
    Computes simple linguistic metrics per speaker and aggregates results
    across dialogues for dataset-level comparison.
    """

    def __init__(self, config: EvalConfig = None):
        self.config = config or EvalConfig(
            features=[
                "mean_turn_length",
                "hesitation_rate",
                "gunning_fog",
                "flesch_reading_ease",
            ]
        )

    # ------------------------
    # Text Cleaning
    # ------------------------
    @staticmethod
    def clean_utterance(text: str) -> str:
        text = re.sub(r"<[^>]*>", "", text)
        text = re.sub(r"\*[^*]*\*", "", text)
        text = re.sub(r"\([^)]*\)", "", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    # ------------------------
    # Basic Metrics
    # ------------------------
    @staticmethod
    def count_syllables(word: str) -> int:
        try:
            return max(1, syllables.estimate(word))
        except Exception:
            return 1  # safe fallback

    @staticmethod
    def calculate_gunning_fog(text: str) -> float:
        sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
        words = re.findall(r"\b[a-zA-Z]+\b", text)
        if not sentences or not words:
            return 0.0

        complex_words = sum(
            1 for w in words if LinguisticFeaturesDatasetEvaluator.count_syllables(w) >= 3
        )

        avg_sentence_length = len(words) / len(sentences)
        complex_ratio = (complex_words / len(words)) * 100
        return 0.4 * (avg_sentence_length + complex_ratio)

    @staticmethod
    def calculate_flesch(text: str) -> float:
        sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
        words = re.findall(r"\b[a-zA-Z]+\b", text)
        if not sentences or not words:
            return 0.0

        total_syllables = sum(
            LinguisticFeaturesDatasetEvaluator.count_syllables(w) for w in words
        )
        avg_sentence_length = len(words) / len(sentences)
        avg_syllables = total_syllables / len(words)

        return 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables)

    @staticmethod
    def count_hesitations(text: str) -> int:
        patterns = [
            r"\buh+\b", r"\bum+\b", r"\ber+\b",
            r"\bahh*\b", r"\bohh*\b", r"\bhmm+\b"
        ]
        text = text.lower()
        return sum(len(re.findall(p, text)) for p in patterns)

    # ------------------------
    # Core Evaluation
    # ------------------------
    def evaluate_dialog(self, dialog) -> Dict[str, Any]:
        speaker_stats = {}
        for turn in dialog.turns:
            if not getattr(turn, "speaker", None) or not getattr(turn, "text", None):
                continue

            cleaned = self.clean_utterance(turn.text)
            speaker_stats.setdefault(turn.speaker, []).append(cleaned)

        results = {}

        for speaker, utts in speaker_stats.items():
            if not utts:
                continue

            all_text = " ".join(utts)
            turn_lengths = [len(u.split()) for u in utts]
            total_words = max(1, sum(turn_lengths))
            total_hes = sum(self.count_hesitations(u) for u in utts)

            results[f"{speaker}_mean_turn_length"] = float(np.mean(turn_lengths))
            results[f"{speaker}_hesitation_rate"] = (total_hes / total_words) * 100
            results[f"{speaker}_gunning_fog"] = self.calculate_gunning_fog(all_text)
            results[f"{speaker}_flesch_reading_ease"] = self.calculate_flesch(all_text)

        return results

    def run(self, dialogs: List[Any], dataset_name: str = "unknown") -> Dict[str, Any]:
        """
        Run evaluation for a dataset. Results are isolated per run.
        """
        run_id = str(uuid.uuid4())
        all_results = []

        for dialog in dialogs:
            res = self.evaluate_dialog(dialog)
            res["dataset"] = dataset_name
            res["run_id"] = run_id
            all_results.append(res)

        df = pd.DataFrame(all_results)
        numeric_cols = df.select_dtypes(include=np.number).columns

        summary = df[numeric_cols].mean().to_dict()
        summary["dataset"] = dataset_name
        summary["run_id"] = run_id

        return summary
