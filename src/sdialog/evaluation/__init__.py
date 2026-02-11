"""
evaluation: Evaluation components for dialogue generation and analysis.

This module provides abstract base classes for evaluating dialogues,
including LLM judges, metrics, and similarity scores.
"""
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>
# SPDX-License-Identifier: MIT
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import syllables

from jinja2 import Template
from typing import Optional
from tqdm.auto import tqdm
from pydantic import BaseModel
from math import exp, log, sqrt
from abc import ABC, abstractmethod
from typing import Union, List, Dict
from scipy.stats import gaussian_kde
from sentence_transformers import SentenceTransformer
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models.base import BaseLanguageModel

from .. import Dialog
from ..personas import BasePersona
from ..config import config
from .dialog2flow import dialog2graph, DEFAULT_TOKEN_START
from ..util import KNNModel, softmax, get_llm_model, dict_to_table, upper_camel_to_dash


def cs_divergence(p1, p2, resolution=100, bw_method=1):
    """
    Calculates the Cauchy-Schwarz divergence between two probability distributions.
    (0 is the same distribution)
    :param p1: First sample (1D array or list)
    :type p1: array-like
    :param p2: Second sample (1D array or list)
    :type p2: array-like
    :param resolution: Number of points to evaluate the KDEs on (default: 100)
    :type resolution: int
    :param bw_method: Bandwidth for KDE (default: 1, i.e., standard bandwidth)
    :type bw_method: float or str
    :return: Cauchy-Schwarz divergence (0 means identical distributions)
    :rtype: float
    """
    p1 = np.asarray(p1)
    p2 = np.asarray(p2)
    r = np.linspace(min(p1.min(), p2.min()), max(p1.max(), p2.max()), resolution)
    p1_kernel = gaussian_kde(p1, bw_method=bw_method)
    p2_kernel = gaussian_kde(p2, bw_method=bw_method)
    p1_vals = p1_kernel(r)
    p2_vals = p2_kernel(r)
    numerator = np.sum(p1_vals * p2_vals)
    denominator = sqrt(np.sum(p1_vals ** 2) * np.sum(p2_vals ** 2))
    return -log(numerator / denominator)


def kl_divergence(p1, p2, resolution=100, bw_method=1e-1):
    """
    Estimates the Kullback-Leibler (KL) divergence KL(p1 || p2) between two distributions given samples, using KDE.

    KL divergence is not symmetric: KL(p1 || p2) != KL(p2 || p1).
    The result is >= 0, and 0 means the distributions are identical.

    :param p1: First sample (1D array or list) (the 'true' distribution)
    :type p1: array-like
    :param p2: Second sample (1D array or list) (the 'approximate' distribution)
    :type p2: array-like
    :param resolution: Number of points to evaluate the KDEs on (default: 100)
    :type resolution: int
    :param bw_method: Bandwidth for KDE (default: 0.1)
    :type bw_method: float or str
    :return: KL divergence KL(p1 || p2)
    :rtype: float
    """
    r = np.linspace(min(p1.min(), p2.min()), max(p1.max(), p2.max()), resolution)
    p1_kernel = gaussian_kde(p1, bw_method=bw_method)
    p2_kernel = gaussian_kde(p2, bw_method=bw_method)
    p1_vals = p1_kernel(r)
    p2_vals = p2_kernel(r)
    # Avoid division by zero and log(0) by adding a small epsilon
    eps = 1e-12
    p1_vals = np.clip(p1_vals, eps, None)
    p2_vals = np.clip(p2_vals, eps, None)

    return float(np.sum(p1_vals * np.log(p1_vals / p2_vals)) / np.sum(p1_vals))


class LLMJudgeYesNoOutput(BaseModel):
    """
    Pydantic model for LLM-generated dialogue output.
    """
    yes: Union[bool, List[bool]]
    feedback: Optional[Union[str, List[str]]] = None


class BaseEvaluator(ABC):
    def __init__(self, metrics=None):
        pass

    @abstractmethod
    def evaluate(self, input: Union[Dialog, List[Dialog]]) -> dict:
        """
        Evaluate the dialogs.
        """
        raise NotImplementedError("Subclasses should implement this method.")


class BaseMetric(ABC):
    """
    Base class for metrics.
    """
    def __init__(self):
        pass

    @abstractmethod
    def compute(self, input: Union[Dialog, List[Dialog]]) -> Union[dict, float]:
        raise NotImplementedError("Subclasses should implement this method.")


class BaseDialogScore(ABC):
    def __init__(self, name: Optional[str] = None):
        """
        Initialize the dialog score with a name.
        :param name: Name of the dialog score.
        """
        self.name = name

    def __call__(self, dialog: Dialog):
        """
        Computes the score for the provided dialog.

        :param dialog: The dialog to score.
        :return: A float representing the score of the dialog.
        """
        raise NotImplementedError("Subclasses should implement this method.")


class BaseDatasetEvaluator(ABC):
    """
    Base class for dataset evaluators.
    """
    def __init__(self, dialog_score: BaseDialogScore = None):
        """
        Initialize the evaluator with the target dialog score.
        """
        self.dialog_score = dialog_score

    @abstractmethod
    def __call__(self, dialogues: Union[str, List[Dialog]]) -> Union[dict, float]:
        """
        Compute the dialog scores on each dialogue and return the evaluation.
        :return: A dictionary with the evaluation results or a single score.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def eval(self, dialogues: Union[str, List[Dialog]], **kwargs) -> Union[dict, float]:
        return self(dialogues, **kwargs)


class BaseLLMJudge(ABC):
    """
    Base class for LLM judges.
    """
    def __init__(self,
                 model: Union[BaseLanguageModel, str] = None,
                 output_format: Union[dict, BaseModel] = None,
                 **llm_kwargs):
        if model is None:
            model = config["llm"]["model"]

        # Collect LLM parameters from config, only if not None
        llm_config_params = {k: v for k, v in config["llm"].items() if k != "model" and v is not None}
        llm_kwargs = {**llm_config_params, **llm_kwargs}

        self.output_format = output_format

        if isinstance(model, str):
            self.llm = get_llm_model(model_name=model,
                                     output_format=self.output_format,
                                     **llm_kwargs)
        else:
            self.llm = model
            if output_format:
                self.llm.format = self.output_format.model_json_schema()

        with open(config["prompts"]["evaluation"]["llm_as_judge"], encoding="utf-8") as f:
            self.messages = [SystemMessage(f.read()), HumanMessage("")]

    def __call__(self, prompt: str) -> Union[dict, BaseModel]:
        self.messages[1].content = prompt
        return self.llm.invoke(self.messages).content

    @abstractmethod
    def judge(self, dialogs: Union[Dialog, List[Dialog]]) -> dict:
        """
        Judge the dialogs using the LLM.
        """
        raise NotImplementedError("Subclasses should implement this method.")


class LLMJudgeYesNo(BaseLLMJudge, BaseDialogScore):
    """LLM judge for classifying a dialogue as "yes or no" (boolean) output and feedback."""
    def __init__(self,
                 prompt_template: str,
                 model: Union[BaseLanguageModel, str] = None,
                 feedback: bool = False,
                 as_score: bool = False,
                 **llm_kwargs):
        BaseDialogScore.__init__(self,
                                 name=upper_camel_to_dash(self.__class__.__name__))
        super().__init__(output_format=LLMJudgeYesNoOutput,
                         model=model,
                         **llm_kwargs)

        self.prompt_template = Template(prompt_template)
        self.feedback = feedback
        self.as_score = as_score  # If True, returns either 1 or 0 instead of LLMJudgeYesNoOutput object

    def judge(self, dialogs: Union[Dialog, List[Dialog]], feedback: bool = None) -> Union[LLMJudgeYesNoOutput, int]:
        if isinstance(dialogs, Dialog):
            dialogs = [dialogs]  # Wrap single dialog in a list

        prompt = self.prompt_template.render(dialogs=dialogs,
                                             feedback=feedback if feedback is not None else self.feedback)
        output = self.output_format.model_validate(json.loads(super().__call__(prompt)))

        return output if not self.as_score else int(output.yes)

    __call__ = judge  # Allow direct call to judge method


class LLMJudgeRealDialog(LLMJudgeYesNo):
    """
    LLM judge for classifying a dialogue as real (human) or synthetic (machine-generated), with boolean output and feedback.
    Returns an instance of LLMJudgeYesNoOutput.
    """  # noqa: E501
    def __init__(self,
                 model: Union[BaseLanguageModel, str] = None,
                 feedback: bool = False,
                 as_score: bool = False,
                 **llm_kwargs):
        with open(config["prompts"]["evaluation"]["llm_as_judge_real_dialog"], encoding="utf-8") as f:
            prompt_template_real_or_not = f.read()
        super().__init__(prompt_template_real_or_not,
                         model=model,
                         feedback=feedback,
                         as_score=as_score,
                         **llm_kwargs)



class LLMJudgePersonaAttributes(LLMJudgeYesNo):
    """LLM judge for evaluating if a speaker follows the persona attributes in a dialogue."""
    def __init__(self,
                 persona: BasePersona,
                 speaker: str,
                 model: Union[BaseLanguageModel, str] = None,
                 feedback: bool = False,
                 as_score: bool = False,
                 **llm_kwargs):
        with open(config["prompts"]["evaluation"]["llm_as_judge_persona_attributes"], encoding="utf-8") as f:
            prompt_template = f.read()

        prompt_template = prompt_template.render(persona=persona, speaker=speaker)

        super().__init__(prompt_template,
                         model=model,
                         feedback=feedback,
                         as_score=as_score,
                         **llm_kwargs)


class SimilarityScore(BaseMetric, ABC):
    def compute(self, dialog_a: Dialog, dialog_b: Dialog) -> float:
        """
        Compute the similarity score between two dialogs.

        :param dialog_a: The first dialog.
        :param dialog_b: The second dialog.
        :return: A float representing the similarity score.
        """
        raise NotImplementedError("Subclasses should implement this method.")



class LinguisticFeaturesDatasetEvaluator(BaseDatasetEvaluator):
    def __init__(self, features=None, name="linguistic_features"):
        super().__init__()
        self.name = name
        self.features = features or [
            "mean_turn_length", "hesitation_rate", "gunning_fog", "flesch_reading_ease"
        ]
        self.all_results = []

    @staticmethod
    def clean_utterance(text):
        cleaned = re.sub(r'<[^>]*>', '', text)
        cleaned = re.sub(r'\*[^*]*\*', '', cleaned)
        cleaned = re.sub(r'\([^)]*\)', '', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned)
        return cleaned.strip()

    @staticmethod
    def count_syllables(word):
        return max(1, syllables.estimate(word))

    @staticmethod
    def count_complex_words(text):
        words = text.split()
        return sum(1 for word in words if syllables.estimate(word) >= 3), len(words)

    @staticmethod
    def calculate_gunning_fog(text):
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            return 0
        words = re.findall(r'\b[a-zA-Z]+\b', text)
        if not words:
            return 0
        complex_words, total_words = LinguisticFeaturesDatasetEvaluator.count_complex_words(text)
        avg_sentence_length = len(words) / len(sentences)
        complex_word_ratio = (complex_words / total_words) * 100 if total_words > 0 else 0
        fog_index = 0.4 * (avg_sentence_length + complex_word_ratio)
        return fog_index

    @staticmethod
    def calculate_flesch_reading_ease(text):
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            return 0
        words = re.findall(r'\b[a-zA-Z]+\b', text)
        if not words:
            return 0
        total_syllables = sum(LinguisticFeaturesDatasetEvaluator.count_syllables(word) for word in words)
        avg_sentence_length = len(words) / len(sentences)
        avg_syllables_per_word = total_syllables / len(words)
        flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
        return flesch_score

    @staticmethod
    def count_hesitations(text):
        # Exclude the backchannel
        hesitation_patterns = [
            r'\buh+\b',     # uh, uhh, uhhh
            r'\bum+\b',     # um, umm, ummm
            r'\ber+\b',     # er, err, errr
            r'\bahh*\b',    # ah, ahh, ahhh
            r'\bohh*\b',    # oh, ohh, ohhh
            r'\bhmm+\b',    # hmm, hmmm
            r'\bhuh+\b',    # h uh
            r'\bmm+\b',     # mm, mmm
            r'\bmhm+\b',    # mhm, mhmm
            r'\buh\-huh\b',    # uh-huh (backchannel)
            r'\bum-hum+\b',    # um-hum (backchannel)
        ]
        total_hesitations = 0
        text_lower = text.lower()
        for pattern in hesitation_patterns:
            matches = re.findall(pattern, text_lower)
            total_hesitations += len(matches)
        return total_hesitations

    def evaluate(self, dialog, dataset_name=None):
        speaker_stats = {}
        for turn in dialog.turns:
            if not getattr(turn, 'speaker', None) or not getattr(turn, 'text', None):
                continue
            speaker = turn.speaker
            if speaker not in speaker_stats:
                speaker_stats[speaker] = []
            speaker_stats[speaker].append(self.clean_utterance(turn.text))
        results = {"dataset": dataset_name or "unknown"}
        for speaker, utts in speaker_stats.items():
            all_text = " ".join(utts)
            turn_lengths = [len(utt.split()) for utt in utts]
            hesitations = [self.count_hesitations(utt) for utt in utts]
            results[f"{speaker}_mean_turn_length"] = np.mean(turn_lengths)
            # results[f"{speaker}_hesitation_rate"] = sum(hesitations) / max(1, sum(turn_lengths))
            results[f"{speaker}_hesitation_rate"] = (sum(hesitations) / max(1, sum(turn_lengths)) * 100)
            results[f"{speaker}_gunning_fog"] = self.calculate_gunning_fog(all_text)
            results[f"{speaker}_flesch_reading_ease"] = self.calculate_flesch_reading_ease(all_text)
        self.all_results.append(results)
        return results

    def __call__(self, dialogs, dataset_name=None, **kwargs):
        if isinstance(dialogs, list):
            for dialog in dialogs:
                self.evaluate(dialog, dataset_name=dataset_name)
            keys = set(k for res in self.all_results for k in res.keys() if k != "dataset")
            dataset_results = {
                k: np.mean([
                    res[k]
                    for res in self.all_results
                    if (k in res and (dataset_name is None or res["dataset"] == dataset_name))
                ])
                for k in keys
            }
            return dataset_results
        else:
            return self.evaluate(dialogs, dataset_name=dataset_name)

    def plot(self, feature=None, kde_bw=0.3, show=True, save_dir=None, save_stats_csv=True):
        if not self.all_results:
            print("No results to plot. Please run evaluation first.")
            return
        df = pd.DataFrame(self.all_results)
        if feature is None:
            exclude_cols = {"dataset"}
            all_features = [col for col in df.columns if col not in exclude_cols]
            base_names = set("_".join(col.split("_")[1:]) for col in all_features)
        else:
            base_names = [feature]
        stats_all = []
        for base in base_names:
            feature_cols = [col for col in df.columns if base in col]
            if not feature_cols:
                continue
            for f in feature_cols:
                plt.figure(figsize=(8, 5))
                stats = {"feature": f}
                means = {}
                stds = {}
                ax = plt.gca()
                for dataset in df['dataset'].unique():
                    values = df[df['dataset'] == dataset][f].dropna()
                    if len(values) < 2:
                        continue
                    values.plot.kde(bw_method=kde_bw, label=f"{dataset}", ax=ax)
                for i, dataset in enumerate(df['dataset'].unique()):
                    values = df[df['dataset'] == dataset][f].dropna()
                    if len(values) < 2:
                        continue
                    mean = values.mean()
                    std = values.std()
                    color = ax.get_lines()[i].get_color()
                    plt.axvline(mean, linestyle="--", color=color, label=f"{dataset} mean ({mean:.2f})")
                    stats[f"{dataset}_mean"] = mean
                    stats[f"{dataset}_std"] = std
                    means[dataset] = mean
                    stds[dataset] = std
                # sds_away calculation
                if "primock" in means and "ours" in means and stds["primock"] > 0:
                    sds_away = (means["ours"] - means["primock"]) / stds["primock"]
                    stats["sds_away"] = sds_away
                    if sds_away > 0:
                        stats["sds_away_explanation"] = (
                            f"Our dataset is {abs(sds_away):.2f} standard deviations higher than Primock."
                        )
                    else:
                        stats["sds_away_explanation"] = (
                            f"Our dataset is {abs(sds_away):.2f} standard deviations lower than Primock."
                        )
                # plt.xlabel(f)
                plt.xlabel(f"{f} (%)" if "hesitation_rate" in f else f)
                plt.ylabel("Density")
                plt.title(f"KDE plot of {f} by dataset")
                plt.legend()
                plt.grid(alpha=0.3)
                if save_dir:
                    os.makedirs(save_dir, exist_ok=True)
                    plt.savefig(os.path.join(save_dir, f"{f}.png"), dpi=300)
                if show:
                    plt.show()
                plt.close()
                stats_all.append(stats)
        # Save all statistics as CSV
        if save_stats_csv and save_dir:
            stats_df = pd.DataFrame(stats_all)
            stats_csv_path = os.path.join(save_dir, "all_feature_stats.csv")
            stats_df.to_csv(stats_csv_path, index=False)
            print(f"All feature statistics saved to {stats_csv_path}")
        if save_dir:
            print(f"All plots saved to {save_dir}")


class DatasetComparator:
    def __init__(self, evaluators: List[BaseDatasetEvaluator]):
        if not evaluators:
            raise ValueError("No evaluators provided for comparison.")
        for evaluator in evaluators:
            if not isinstance(evaluator, BaseDatasetEvaluator):
                raise TypeError(f"Evaluator {evaluator} is not an instance of `BaseDatasetEvaluator`")

        self.evaluators = evaluators

    def __call__(
        self,
        candidates: Union[str, List[Dialog], List[str], List[List[Dialog]], Dict[str, str], Dict[str, List[Dialog]]],
        digits: int = 2,
        output: str = "table",
    ) -> dict:
        if not candidates:
            raise ValueError("No candidates provided for comparison.")

        if isinstance(candidates, str) or isinstance(candidates, list) and isinstance(candidates[0], Dialog):
            candidates = [candidates]  # Ensure candidates is always a list of datasets (set of dialogues)

        results = {}
        dataset_iterator = candidates.items() if isinstance(candidates, dict) else enumerate(candidates)
        for dataset_name, dataset in dataset_iterator:
            if isinstance(dataset_name, int):
                dataset_name += 1
            results[dataset_name] = {}
            for evaluator in self.evaluators:
                evaluator_name = evaluator.name
                score = evaluator(dataset, dataset_name=dataset_name)
                if isinstance(score, dict):
                    for metric, value in score.items():
                        results[dataset_name][f"{evaluator_name}-{metric}"] = value
                else:
                    results[dataset_name][evaluator_name] = score

        if output == "dict":
            return results
        elif output in ["markdown", "table"]:
            dict_to_table(results, markdown=output == "markdown", format=f".{digits}f")  # sort_by="evaluator_name"
        else:
            raise ValueError(f"Unsupported output format: {output}. Supported formats are "
                             "'dict', 'markdown', and 'table'.")

    def plot(self, show: bool = True, save_folder_path: str = None):
        """
        Plot the results of the evaluators.
        """
        if not self.evaluators:
            raise ValueError("No evaluators to plot.")

        for evaluator in self.evaluators:
            evaluator.plot(show=show,
                           save_path=os.path.join(save_folder_path,
                                                  f"{evaluator.name}.png") if save_folder_path else None)

    compare = __call__  # Allow direct call to compare method
