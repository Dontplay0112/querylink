"""Evaluation utilities adapted from the A-Mem project.

Upstream: https://github.com/WujiangXu/A-mem/blob/main/utils.py
License and attribution: THIRD_PARTY_NOTICES.md

@inproceedings{xu2025amem,
  title={A-mem: Agentic memory for llm agents},
  author={Xu, Wujiang and Liang, Zujie and Mei, Kai and Gao, Hang and Tan, Juntao and Zhang, Yongfeng},
  booktitle={Advances in Neural Information Processing Systems},
  year={2025}
}
"""

import string
import numpy as np
from typing import List, Dict, Union
import statistics
from collections import defaultdict
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score as bert_score
import nltk
from nltk.translate.meteor_score import meteor_score
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import pytorch_cos_sim
from collections import Counter
import re

def simple_tokenize(text):
    """Simple tokenization function."""
    # Convert to string if not already
    text = str(text)
    return text.lower().replace('.', ' ').replace(',', ' ').replace('!', ' ').replace('?', ' ').split()

def calculate_rouge_scores(prediction: str, reference: str) -> Dict[str, float]:
    """Calculate ROUGE scores for prediction against reference."""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, prediction)
    return {
        'rouge1_f': scores['rouge1'].fmeasure,
        'rouge2_f': scores['rouge2'].fmeasure,
        'rougeL_f': scores['rougeL'].fmeasure
    }

def calculate_bleu_scores(prediction: str, reference: str) -> Dict[str, float]:
    """Calculate BLEU scores with different n-gram settings."""
    pred_tokens = nltk.word_tokenize(prediction.lower())
    ref_tokens = [nltk.word_tokenize(reference.lower())]
    
    weights_list = [(1, 0, 0, 0), (0.5, 0.5, 0, 0), (0.33, 0.33, 0.33, 0), (0.25, 0.25, 0.25, 0.25)]
    smooth = SmoothingFunction().method1
    
    scores = {}
    for n, weights in enumerate(weights_list, start=1):
        try:
            score = sentence_bleu(ref_tokens, pred_tokens, weights=weights, smoothing_function=smooth)
        except Exception as e:
            score = 0.0
        scores[f'bleu{n}'] = score
    return scores

def calculate_bert_scores(prediction: str, reference: str) -> Dict[str, float]:
    """Calculate BERTScore for semantic similarity."""
    try:
        P, R, F1 = bert_score([prediction], [reference], lang='en', verbose=False, rescale_with_baseline=True)
        return {
            'bert_precision': P.item(),
            'bert_recall': R.item(),
            'bert_f1': F1.item()
        }
    except Exception as e:
        print(f"Error calculating BERTScore: {e}")
        return {
            'bert_precision': 0.0,
            'bert_recall': 0.0,
            'bert_f1': 0.0
        }

def calculate_meteor_score(prediction: str, reference: str) -> float:
    """Calculate METEOR score for the prediction."""
    try:
        return meteor_score([reference.split()], prediction.split())
    except Exception as e:
        print(f"Error calculating METEOR score: {e}")
        return 0.0

def calculate_sentence_similarity(prediction: str, reference: str) -> float:
    """Calculate sentence embedding similarity using SentenceBERT."""
    try:
        sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    except Exception as e:
        print(f"Warning: Could not load SentenceTransformer model: {e}")
        sentence_model = None
    
    if sentence_model is None:
        return 0.0
    try:
        # Encode sentences
        embedding1 = sentence_model.encode([prediction], convert_to_tensor=True)
        embedding2 = sentence_model.encode([reference], convert_to_tensor=True)
        
        # Calculate cosine similarity
        similarity = pytorch_cos_sim(embedding1, embedding2).item()
        return float(similarity)
    except Exception as e:
        print(f"Error calculating sentence similarity: {e}")
        return 0.0

def calculate_metrics(prediction: str, reference: str) -> Dict[str, float]:
    """Calculate comprehensive evaluation metrics for a prediction."""
    # Handle empty or None values
    if not prediction or not reference:
        return {
            "exact_match": 0,
            "f1": 0.0,
            "rouge1_f": 0.0,
            "rouge2_f": 0.0,
            "rougeL_f": 0.0,
            "bleu1": 0.0,
            "bleu2": 0.0,
            "bleu3": 0.0,
            "bleu4": 0.0,
            "bert_f1": 0.0,
            "meteor": 0.0,
            "sbert_similarity": 0.0
        }
    
    # Convert to strings if they're not already
    prediction = str(prediction).strip()
    reference = str(reference).strip()
    
    # Calculate exact match
    exact_match = int(prediction.lower() == reference.lower())
    
    # Calculate token-based F1 score
    pred_tokens = set(simple_tokenize(prediction))
    ref_tokens = set(simple_tokenize(reference))
    common_tokens = pred_tokens & ref_tokens
    
    # # NOTE DEBUG 尝试不同的f1
    # def normalize_answer(s):
    #     def remove_articles(text):
    #         return re.sub(r"\b(a|an|the)\b", " ", text)
    #     def white_space_fix(text):
    #         return " ".join(text.split())
    #     def remove_punc(text):
    #         exclude = set(string.punctuation)
    #         return "".join(ch for ch in text if ch not in exclude)
    #     def lower(text):
    #         return text.lower()
    #     return white_space_fix(remove_articles(remove_punc(lower(s))))
    # def f1_score(prediction, ground_truth, **kwargs):
    #     common = Counter(prediction) & Counter(ground_truth)
    #     num_same = sum(common.values())
    #     if num_same == 0:
    #         return 0
    #     precision = 1.0 * num_same / len(prediction)
    #     recall = 1.0 * num_same / len(ground_truth)
    #     f1 = (2 * precision * recall) / (precision + recall)
    #     return f1
    # def qa_f1_score(prediction, ground_truth, **kwargs):
    #     normalized_prediction = normalize_answer(prediction)
    #     normalized_ground_truth = normalize_answer(ground_truth)

    #     prediction_tokens = normalized_prediction.split()
    #     ground_truth_tokens = normalized_ground_truth.split()
    #     return f1_score(prediction_tokens, ground_truth_tokens)
    
    # f1 = qa_f1_score(prediction, reference)
    
    if not pred_tokens or not ref_tokens:
        f1 = 0.0
    else:
        precision = len(common_tokens) / len(pred_tokens)
        recall = len(common_tokens) / len(ref_tokens)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        

    
    # NOTE DEBUG 暂时不计算不重要的指标
    # Calculate all scores
    # rouge_scores = calculate_rouge_scores(prediction, reference)
    bleu_scores = calculate_bleu_scores(prediction, reference)
    # bert_scores = calculate_bert_scores(prediction, reference)
    # meteor = calculate_meteor_score(prediction, reference)
    # sbert_similarity = calculate_sentence_similarity(prediction, reference)
    rouge_scores = [0.0]
    bert_scores = [0.0]
    meteor = 0.0
    sbert_similarity = 0.0
    
    # Combine all metrics
    metrics = {
        "exact_match": exact_match,
        "f1": f1,
        # **rouge_scores,
        **bleu_scores,
        # **bert_scores,
        "meteor": meteor,
        "sbert_similarity": sbert_similarity
    }
    
    return metrics

def aggregate_metrics(all_metrics: List[Dict[str, float]], all_categories: List[int]) -> Dict[str, Dict[str, Union[float, Dict[str, float]]]]:
    """Calculate aggregate statistics for all metrics, split by category."""
    if not all_metrics:
        return {}
    
    # Initialize aggregates for overall and per-category metrics
    aggregates = defaultdict(list)
    category_aggregates = defaultdict(lambda: defaultdict(list))
    
    # Collect all values for each metric, both overall and per category
    for metrics, category in zip(all_metrics, all_categories):
        for metric_name, value in metrics.items():
            aggregates[metric_name].append(value)
            category_aggregates[category][metric_name].append(value)
    
    # Calculate statistics for overall metrics
    results = {
        "overall": {}
    }
    
    for metric_name, values in aggregates.items():
        results["overall"][metric_name] = {
            'mean': statistics.mean(values),
            'std': statistics.stdev(values) if len(values) > 1 else 0.0,
            'median': statistics.median(values),
            'min': min(values),
            'max': max(values),
            'count': len(values)
        }
    
    # Calculate statistics for each category
    for category in sorted(category_aggregates.keys()):
        results[f"category_{category}"] = {}
        for metric_name, values in category_aggregates[category].items():
            if values:  # Only calculate if we have values for this category
                results[f"category_{category}"][metric_name] = {
                    'mean': statistics.mean(values),
                    'std': statistics.stdev(values) if len(values) > 1 else 0.0,
                    'median': statistics.median(values),
                    'min': min(values),
                    'max': max(values),
                    'count': len(values)
                }
    
    return results
