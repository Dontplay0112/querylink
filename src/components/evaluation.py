# """
# Various evaluation utilities.

# Copy from snap-research/locomo project: https://github.com/snap-research/locomo/blob/main/task_eval/evaluation.py
# @article{maharana2024evaluating,
#   title={Evaluating very long-term conversational memory of llm agents},
#   author={Maharana, Adyasha and Lee, Dong-Ho and Tulyakov, Sergey and Bansal, Mohit and Barbieri, Francesco and Fang, Yuwei},
#   journal={arXiv preprint arXiv:2402.17753},
#   year={2024}
# }

# """

# import regex
# import json
# import string
# import unicodedata
# from typing import List
# import numpy as np
# from collections import Counter
# from bert_score import score
# from nltk.stem import PorterStemmer


# from transformers import logging as hf_logging
# # 将 transformers 库的日志级别设置为 ERROR，
# # 这样它就只会显示错误（ERROR），而会隐藏警告（WARNING）和信息（INFO）
# hf_logging.set_verbosity_error()


# ps = PorterStemmer()

# LENGTH_THRESHOLD = 5

# class SimpleTokenizer(object):
#     ALPHA_NUM = r'[\p{L}\p{N}\p{M}]+'
#     NON_WS = r'[^\p{Z}\p{C}]'

#     def __init__(self):
#         """
#         Args:
#             annotators: None or empty set (only tokenizes).
#         """
#         self._regexp = regex.compile(
#             '(%s)|(%s)' % (self.ALPHA_NUM, self.NON_WS),
#             flags=regex.IGNORECASE + regex.UNICODE + regex.MULTILINE
#         )

#     def tokenize(self, text, uncased=False):
#         matches = [m for m in self._regexp.finditer(text)]
#         if uncased:
#             tokens = [m.group().lower() for m in matches]
#         else:
#             tokens = [m.group() for m in matches]
#         return tokens


# def check_answer(example, tokenizer) -> List[bool]:
#     """Search through all the top docs to see if they have any of the answers."""
#     answers = example['answers']
#     ctxs = example['ctxs']

#     hits = []

#     for _, doc in enumerate(ctxs):
#         text = doc['text']

#         if text is None:  # cannot find the document for some reason
#             hits.append(False)
#             continue

#         hits.append(has_answer(answers, text, tokenizer))

#     return hits


# def has_answer(answers, text, tokenizer=SimpleTokenizer()) -> bool:
#     """Check if a document contains an answer string."""
#     text = _normalize(text)
#     text = tokenizer.tokenize(text, uncased=True)

#     for answer in answers:
#         answer = _normalize(answer)
#         answer = tokenizer.tokenize(answer, uncased=True)
#         for i in range(0, len(text) - len(answer) + 1):
#             if answer == text[i: i + len(answer)]:
#                 return True
#     return False


# def _normalize(text):
#     return unicodedata.normalize('NFD', text)


# def normalize_answer(s):

#     s = s.replace(',', "")
#     def remove_articles(text):
#         # return regex.sub(r'\b(a|an|the)\b', ' ', text)
#         return regex.sub(r'\b(a|an|the|and)\b', ' ', text)

#     def white_space_fix(text):
#         return ' '.join(text.split())

#     def remove_punc(text):
#         exclude = set(string.punctuation)
#         return ''.join(ch for ch in text if ch not in exclude)

#     def lower(text):
#         return text.lower()

#     return white_space_fix(remove_articles(remove_punc(lower(s))))


# def exact_match_score(prediction, ground_truth):

#     prediction = normalize_answer(prediction)
#     ground_truth = normalize_answer(ground_truth)
#     # print('# EM #', prediction, ' | ', ground_truth, ' #', set(prediction.split()) == set(ground_truth.split()))
#     # return normalize_answer(prediction) == normalize_answer(ground_truth)
#     return set(prediction.split()) == set(ground_truth.split())
    
# # def bert_score(prediction, ground_truths):
# #     prediction = normalize_answer(prediction)
# #     values = []
# #     for ground_truth in ground_truths:
# #         ground_truth = normalize_answer(ground_truth)
# #         P, R, F1 = score([prediction], [ground_truth], lang='en', verbose=False, rescale_with_baseline=True)
# #         values.append(R[0].item())
# #     print('# BERT # ', normalize_answer(prediction), ' | ', normalize_answer(ground_truth), ' #', P, R, F1)
# #     return max(0, max(values))


# # TODO COMMENT BACK IN
# # def bert_score(prediction, ground_truth):
# #     prediction = normalize_answer(prediction)
# #     ground_truth = normalize_answer(ground_truth)
# #     P, R, F1 = score([prediction], [ground_truth], lang='en', verbose=False, rescale_with_baseline=True)
# #     # print('# BERT # ', normalize_answer(prediction), ' | ', normalize_answer(ground_truth), ' #', P, R, F1)
# #     return max(0, F1[0].item())


# def ems(prediction, ground_truths):
#     return max([exact_match_score(prediction, gt) for gt in ground_truths])


# def f1_score(prediction, ground_truth):
#     prediction_tokens = [ps.stem(w) for w in normalize_answer(prediction).split()]
#     ground_truth_tokens = [ps.stem(w) for w in normalize_answer(ground_truth).split()]
#     common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
#     num_same = sum(common.values())
#     if num_same == 0:
#         return 0
#     precision = 1.0 * num_same / len(prediction_tokens)
#     recall = 1.0 * num_same / len(ground_truth_tokens)
#     f1 = (2 * precision * recall) / (precision + recall)
#     # print('# F1 #', prediction, ' | ', ground_truth, ' #', precision, recall, f1)
#     # return recall
#     return f1


# def f1(prediction, ground_truth):
#     predictions = [p.strip() for p in prediction.split(',')]
#     ground_truths = [g.strip() for g in ground_truth.split(',')]
#     # print('# F1 [multi-answer]#', predictions, ' | ', ground_truths, ' #', np.mean([max([f1_score(prediction, gt) for prediction in predictions]) for gt in ground_truths]))
#     return np.mean([max([f1_score(prediction, gt) for prediction in predictions]) for gt in ground_truths])


# def rougel_score(prediction, ground_truth):
#     from rouge import Rouge
#     rouge = Rouge()
#     prediction = ' '.join([ps.stem(w) for w in normalize_answer(prediction).split()])
#     ground_truth = ' '.join([ps.stem(w) for w in normalize_answer(ground_truth).split()])
#     # no normalization
#     try:
#         scores = rouge.get_scores(prediction, ground_truth, avg=True)
#     except ValueError:  # "Hypothesis is empty."
#         return 0.0
#     return scores["rouge-1"]["f"]


# def rl(prediction, ground_truths):
#     return max([rougel_score(prediction, gt) for gt in ground_truths])


# ## file-level evaluation ... ### 
# def eval_recall(infile):

#     tokenizer = SimpleTokenizer()
#     lines = open(infile, 'r').readlines()[1:]

#     has_answer_count = 0
#     answer_lengths = []
#     for line in lines:
#         line = json.loads(line)
#         answer = line['answer']
#         output = ' || '.join(line['output'])

#         if has_answer(answer, output, tokenizer):
#             has_answer_count += 1

#         answer_lengths.append(len(output.split()))

#     recall = round(has_answer_count/len(lines), 4)
#     lens = round(np.mean(answer_lengths), 4)

#     return recall, lens


# def eval_question_answering(qas, eval_key='prediction', metric='f1'):
#     all_ems = []
#     for i, line in enumerate(qas):
#         # line = json.loads(line)
#         if type(line[eval_key]) == list:
#             answer = line['answer']
#         else:
#             answer = str(line['answer'])
#         if line['questionType'] == 3:
#             answer = answer.split(';')[0].strip()
        
#         output = line[eval_key]
        
#         # single-hop, temporal, open-domain eval without splitting for sub-answers 
#         if int(line['questionType']) in [2, 3, 4]:
#             all_ems.append(f1_score(output, answer))
        
#         # multi-hop eval by splitting entire phrase into sub-answers and computing partial F1 for each
#         elif int(line['questionType']) in [1]:
#             all_ems.append(f1(output, answer))

#         # adversarial eval --> check for selection of correct option
#         elif int(line['questionType']) in [5]:
#             if ('no information available' in output.lower()
#                 or 'not mentioned' in output.lower()
#                 or ('no' in output.lower() and 'mention' in output.lower())
#                 ):
#                 all_ems.append(1)
#             else:
#                 all_ems.append(0)
#         else:
#             print(line)
#             raise ValueError
        
#         assert i+1 == len(all_ems), all_ems
        
#     # recall has been computed during retrieval stage, skip here
#     return all_ems


# def eval_fact_checking(infile):

#     tokenizer = SimpleTokenizer()
#     lines = open(infile, 'r').readlines()[1:]

#     exact_match_count = 0
#     answer_lengths = []
#     for line in lines:
#         line = json.loads(line)
#         answer = line['answer']
#         output = line['output'][0]

#         if answer == ["refutes"]:
#             answer = ["refutes", "no", "false"]
#         if answer == ["supports"]:
#             answer = ["supports", "yes", "true"]

#         if has_answer(answer, output, tokenizer):
#             exact_match_count += 1
        
#         answer_lengths.append(len(output.split()))

#     em = round(exact_match_count/len(lines), 4)
#     lens = round(np.mean(answer_lengths), 4)

#     return em, lens


# def eval_dialogue_system(infile):

#     lines = open(infile, 'r').readlines()[1:]

#     f1_scores = []
#     rl_scores = []
#     answer_lengths = []
#     for line in lines:
#         line = json.loads(line)
#         answer = line['answer']
#         output = line['output'][0]

#         f1_scores.append(f1(output, answer))
#         rl_scores.append(rl(output, answer))
#         answer_lengths.append(len(output.split()))

#     F1 = round(np.mean(f1_scores), 4)
#     RL = round(np.mean(rl_scores), 4)
#     lens = round(np.mean(answer_lengths), 4)

#     return F1, RL, lens



"""
Various evaluation utilities.

Copy from WujiangXu/A-mem project: https://github.com/WujiangXu/A-mem/blob/main/utils.py
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
