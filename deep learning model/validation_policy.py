"""Domain-aware validation scores and the deployed Unknown/attack semantics."""
import numpy as np
from sklearn.metrics import f1_score


def domain_slices(size, domains=None):
    domains = domains or {'validation': size}
    if sum(domains.values()) != size or any(count <= 0 for count in domains.values()):
        raise ValueError('Validation domains must partition every row exactly')
    start = 0
    for name, count in domains.items():
        yield name, slice(start, start + count)
        start += count


def macro_scores(truth, predictions, domains=None, labels=None):
    truth, predictions = np.asarray(truth), np.asarray(predictions)
    if len(truth) != len(predictions):
        raise ValueError('Prediction count differs from validation labels')
    return {name: float(f1_score(truth[part], predictions[part], average='macro',
        labels=(labels or {}).get(name), zero_division=0))
        for name, part in domain_slices(len(truth), domains)}


def runtime_attack_mask(probabilities, benign_index, threshold):
    """Unknown is an alert in HybridIDSEngine, including uncertain Benign."""
    probabilities = np.asarray(probabilities)
    winners = probabilities.argmax(axis=1)
    confidences = probabilities[np.arange(len(probabilities)), winners]
    return (winners != benign_index) | (confidences < threshold)


def select_runtime_threshold(probabilities, truth, benign_index, grid, domains=None):
    truth_attack = np.asarray(truth) != benign_index
    best = None
    for threshold in grid:
        prediction = runtime_attack_mask(probabilities, benign_index, threshold)
        scores = {name: float(f1_score(truth_attack[part], prediction[part], zero_division=0))
                  for name, part in domain_slices(len(truth_attack), domains)}
        score = min(scores.values())
        item = {'global_threshold': float(threshold), 'validation_score': score,
                'domain_scores': scores, 'metric': 'minimum_domain_runtime_binary_attack_f1'}
        if best is None or score > best['validation_score']:
            best = item
    if best is None:
        raise ValueError('Threshold grid is empty')
    return best
