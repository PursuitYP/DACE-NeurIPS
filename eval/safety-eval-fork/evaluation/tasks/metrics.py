import re
import string


def _normalize_text(
    text: str,
    ignore_case: bool = True,
    ignore_punctuation: bool = True,
) -> str:
    if ignore_case:
        text = text.lower()
    if ignore_punctuation:
        text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_exact_match(
    predictions: list[str],
    references: list[str],
    ignore_case: bool = True,
    ignore_punctuation: bool = True,
) -> dict[str, float]:
    assert len(predictions) == len(references), "predictions and references must have the same length"
    if len(predictions) == 0:
        return {"exact_match": 0.0}

    correct = 0
    for prediction, reference in zip(predictions, references):
        normalized_prediction = _normalize_text(
            str(prediction),
            ignore_case=ignore_case,
            ignore_punctuation=ignore_punctuation,
        )
        normalized_reference = _normalize_text(
            str(reference),
            ignore_case=ignore_case,
            ignore_punctuation=ignore_punctuation,
        )
        if normalized_prediction == normalized_reference:
            correct += 1

    return {"exact_match": correct / len(predictions)}


def get_average(numbers: list[int | bool | float]) -> float:
    if len(numbers) == 0:
        return -1
    if isinstance(numbers[0], bool):
        numbers = [int(x) for x in numbers]
    return sum(numbers) / len(numbers)


def get_f1(numbers: list[dict[str, bool]]) -> float:
    if len(numbers) == 0:
        return -1
    tp = sum([1 for x in numbers if x["pred"] and x["gt"]])
    fp = sum([1 for x in numbers if x["pred"] and not x["gt"]])
    fn = sum([1 for x in numbers if not x["pred"] and x["gt"]])
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    return f1
