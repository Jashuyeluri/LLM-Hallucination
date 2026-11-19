import bisect
import csv
import json
import os
import random
from collections import defaultdict

import settings

_cache = {"path": None, "mtime": None, "calibrator": None}


class Calibrator:
    def __init__(self, starts, values):
        self.starts = list(starts)
        self.values = list(values)

    @classmethod
    def fit(cls, confidences, correct):
        if len(confidences) < 2 or len(confidences) != len(correct):
            raise ValueError("Need at least two (confidence, correct) pairs of equal length.")
        pooled = {}
        for confidence, flag in zip(confidences, correct):
            total, count = pooled.get(confidence, (0.0, 0))
            pooled[confidence] = (total + (1.0 if flag else 0.0), count + 1)
        blocks = []
        for confidence in sorted(pooled):
            total, count = pooled[confidence]
            blocks.append({"sum": total, "weight": count, "start": confidence})
            while len(blocks) > 1 and blocks[-2]["sum"] / blocks[-2]["weight"] > blocks[-1]["sum"] / blocks[-1]["weight"]:
                last = blocks.pop()
                blocks[-1]["sum"] += last["sum"]
                blocks[-1]["weight"] += last["weight"]
        return cls([block["start"] for block in blocks], [block["sum"] / block["weight"] for block in blocks])

    def predict(self, confidence):
        position = bisect.bisect_right(self.starts, confidence) - 1
        return self.values[max(position, 0)]

    def to_dict(self):
        return {"method": "isotonic", "starts": self.starts, "values": self.values}

    @classmethod
    def from_dict(cls, payload):
        return cls(payload["starts"], payload["values"])

    def save(self, path):
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))


def get_calibrator():
    path = settings.calibrator_path()
    if not os.path.exists(path):
        _cache.update(path=None, mtime=None, calibrator=None)
        return None
    mtime = os.path.getmtime(path)
    if _cache["path"] != path or _cache["mtime"] != mtime:
        _cache.update(path=path, mtime=mtime, calibrator=Calibrator.load(path))
    return _cache["calibrator"]


def reliability_bins(confidences, correct, bins=10):
    buckets = [[] for _ in range(bins)]
    for confidence, flag in zip(confidences, correct):
        index = min(int(confidence * bins), bins - 1)
        buckets[index].append((confidence, 1.0 if flag else 0.0))
    summary = []
    for index, bucket in enumerate(buckets):
        if not bucket:
            continue
        summary.append({
            "bin": index,
            "count": len(bucket),
            "mean_confidence": sum(item[0] for item in bucket) / len(bucket),
            "accuracy": sum(item[1] for item in bucket) / len(bucket),
        })
    return summary


def expected_calibration_error(confidences, correct, bins=10):
    total = len(confidences)
    if total == 0:
        return 0.0
    return sum(
        item["count"] / total * abs(item["accuracy"] - item["mean_confidence"])
        for item in reliability_bins(confidences, correct, bins)
    )


def collect_pairs(csv_path, model=None, mode=None):
    from ensemble import verify_claims

    grouped = defaultdict(list)
    with open(csv_path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            grouped[row["source_text"]].append((row["claim"], row["gold_label"].strip().lower()))

    pairs = []
    for source_text, items in grouped.items():
        claims = [claim for claim, _ in items]
        verification = verify_claims(source_text, claims, model=model, mode=mode, use_calibration=False)
        for result, (_, gold) in zip(verification["results"], items):
            if result["label"] == "needs_review":
                continue
            pairs.append((result["confidence"], result["label"] == gold))
    return pairs


def split_pairs(pairs, holdout, seed=0):
    shuffled = list(pairs)
    random.Random(seed).shuffle(shuffled)
    cut = int(len(shuffled) * (1 - holdout))
    return shuffled[:cut], shuffled[cut:]
