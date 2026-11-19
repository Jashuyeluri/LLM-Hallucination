LABELS = ("supported", "contradicted", "unsupported", "needs_review")


def compute_faithfulness_score(results):
    """Score support over every claim; expose coverage separately.

    `unsupported` means the supplied source did not establish a claim; it is
    not, by itself, proof that the claim is false.
    """
    counts = {label: sum(1 for item in results if item["label"] == label) for label in LABELS}
    total = len(results)
    resolved = total - counts["needs_review"]
    score = round((counts["supported"] / total) * 100, 2) if total else 0.0
    resolved_score = round((counts["supported"] / resolved) * 100, 2) if resolved else 0.0
    coverage = round((resolved / total) * 100, 2) if total else 0.0
    return {
        "faithfulness_score": score,
        "resolved_faithfulness_score": resolved_score,
        "verification_coverage": coverage,
        "total_claims": total,
        **counts,
    }
