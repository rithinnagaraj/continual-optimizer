"""Metrics for recurrent Domain-IL; accuracies are fractions, gaps intervene."""
from statistics import fmean

from miniimagenet_scale_common import scale_metrics


def continual_metrics(segments, evaluation):
    """Final forgetting is max *previous post-first-exposure* accuracy minus final.

    Signed values are retained (improvement can be negative forgetting).
    Recurrent BWT = mean_d(A_final,d - A_first_training_segment(d),d),
    including all domains; revisits mean this is not single-pass task BWT.
    """
    first = {}
    for row in segments:
        first.setdefault(row["domain_name"], int(row["segment"]))
    final_segment = max(int(row["segment"]) for row in segments)
    per_domain = []
    for domain, first_segment in first.items():
        history = {int(row["checkpoint_segment"]): float(row["accuracy"])
                   for row in evaluation if row["evaluated_domain"] == domain}
        if first_segment not in history or final_segment not in history:
            raise ValueError(f"incomplete accuracy matrix for {domain}")
        final = history[final_segment]
        previous = [value for segment, value in history.items() if first_segment <= segment < final_segment]
        per_domain.append({"domain": domain, "final_accuracy": final,
                           "forgetting": max(previous, default=final) - final,
                           "backward_transfer": final - history[first_segment]})
    return {
        "final_all_domain_accuracy": fmean(row["final_accuracy"] for row in per_domain),
        "mean_forgetting": fmean(row["forgetting"] for row in per_domain),
        "backward_transfer": fmean(row["backward_transfer"] for row in per_domain),
    }, per_domain


def dn4il_metrics(segments, evaluation):
    metrics, domains = continual_metrics(segments, evaluation)
    return {**scale_metrics(segments, total_segments=30), **metrics}, domains
