import torch.nn.functional as F


def hop_weighted_label_loss(logits, pairs):
    """Hop-distance-weighted label propagation loss.

    Pushes each destination node toward its source training node's true label.
    Pair weights = alpha^hop * source_homophily (from rf_pairs.build_rf_pairs).
    Works best on high-homophily graphs (h > 0.75).
    """
    if pairs["num_pairs"] == 0:
        return logits.sum() * 0.0
    per_pair = F.cross_entropy(logits[pairs["dst"]], pairs["source_label"], reduction="none")
    w = pairs["weight"]
    return (per_pair * w).sum() / w.sum().clamp_min(1e-12)


def symmetric_kl_loss(logits, pairs):
    """Symmetric KL prediction-consistency loss.

    Pushes source and destination nodes to have similar class probability
    distributions, without assuming what class that should be.
    Makes no label-consistency assumption — safe on any homophily level.
    Gradients flow through log_p_dst (in the forward term) and
    log_p_src (in the backward term), so both sides are optimized.
    """
    if pairs["num_pairs"] == 0:
        return logits.sum() * 0.0
    src, dst = pairs["src"], pairs["dst"]
    log_p_src = F.log_softmax(logits[src], dim=1)
    log_p_dst = F.log_softmax(logits[dst], dim=1)
    p_src = log_p_src.exp().detach()
    p_dst = log_p_dst.exp().detach()
    kl_fwd = F.kl_div(log_p_dst, p_src, reduction="none").sum(1)
    kl_bwd = F.kl_div(log_p_src, p_dst, reduction="none").sum(1)
    values = 0.5 * (kl_fwd + kl_bwd)
    w = pairs["weight"]
    return (values * w).sum() / w.sum().clamp_min(1e-12)


def rf_loss(logits, pairs, loss_type="label_ce"):
    if loss_type == "label_ce":
        return hop_weighted_label_loss(logits, pairs)
    if loss_type == "symmetric_kl":
        return symmetric_kl_loss(logits, pairs)
    raise ValueError(f"Unknown loss_type: {loss_type}")
