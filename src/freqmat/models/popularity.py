"""Item popularity statistics for the debiasing terms of the training objective."""

import torch


def popularity_counts(dataset, n_items: int) -> torch.Tensor | None:
    """Per-item counts of the training targets in ``dataset`` ``(n_items,)``.

    ``dataset`` is RecBole's training split (``train_data.dataset``): one row per training sample, whose ``item_id`` is
    that sample's target. Validation and test targets are not in it, so the counts cannot leak them. Returns ``None``
    when the dataset exposes no interaction table.
    """
    inter = getattr(dataset, "inter_feat", None)
    if inter is None:
        return None
    iid = torch.as_tensor(inter[dataset.iid_field]).long()
    return torch.bincount(iid, minlength=n_items).float()


def head_mask(counts: torch.Tensor, head_frac: float) -> torch.Tensor:
    """Boolean ``(n_items,)`` mask of the most popular ``head_frac`` of the items that were seen in training."""
    seen = int((counts > 0).sum())
    k = max(1, round(head_frac * seen))
    mask = torch.zeros_like(counts, dtype=torch.bool)
    mask[counts.topk(k).indices] = True
    return mask & (counts > 0)
