"""Item popularity statistics for the debiasing terms of the training objective."""

import numpy as np
import torch


def popularity_counts(dataset, n_items: int, hold_out: int = 2) -> torch.Tensor | None:
    """Per-item interaction counts from the training part of ``dataset`` ``(n_items,)``.

    RecBole's ``train.dataset`` still holds every interaction (the leave-one-out split only slices indices), ordered
    by time (``order: TO``). The last ``hold_out`` interactions of each user are the validation and test
    targets, so they are excluded. Returns ``None`` when the dataset exposes no interaction table.
    """
    inter = getattr(dataset, "inter_feat", None)
    if inter is None:
        return None
    uid = np.asarray(inter[dataset.uid_field])
    iid = np.asarray(inter[dataset.iid_field])
    # `order: TO` sorts globally by time; a stable sort by user keeps each user's interactions in time order
    order = np.argsort(uid, kind="stable")
    uid, iid = uid[order], iid[order]
    # position of each interaction counted from the end of its user's history (1 = last)
    boundary = np.flatnonzero(np.diff(uid, append=uid[-1] + 1) != 0)
    last_pos = np.repeat(boundary, np.diff(np.concatenate(([-1], boundary))))
    from_end = last_pos - np.arange(len(uid)) + 1
    keep = from_end > hold_out
    return torch.bincount(torch.as_tensor(iid[keep]), minlength=n_items).float()


def head_mask(counts: torch.Tensor, head_frac: float) -> torch.Tensor:
    """Boolean ``(n_items,)`` mask of the most popular ``head_frac`` of the items that were seen in training."""
    seen = int((counts > 0).sum())
    k = max(1, round(head_frac * seen))
    mask = torch.zeros_like(counts, dtype=torch.bool)
    mask[counts.topk(k).indices] = True
    return mask & (counts > 0)
