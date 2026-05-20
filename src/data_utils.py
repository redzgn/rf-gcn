import os
import random
import torch
from torch_geometric.utils import remove_self_loops, add_self_loops


def load_dataset(name, data_root, device="cpu", normalize_self_loops=True):
    path = os.path.join(data_root, name, "data.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")
    data = torch.load(path, map_location="cpu", weights_only=False)
    data.y = data.y.long()
    if normalize_self_loops:
        ei, _ = remove_self_loops(data.edge_index)
        ei, _ = add_self_loops(ei, num_nodes=data.num_nodes)
        data.edge_index = ei
    for mask in ("train_mask", "val_mask", "test_mask"):
        if not hasattr(data, mask):
            raise ValueError(f"{name}: missing {mask}")
    return data.to(device)


def subsample_train_mask(data, labels_per_class, seed=1):
    """Subsample training nodes to at most labels_per_class per class.

    The rest of the original train_mask nodes become unlabeled (neither train
    nor val/test) — they receive no CE supervision but are valid RF pair targets.
    """
    rng = random.Random(int(seed))
    y = data.y.cpu().tolist()
    train_nodes = torch.where(data.train_mask.cpu())[0].tolist()

    by_class = {}
    for n in train_nodes:
        by_class.setdefault(y[n], []).append(n)

    keep = []
    for cls in sorted(by_class):
        nodes = by_class[cls][:]
        rng.shuffle(nodes)
        keep.extend(nodes[:labels_per_class])

    mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=data.x.device)
    mask[keep] = True
    data.train_mask = mask
    return data


def dataset_stats(name, data):
    return {
        "dataset": name,
        "num_nodes": int(data.num_nodes),
        "num_edges": int(data.num_edges),
        "num_features": int(data.num_features),
        "num_classes": int(data.y.max().item()) + 1,
        "train_size": int(data.train_mask.sum().item()),
        "val_size": int(data.val_mask.sum().item()),
        "test_size": int(data.test_mask.sum().item()),
    }
