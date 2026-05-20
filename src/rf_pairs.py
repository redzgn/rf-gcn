import random
import torch


def _local_homophily(node, adj, y_list, train_mask_list):
    """Fraction of labeled 1-hop neighbors that share node's class label."""
    labeled_nbrs = [v for v in adj[node] if train_mask_list[v]]
    if not labeled_nbrs:
        return 1.0  # no evidence — give benefit of the doubt
    same = sum(1 for v in labeled_nbrs if y_list[v] == y_list[node])
    return same / len(labeled_nbrs)


def build_rf_pairs(data, seed=1, min_hop=1, max_hop=3, alpha=0.7,
                   pairs_per_source=16, pair_graph="undirected"):
    """Sample (source, destination) node pairs at graph distances [min_hop, max_hop].

    Sources are labeled training nodes. Destinations are any reachable node
    at the specified hop distances — typically unlabeled nodes not supervised
    by the cross-entropy loss.

    Each pair weight = alpha^hop * source_homophily, where source_homophily is
    the fraction of the source node's labeled 1-hop neighbors that share its class.
    This suppresses label propagation from structurally heterophilic training nodes,
    making the method robust across datasets with varying homophily.

    Returns a dict with tensors on data.x.device:
      src          : source (training) node indices
      dst          : destination node indices
      weight       : alpha^hop * source_homophily
      source_label : y[src], used by hop_weighted_label_loss
      num_pairs    : int
    """
    rng = random.Random(int(seed))
    device = data.x.device
    num_nodes = int(data.num_nodes)

    # Build adjacency list on CPU
    adj = [[] for _ in range(num_nodes)]
    ei = data.edge_index.cpu()
    for u, v in zip(ei[0].tolist(), ei[1].tolist()):
        if u == v:
            continue
        adj[u].append(v)
        if pair_graph == "undirected":
            adj[v].append(u)
    adj = [list(set(nb)) for nb in adj]

    y_list = data.y.cpu().tolist()
    train_mask_list = data.train_mask.cpu().tolist()

    # Pre-compute local homophily for each source node
    sources = torch.where(data.train_mask.cpu())[0].tolist()
    homophily = {s: _local_homophily(s, adj, y_list, train_mask_list) for s in sources}
    rng.shuffle(sources)

    src_all, dst_all, weight_all = [], [], []

    for s in sources:
        hom_w = homophily[s]

        # BFS from source, track exact hop distances
        visited = {s}
        frontier = [s]
        hop_nodes = {}

        for hop in range(1, max_hop + 1):
            nxt = []
            for u in frontier:
                for v in adj[u]:
                    if v not in visited:
                        visited.add(v)
                        nxt.append(v)
            hop_nodes[hop] = nxt
            frontier = nxt
            if not frontier:
                break

        for hop in range(min_hop, max_hop + 1):
            candidates = hop_nodes.get(hop, [])
            if not candidates:
                continue
            rng.shuffle(candidates)
            chosen = candidates[:pairs_per_source]
            for t in chosen:
                src_all.append(s)
                dst_all.append(t)
                weight_all.append(float(alpha) ** hop * hom_w)

    if not src_all:
        el = torch.empty(0, dtype=torch.long, device=device)
        ef = torch.empty(0, dtype=torch.float, device=device)
        return {"src": el, "dst": el, "weight": ef, "source_label": el, "num_pairs": 0}

    src_t = torch.tensor(src_all, dtype=torch.long, device=device)
    dst_t = torch.tensor(dst_all, dtype=torch.long, device=device)
    weight_t = torch.tensor(weight_all, dtype=torch.float, device=device)

    return {
        "src": src_t,
        "dst": dst_t,
        "weight": weight_t,
        "source_label": data.y[src_t],
        "num_pairs": len(src_all),
    }
