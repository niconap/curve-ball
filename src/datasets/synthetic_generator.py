import argparse
import hashlib
import json
import os
import pathlib
import re
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations

import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data, InMemoryDataset

# Synthetic graph generation helpers.
# The module builds four synthetic graph families, computes summary metrics,
# and writes a compact visual summary for each preset.

warnings.filterwarnings("ignore")

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    def tqdm(iterable=None, total=None, **kwargs):
        if iterable is None:
            class _NoOpPbar:
                def update(self, n=1):
                    return None

                def close(self):
                    return None

            return _NoOpPbar()
        return iterable


def _largest_cc(G):
    Gc = nx.Graph(G)
    Gc.remove_edges_from(nx.selfloop_edges(Gc))
    if not nx.is_connected(Gc):
        Gc = Gc.subgraph(max(nx.connected_components(Gc), key=len)).copy()
    return Gc


def spectral_gap(G):
    """Fiedler value: 2nd smallest eigenvalue of the *normalized* Laplacian."""
    Gc = _largest_cc(G)
    A = nx.to_numpy_array(Gc, dtype=float)
    deg = A.sum(axis=1)
    with np.errstate(divide="ignore"):
        inv_sqrt_deg = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    D_inv_sqrt = np.diag(inv_sqrt_deg)
    L = np.eye(A.shape[0]) - D_inv_sqrt @ A @ D_inv_sqrt
    eigvals = np.linalg.eigvalsh(L)
    if len(eigvals) < 2:
        return 0.0
    return float(sorted(eigvals)[1])


def gromov_delta(G, n_samples=1000, seed=42):
    rng = np.random.default_rng(seed)
    Gc = _largest_cc(G)
    nodes = list(Gc.nodes())
    dist = dict(nx.all_pairs_shortest_path_length(Gc))
    diam = nx.diameter(Gc)
    if diam == 0 or len(nodes) < 4:
        return 0.0

    deltas = []
    for _ in range(n_samples):
        u, v, w, x = rng.choice(nodes, size=4, replace=False)
        d = lambda a, b: dist[a][b]
        s = sorted([d(u, v) + d(w, x), d(u, w) + d(v, x), d(u, x) + d(v, w)])
        deltas.append((s[2] - s[1]) / 2.0)

    return float(np.max(deltas)) / diam


def modularity(G):
    if G.number_of_nodes() < 2 or G.number_of_edges() == 0:
        return 0.0
    communities = nx.community.greedy_modularity_communities(G)
    return float(nx.community.modularity(G, communities))


def compute_all_metrics(G):
    Gc = _largest_cc(G)
    return {
        "Avg Degree": float(np.mean([d for _, d in Gc.degree()])),
        "Avg Clustering": float(nx.average_clustering(Gc)),
        "Diameter": float(nx.diameter(Gc)),
        "Avg Path Length": float(nx.average_shortest_path_length(Gc)),
        "Spectral Gap": spectral_gap(G),
        "Modularity": modularity(Gc),
        "Gromov δ": gromov_delta(G),
    }


def _balanced_block_sizes(n, n_blocks):
    base = n // n_blocks
    rem = n % n_blocks
    return [base + (1 if i < rem else 0) for i in range(n_blocks)]


def _ensure_connected(G):
    if nx.is_connected(G):
        return G

    comps = [list(comp) for comp in nx.connected_components(G)]
    for i in range(len(comps) - 1):
        G.add_edge(comps[i][0], comps[i + 1][0])
    return G


def _graph_fingerprint(G):
    A = nx.to_numpy_array(nx.Graph(G), dtype=np.uint8)
    return hashlib.sha1(A.tobytes()).hexdigest()


def _graph_to_data(G):
    Gc = nx.Graph(G)
    Gc.remove_edges_from(nx.selfloop_edges(Gc))
    n = Gc.number_of_nodes()
    edge_index = torch.tensor(list(Gc.edges()), dtype=torch.long).t().contiguous()

    if edge_index.numel() == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 2), dtype=torch.float)
    else:
        reverse_edges = edge_index[[1, 0], :]
        edge_index = torch.cat([edge_index, reverse_edges], dim=1)
        edge_attr = torch.zeros((edge_index.size(1), 2), dtype=torch.float)
        edge_attr[:, 1] = 1.0

    x = torch.ones((n, 1), dtype=torch.float)
    y = torch.zeros((1, 0), dtype=torch.float)
    n_nodes = torch.tensor([n], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, n_nodes=n_nodes)


def _connected_erdos_renyi(n, p, seed, fallback_p=None):
    G = nx.erdos_renyi_graph(n, p=p, seed=seed)
    if nx.is_connected(G):
        return G
    if fallback_p is not None:
        G = nx.erdos_renyi_graph(n, p=fallback_p, seed=seed + 1)
        if nx.is_connected(G):
            return G

    return _ensure_connected(G)


def _make_sbm(n, n_blocks, p_in, p_out, seed):
    actual_blocks = min(n, n_blocks)
    sizes = _balanced_block_sizes(n, actual_blocks)
    probs = [
        [p_in if i == j else p_out for j in range(actual_blocks)]
        for i in range(actual_blocks)
    ]

    G = nx.stochastic_block_model(sizes, probs, seed=seed)
    return _ensure_connected(G)


def _make_path_of_cliques(n, clique_size, seed=None, n_cross_edges=0):
    if n < 2:
        G = nx.Graph()
        G.add_nodes_from(range(n))
        return G

    # Dynamically scale down clique target for extremely small graphs
    actual_clique_size = min(clique_size, max(2, n // 2))
    n_cliques = max(2, n // actual_clique_size)
    sizes = _balanced_block_sizes(n, n_cliques)

    rng = np.random.default_rng(seed)
    G = nx.Graph()
    cursor = 0
    clique_nodes = []
    
    for size in sizes:
        nodes = list(range(cursor, cursor + size))
        clique_nodes.append(nodes)
        cursor += size
        G.add_nodes_from(nodes)
        for u, v in combinations(nodes, 2):
            G.add_edge(u, v)

    for i in range(len(clique_nodes) - 1):
        u = int(rng.choice(clique_nodes[i]))
        v = int(rng.choice(clique_nodes[i + 1]))
        G.add_edge(u, v)

    if n_cross_edges > 0:
        for _ in range(n_cross_edges):
            if len(clique_nodes) < 3:
                break
            i = int(rng.integers(0, len(clique_nodes) - 2))
            j = int(rng.integers(i + 2, len(clique_nodes)))
            u = int(rng.choice(clique_nodes[i]))
            v = int(rng.choice(clique_nodes[j]))
            G.add_edge(u, v)

    return G


def _make_random_regular(n, degree, seed):
    actual_degree = min(degree, n - 1)
    if (n * actual_degree) % 2 != 0:
        actual_degree -= 1

    if actual_degree <= 0:
        G = nx.Graph()
        G.add_nodes_from(range(n))
        return _ensure_connected(G)

    G = nx.random_regular_graph(actual_degree, n, seed=seed)
    return _ensure_connected(G)


def _write_split_artifacts(root_dir, split_name, graphs, metrics):
    raw_dir = os.path.join(root_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    raw_path = os.path.join(raw_dir, f"{split_name}.pt")

    metrics_path = os.path.join(raw_dir, f"{split_name}_metrics.pt")

    torch.save(graphs, raw_path)
    torch.save(metrics, metrics_path)
    return raw_path, metrics_path


def sample_node_count(method="mixture", seed=None):
    rng = np.random.default_rng(seed)

    if method == "mixture":
        ego_support = np.arange(4, 19)
        p_ego = np.exp(-0.25 * (ego_support - 4))
        p_ego /= p_ego.sum()

        comm_support = np.array([12, 14, 16, 18, 20])

        if rng.random() < 0.5:
            return int(rng.choice(ego_support, p=p_ego))
        else:
            return int(rng.choice(comm_support))

    elif method == "smoothed":
        continuous_n = rng.beta(a=1.8, b=2.2) * (20 - 4) + 4
        return int(np.round(continuous_n))

    raise ValueError(f"Unknown sampling method: {method}")


def _is_valid_node_count_for_preset(preset, n):
    if n < 2:
        return False

    kind = preset["kind"]
    if kind == "random_regular":
        degree = preset["degree"]
        return n > degree and (n * degree) % 2 == 0
    return True


def _sample_valid_node_count_for_preset(preset, seed, method="mixture", max_attempts=128):
    rng = np.random.default_rng(seed)
    for _ in range(max_attempts):
        draw_seed = int(rng.integers(0, np.iinfo(np.uint32).max))
        n = sample_node_count(method=method, seed=draw_seed)
        if _is_valid_node_count_for_preset(preset, n):
            return n
    raise RuntimeError("Could not sample a valid node count for preset")


def generate_graph_from_preset(preset, n=80, seed=42):
    kind = preset["kind"]

    if kind == "erdos_renyi":
        return _connected_erdos_renyi(
            n=n,
            p=preset["p"],
            seed=seed,
            fallback_p=preset.get("fallback_p"),
        )
    if kind == "sbm":
        return _make_sbm(
            n=n,
            n_blocks=preset["n_blocks"],
            p_in=preset["p_in"],
            p_out=preset["p_out"],
            seed=seed,
        )
    if kind == "path_of_cliques":
        return _make_path_of_cliques(
            n=n,
            clique_size=preset["clique_size"],
            seed=seed,
            n_cross_edges=preset.get("n_cross_edges", 0),
        )
    if kind == "random_regular":
        return _make_random_regular(
            n=n,
            degree=preset["degree"],
            seed=seed,
        )
    raise ValueError(f"Unknown graph preset kind: {kind}")


class SyntheticGraphDataset(InMemoryDataset):
    def __init__(self, preset_name, split, root, transform=None, pre_transform=None, pre_filter=None):
        self.preset_name = preset_name
        self.split = split
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["train.pt", "val.pt", "test.pt"]

    @property
    def processed_file_names(self):
        return [f"{self.split}.pt"]

    def process(self):
        file_idx = {"train": 0, "val": 1, "test": 2}
        raw_dataset = torch.load(self.raw_paths[file_idx[self.split]])

        data_list = []
        for graph in raw_dataset:
            data = _graph_to_data(graph)
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])


def _split_counts(total, train_ratio, val_ratio, test_ratio):
    ratios = np.array([train_ratio, val_ratio, test_ratio], dtype=float)
    if np.any(ratios < 0):
        raise ValueError("Split ratios must be non-negative")
    if not np.isclose(ratios.sum(), 1.0):
        raise ValueError("Split ratios must sum to 1.0")

    counts = np.floor(ratios * total).astype(int)
    remainder = total - counts.sum()
    for i in range(remainder):
        counts[i % len(counts)] += 1
    return tuple(int(c) for c in counts)


def _generate_unique_samples_parallel(
    preset_name,
    preset,
    n_graphs,
    num_workers,
    base_seed=42,
):
    seen_fingerprints = set()
    graphs = []
    metrics = []

    seed_cursor = 0
    max_seed_tries = max(n_graphs * 200, 1000)

    desc = f"{preset_name[:28]:<28}"
    pbar = tqdm(total=n_graphs, desc=desc, unit="graph")

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        while len(graphs) < n_graphs and seed_cursor < max_seed_tries:
            missing = n_graphs - len(graphs)
            batch_size = max(num_workers, min(missing * 2, 256))

            futures = []
            for _ in range(batch_size):
                if seed_cursor >= max_seed_tries:
                    break
                seed = base_seed + seed_cursor
                seed_cursor += 1
                futures.append(
                    executor.submit(_generate_one_graph_and_metrics, preset, seed)
                )

            for fut in as_completed(futures):
                try:
                    G, m, fp = fut.result()
                except Exception:
                    continue
                if fp in seen_fingerprints:
                    continue
                seen_fingerprints.add(fp)
                graphs.append(G)
                metrics.append(m)
                pbar.update(1)
                if len(graphs) >= n_graphs:
                    break

    pbar.close()

    if len(graphs) < n_graphs:
        raise RuntimeError(
            f"Could not generate {n_graphs} unique graphs for preset '{preset_name}' "
            f"after trying {max_seed_tries} seeds"
        )

    return graphs, metrics


def build_and_store_preset_dataset(
    preset_name,
    preset,
    output_root,
    n_graphs,
    train_ratio,
    val_ratio,
    test_ratio,
    num_workers,
    seed=42,
):
    graphs, metrics = _generate_unique_samples_parallel(
        preset_name=preset_name,
        preset=preset,
        n_graphs=n_graphs,
        num_workers=num_workers,
        base_seed=seed,
    )

    n_train, n_val, n_test = _split_counts(n_graphs, train_ratio, val_ratio, test_ratio)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_graphs)

    split_indices = {
        "train": indices[:n_train],
        "val": indices[n_train:n_train + n_val],
        "test": indices[n_train + n_val:n_train + n_val + n_test],
    }

    preset_root = os.path.join(output_root, _slugify(preset_name))
    os.makedirs(preset_root, exist_ok=True)

    split_metrics = {}
    split_paths = {}
    for split_name, split_idx in split_indices.items():
        split_graphs = [graphs[i] for i in split_idx]
        split_metrics[split_name] = [metrics[i] for i in split_idx]
        split_paths[split_name] = _write_split_artifacts(
            preset_root,
            split_name,
            split_graphs,
            split_metrics[split_name],
        )

    for split_name in split_indices:
        SyntheticGraphDataset(preset_name=preset_name, split=split_name, root=preset_root)

    metadata = {
        "preset_name": preset_name,
        "preset": preset,
        "num_graphs": n_graphs,
        "splits": {
            "train": n_train,
            "val": n_val,
            "test": n_test,
        },
        "split_paths": {
            split_name: {
                "raw": raw_path,
                "metrics": metrics_path,
                "processed": os.path.join(preset_root, "processed", f"{split_name}.pt"),
            }
            for split_name, (raw_path, metrics_path) in split_paths.items()
        },
        "mean_metrics": {
            split_name: _mean_metrics(split_metrics[split_name])
            for split_name in split_metrics
        },
    }

    with open(os.path.join(preset_root, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return preset_root, metadata


DATASET_PRESETS = {
    "very_low_gap / very_low_delta": {
        "kind": "path_of_cliques",
        "clique_size": 8,
        "n_cross_edges": 0,
    },
    "low_gap / medium_delta": {
        "kind": "sbm",
        "n_blocks": 4,
        "p_in": 0.40,
        "p_out": 0.005,
    },
    "medium_gap / high_delta": {
        "kind": "random_regular",
        "degree": 6,
    },
    "high_gap / medium_delta": {
        "kind": "erdos_renyi",
        "p": 0.11,
        "fallback_p": 0.14,
    },
}

DEFAULT_GRAPHS_PER_PRESET = 512
DEFAULT_TRAIN_RATIO = 0.8
DEFAULT_VAL_RATIO = 0.1
DEFAULT_TEST_RATIO = 0.1
DEFAULT_NUM_WORKERS = max(1, min(32, (os.cpu_count() or 1) * 2))

METRICS_ORDER = ["Avg Degree", "Avg Clustering", "Diameter",
                 "Avg Path Length", "Spectral Gap", "Modularity", "Gromov δ"]

REFERENCE = {
    "ego_small": {"Avg Degree": 2.4058, "Avg Clustering": 0.3875,
                  "Diameter": 2.00, "Avg Path Length": 1.5072,
                  "Spectral Gap": 0.7894, "Modularity": 0.0486,
                  "Gromov δ": 0.0701},
    "comm20":    {"Avg Degree": 4.5669, "Avg Clustering": 0.5702,
                  "Diameter": 4.90,  "Avg Path Length": 2.3854,
                  "Spectral Gap": 0.0548, "Modularity": 0.4587,
                  "Gromov δ": 0.1382},
}


def _slugify(text):
    slug = text.lower().replace("δ", "delta")
    slug = re.sub(r"[^a-z0-9]+", "_", slug)
    return slug.strip("_")


def _mean_metrics(metrics_list):
    return {
        k: float(np.mean([m[k] for m in metrics_list]))
        for k in METRICS_ORDER
    }


def _generate_one_graph_and_metrics(preset, seed):
    n = _sample_valid_node_count_for_preset(preset, seed=seed, method="mixture")

    G = generate_graph_from_preset(preset, n=n, seed=seed)
    m = compute_all_metrics(G)
    fp = _graph_fingerprint(G)
    return G, m, fp


def _print_header(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def _validate_args(args, parser):
    if args.n_graphs <= 0:
        parser.error("--n-graphs must be > 0")
    if args.workers <= 0:
        parser.error("--workers must be > 0")
    if args.train_ratio < 0 or args.val_ratio < 0 or args.test_ratio < 0:
        parser.error("Split ratios must be non-negative")
    if not np.isclose(args.train_ratio + args.val_ratio + args.test_ratio, 1.0):
        parser.error("Split ratios must sum to 1.0")


def main(
    n_graphs_per_preset=DEFAULT_GRAPHS_PER_PRESET,
    output_root=None,
    train_ratio=DEFAULT_TRAIN_RATIO,
    val_ratio=DEFAULT_VAL_RATIO,
    test_ratio=DEFAULT_TEST_RATIO,
    num_workers=DEFAULT_NUM_WORKERS,
    seed=42,
):
    if output_root is None:
        output_root = os.path.join(pathlib.Path(os.path.realpath(__file__)).parents[2], "data", "synthetic")

    _print_header("SYNTHETIC DATASET GENERATION")
    print(f"  Graphs per preset: {n_graphs_per_preset}")
    print(f"  Output root: {output_root}")
    print(f"  Split ratios: train={train_ratio:.2f}, val={val_ratio:.2f}, test={test_ratio:.2f}")
    print(f"  Worker threads: {num_workers}")

    os.makedirs(output_root, exist_ok=True)

    for name, preset in DATASET_PRESETS.items():
        print(f"\n▶  Building: {name}")
        preset_root, metadata = build_and_store_preset_dataset(
            preset_name=name,
            preset=preset,
            output_root=output_root,
            n_graphs=n_graphs_per_preset,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            num_workers=num_workers,
            seed=seed,
        )
        print(f"   stored at: {preset_root}")
        for split_name, split_counts in metadata["splits"].items():
            mean_gap = metadata["mean_metrics"][split_name]["Spectral Gap"]
            mean_delta = metadata["mean_metrics"][split_name]["Gromov δ"]
            print(
                f"   {split_name:<5} count={split_counts:<4} "
                f"gap={mean_gap:.4f}  δ={mean_delta:.4f}"
            )

    print("\nDone.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic graph datasets and store them on disk."
    )
    parser.add_argument(
        "--n-graphs",
        type=int,
        default=DEFAULT_GRAPHS_PER_PRESET,
        help=f"Number of graphs to generate per preset (default: {DEFAULT_GRAPHS_PER_PRESET}).",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Root directory where preset dataset folders will be written.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=DEFAULT_TRAIN_RATIO,
        help=f"Training split ratio (default: {DEFAULT_TRAIN_RATIO}).",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=DEFAULT_VAL_RATIO,
        help=f"Validation split ratio (default: {DEFAULT_VAL_RATIO}).",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=DEFAULT_TEST_RATIO,
        help=f"Test split ratio (default: {DEFAULT_TEST_RATIO}).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help=f"Worker threads for parallel generation (default: {DEFAULT_NUM_WORKERS}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base seed used for generation and splitting.",
    )

    args = parser.parse_args()

    _validate_args(args, parser)

    main(
        n_graphs_per_preset=args.n_graphs,
        output_root=args.output_root,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        num_workers=args.workers,
        seed=args.seed,
    )