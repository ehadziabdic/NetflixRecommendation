from typing import Tuple, List

import numpy as np
import networkx as nx

# 2-hop Jaccard score
def jaccard_2hop_score(user_node: str, movie_node: str, G: nx.Graph) -> float:

    # Basic existence checks
    if user_node not in G:
        raise KeyError(f"user_node {user_node} not in graph")
    if movie_node not in G:
        raise KeyError(f"movie_node {movie_node} not in graph")

    # Find nodes at distance <= 2 from the user
    lengths_u = nx.single_source_shortest_path_length(G, source=user_node, cutoff=2)
    gamma2_u = set()
    for n, dist in lengths_u.items():
        if dist == 2 and G.nodes[n].get("bipartite") == "user":
            gamma2_u.add(n)

    # Find users who liked the movie (distance 1)
    lengths_m1 = nx.single_source_shortest_path_length(G, source=movie_node, cutoff=1)
    gamma1_m = set()
    for n, dist in lengths_m1.items():
        if dist == 1 and G.nodes[n].get("bipartite") == "user":
            gamma1_m.add(n)

    # Compute intersection and union
    intersection = gamma2_u.intersection(gamma1_m)
    union = gamma2_u.union(gamma1_m)

    if len(union) == 0:
        return 0.0
    return len(intersection) / len(union)

# Random Walk with Restart (RWR) scores
def rwr_scores_for_user(user_node: str, G: nx.Graph, alpha: float = 0.15, max_iter: int = 100, tol: float = 1e-6,) -> Tuple[np.ndarray, List[str]]:

    if user_node not in G:
        raise KeyError(f"user_node {user_node} not in graph")

    # Prepare node ordering and adjacency matrix
    nodes = list(G.nodes())
    n = len(nodes)
    index_of = {node: i for i, node in enumerate(nodes)}

    # Build adjacency matrix A (dense) using edge weights if present
    A = nx.to_numpy_array(G, nodelist=nodes, weight="weight", dtype=float)

    # Column sums for normalization (degree per column)
    col_sums = A.sum(axis=0)

    # Build transition matrix P = A.T @ D_inv where D_inv is diagonal(1/col_sums)
    P = np.zeros_like(A)
    A_T = A.T.copy()
    for j in range(n):
        if col_sums[j] > 0:
            # divide column j of A_T by col_sums[j]
            for i in range(n):
                P[i, j] = A_T[i, j] / col_sums[j]
        else:
            # column has zero sum; leave zeros
            for i in range(n):
                P[i, j] = 0.0

    # Initial vector p0: 1 at user_node, zero elsewhere
    p0 = np.zeros(n, dtype=float)
    p0[index_of[user_node]] = 1.0

    # Power-iteration until convergence
    p = p0.copy()
    for _ in range(max_iter):
        p_next = alpha * p0 + (1.0 - alpha) * (P @ p)
        diff = np.linalg.norm(p_next - p, ord=1)
        p = p_next
        if diff < tol:
            break

    return p, nodes

# RWR score for a single movie node
def rwr_score(user_node: str, movie_node: str, G: nx.Graph, **kwargs) -> float:
    p, nodes = rwr_scores_for_user(user_node, G, **kwargs)

    if movie_node not in nodes:
        raise KeyError(f"movie_node {movie_node} not in graph")
    
    idx = nodes.index(movie_node)
    return float(p[idx])

# Unified scoring function
def score(user_node: str, movie_node: str, G: nx.Graph, method: str = "jaccard", **kwargs) -> float:
    method_lower = method.lower()
    
    if method_lower == "jaccard":
        return jaccard_2hop_score(user_node, movie_node, G)
    elif method_lower == "rwr":
        return rwr_score(user_node, movie_node, G, **kwargs)
    else:
        raise ValueError("method must be 'jaccard' or 'rwr'")