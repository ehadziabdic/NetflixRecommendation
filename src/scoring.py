from typing import List
import networkx as nx

# 2-hop Jaccard score between user node and movie node
def jaccard_2hop_score(user_node: str, movie_node: str, G: nx.Graph) -> float:
    if user_node not in G or movie_node not in G:
        raise KeyError("user_node or movie_node not in graph")

    # Users at distance 2 from the user
    lengths_u = nx.single_source_shortest_path_length(G, source=user_node, cutoff=2)
    users_2hop = {n for n, d in lengths_u.items() if d == 2 and G.nodes[n].get("bipartite") == "user"}

    # Users who liked the movie (neighbors at distance 1)
    lengths_m = nx.single_source_shortest_path_length(G, source=movie_node, cutoff=1)
    likers = {n for n, d in lengths_m.items() if d == 1 and G.nodes[n].get("bipartite") == "user"}

    union = users_2hop | likers
    if not union:
        return 0.0
    return len(users_2hop & likers) / len(union)

# Common neighbors count between user node and movie node
def common_neighbors_count(user_node: str, movie_node: str, G: nx.Graph) -> int:
    if user_node not in G or movie_node not in G:
        raise KeyError("user_node or movie_node not in graph")

    lengths_u = nx.single_source_shortest_path_length(G, source=user_node, cutoff=2)
    users_2hop = {n for n, d in lengths_u.items() if d == 2 and G.nodes[n].get("bipartite") == "user"}

    lengths_m = nx.single_source_shortest_path_length(G, source=movie_node, cutoff=1)
    likers = {n for n, d in lengths_m.items() if d == 1 and G.nodes[n].get("bipartite") == "user"}

    return len(users_2hop & likers)

# General scoring function
def score(user_node: str, movie_node: str, G: nx.Graph, method: str = "jaccard", **kwargs) -> float:
    m = method.lower()
    if m == "jaccard":
        return jaccard_2hop_score(user_node, movie_node, G)
    if m in ("cn", "common_neighbors"):
        return float(common_neighbors_count(user_node, movie_node, G))
    raise ValueError("method must be 'jaccard' or 'cn'/'common_neighbors'")