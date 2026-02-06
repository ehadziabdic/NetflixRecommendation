from typing import List, Set, Dict
import networkx as nx

# Optimized 2-hop Jaccard score between user node and movie node
def jaccard_2hop_score(user_node: str, movie_node: str, G: nx.Graph, 
                       users_2hop: Set[str] = None, likers: Set[str] = None) -> float:
    """
    Compute Jaccard similarity for 2-hop connections.
    
    Optimized: Accepts pre-computed users_2hop and likers sets to avoid redundant calculations.
    """
    if user_node not in G or movie_node not in G:
        raise KeyError("user_node or movie_node not in graph")

    # Use pre-computed sets if provided, otherwise compute
    if users_2hop is None:
        lengths_u = nx.single_source_shortest_path_length(G, source=user_node, cutoff=2)
        users_2hop = {n for n, d in lengths_u.items() if d == 2 and G.nodes[n].get("bipartite") == "user"}
    
    if likers is None:
        # Direct neighbors are more efficient than shortest path for distance 1
        likers = {n for n in G.neighbors(movie_node) if G.nodes[n].get("bipartite") == "user"}

    # Fast intersection and union
    intersection = users_2hop & likers
    union_size = len(users_2hop) + len(likers) - len(intersection)
    
    if union_size == 0:
        return 0.0
    return len(intersection) / union_size

# Optimized common neighbors count
def common_neighbors_count(user_node: str, movie_node: str, G: nx.Graph,
                          users_2hop: Set[str] = None, likers: Set[str] = None) -> int:
    """
    Count common neighbors (users at distance 2 from user who also liked the movie).
    
    Optimized: Accepts pre-computed users_2hop and likers sets to avoid redundant calculations.
    """
    if user_node not in G or movie_node not in G:
        raise KeyError("user_node or movie_node not in graph")

    # Use pre-computed sets if provided, otherwise compute
    if users_2hop is None:
        lengths_u = nx.single_source_shortest_path_length(G, source=user_node, cutoff=2)
        users_2hop = {n for n, d in lengths_u.items() if d == 2 and G.nodes[n].get("bipartite") == "user"}
    
    if likers is None:
        likers = {n for n in G.neighbors(movie_node) if G.nodes[n].get("bipartite") == "user"}

    return len(users_2hop & likers)

# General scoring function
def score(user_node: str, movie_node: str, G: nx.Graph, method: str = "jaccard", **kwargs) -> float:
    m = method.lower()
    if m == "jaccard":
        return jaccard_2hop_score(user_node, movie_node, G, **kwargs)
    if m in ("cn", "common_neighbors"):
        return float(common_neighbors_count(user_node, movie_node, G, **kwargs))
    raise ValueError("method must be 'jaccard' or 'cn'/'common_neighbors'")

# Batch scoring for multiple candidates (highly optimized)
def batch_score(user_node: str, candidate_movies: List[str], G: nx.Graph, 
                method: str = "jaccard") -> Dict[str, float]:

    if user_node not in G:
        raise KeyError(f"user_node '{user_node}' not in graph")
    
    # Pre-compute 2-hop users once for all candidates
    lengths_u = nx.single_source_shortest_path_length(G, source=user_node, cutoff=2)
    users_2hop = {n for n, d in lengths_u.items() if d == 2 and G.nodes[n].get("bipartite") == "user"}
    
    scores = {}
    for movie_node in candidate_movies:
        if movie_node not in G:
            scores[movie_node] = 0.0
            continue
        
        # Get likers efficiently
        likers = {n for n in G.neighbors(movie_node) if G.nodes[n].get("bipartite") == "user"}
        
        # Calculate score
        if method.lower() == "jaccard":
            intersection = users_2hop & likers
            union_size = len(users_2hop) + len(likers) - len(intersection)
            scores[movie_node] = len(intersection) / union_size if union_size > 0 else 0.0
        else:  # common neighbors
            scores[movie_node] = float(len(users_2hop & likers))
    
    return scores