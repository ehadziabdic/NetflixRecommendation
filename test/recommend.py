import os
import sys
from typing import List, Dict, Any, Optional

# Ensure we can import project modules from the `src` folder
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.join(REPO_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

import networkx as nx
import scoring as score
from typing import List, Dict, Any


def recommend_for_user(
    G,
    mappings: Dict[str, Dict],
    ratings_df,
    movies_df,
    user_node: str,
    top_n: int = 10,
) -> List[Dict[str, Any]]:
    """Return top-N movie recommendations for a given `user_node`.

    Primary rank: Jaccard 2-hop score. Secondary rank: number of common
    neighbors (`common_2hop`) — users that connect the target user and the
    candidate movie.
    """

    # Movies the user has already seen
    seen_movies = {nbr for nbr in G.neighbors(user_node) if G.nodes[nbr].get("bipartite") == "movie"}

    # All movie nodes in the graph
    all_movies = [n for n, d in G.nodes(data=True) if d.get("bipartite") == "movie"]

    # Candidates: movies not yet seen
    candidates = [m for m in all_movies if m not in seen_movies]

    # Users at distance 2 from the user (user -> movie -> user)
    lengths_u = nx.single_source_shortest_path_length(G, source=user_node, cutoff=2)
    two_hop_users = {n for n, dist in lengths_u.items() if dist == 2 and G.nodes[n].get("bipartite") == "user"}

    # Prepare optional movie lookup by numeric id
    movies_index = movies_df.set_index("movieId") if movies_df is not None else None

    results: List[Dict[str, Any]] = []

    for m in candidates:
        # Jaccard score between two-hop users and likers of movie
        jacc = score.jaccard_2hop_score(user_node, m, G)

        # Users who liked the movie
        lengths_m = nx.single_source_shortest_path_length(G, source=m, cutoff=1)
        likers = {n for n, dist in lengths_m.items() if dist == 1 and G.nodes[n].get("bipartite") == "user"}

        # Supporting users = intersection
        supporting = two_hop_users & likers
        common_2hop = len(supporting)

        # Numeric ids for supporters -> average rating
        avg_rating = None
        if common_2hop > 0 and movies_df is not None:
            supporter_ids = [mappings["node_to_user_id"].get(u) for u in supporting]
            movie_id = mappings["node_to_movie_id"].get(m)
            if movie_id is not None and supporter_ids:
                rows = ratings_df[(ratings_df["movieId"] == movie_id) & (ratings_df["userId"].isin(supporter_ids))]
                if not rows.empty:
                    avg_rating = float(rows["rating"].mean())

        # Movie metadata
        movie_id = mappings["node_to_movie_id"].get(m)
        title = movies_index.loc[movie_id].get("title") if movie_id is not None and movies_index is not None and movie_id in movies_index.index else None

        results.append({
            "movie_node": m,
            "movie_id": movie_id,
            "title": title,
            "jaccard": jacc,
            "common_2hop": common_2hop,
            "avg_rating": avg_rating,
        })

    # Sort by jaccard desc, then by common_2hop desc
    results.sort(key=lambda x: (-(x.get("jaccard") or 0.0), -(x.get("common_2hop") or 0)))
    return results[:top_n]