from typing import Dict, Tuple, Optional

import pandas as pd
import networkx as nx

# Load ratings and movies data from CSV files
def load_data(ratings_path: str, movies_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:

    # Read CSV files into pandas DataFrames
    ratings = pd.read_csv(ratings_path)
    movies = pd.read_csv(movies_path)

    # Check required columns in ratings
    required_ratings = ["userId", "movieId", "rating"]
    for col in required_ratings:
        if col not in ratings.columns:
            raise ValueError(f"Column '{col}' missing from ratings file")

    # Check required columns in movies
    required_movies = ["movieId", "title"]
    for col in required_movies:
        if col not in movies.columns:
            raise ValueError(f"Column '{col}' missing from movies file")

    # Make copies and normalise dtypes
    ratings = ratings.copy()
    movies = movies.copy()
    ratings["userId"] = ratings["userId"].astype(int)
    ratings["movieId"] = ratings["movieId"].astype(int)
    ratings["rating"] = ratings["rating"].astype(float)
    movies["movieId"] = movies["movieId"].astype(int)

    return ratings, movies

# Downsample users with at least `min_likes` positive ratings (>= threshold)
def downsample_users(
    ratings: pd.DataFrame,
    min_likes: int = 10,
    threshold: float = 3.5,
    sample_n: Optional[int] = None,
) -> pd.DataFrame:

    # Keep only positive interactions
    positive = ratings[ratings["rating"] >= threshold].copy()

    # Count positive likes per user
    counts = positive.groupby("userId").size()

    # Build list of eligible users with enough likes
    eligible_users = []
    for user_id, cnt in counts.items():
        if cnt >= min_likes:
            eligible_users.append(user_id)

    # Optionally downsample the eligible users
    if sample_n is not None and sample_n < len(eligible_users):
        eligible_users = pd.Series(eligible_users).sample(sample_n, random_state=42).tolist()

    # Filter positive interactions to only include eligible users
    filtered = positive[positive["userId"].isin(eligible_users)].copy()
    return filtered

# Build bipartite graph from ratings and movies data
def build_bipartite_graph(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    threshold: float = 3.5,
) -> Tuple[nx.Graph, Dict[str, Dict]]:

    G = nx.Graph()

    # Add user nodes from the ratings DataFrame
    unique_user_ids = ratings["userId"].unique()
    for uid in unique_user_ids:
        node_name = f"u_{uid}"
        G.add_node(node_name, bipartite="user", userId=int(uid))

    # Add ALL movie nodes from movies DataFrame (not just those with ratings)
    movies_index = movies.set_index("movieId")
    for mid in movies["movieId"].unique():
        node_name = f"m_{mid}"
        attrs = {"bipartite": "movie", "movieId": int(mid)}
        if mid in movies_index.index:
            movie_row = movies_index.loc[mid]
            attrs["title"] = movie_row.get("title")
            if "genres" in movies_index.columns:
                attrs["genres"] = movie_row.get("genres")
        G.add_node(node_name, **attrs)

    # Add edges for positive ratings (above threshold)
    positive_ratings = ratings[ratings["rating"] >= threshold]
    for _, row in positive_ratings.iterrows():
        u_node = f"u_{int(row['userId'])}"
        m_node = f"m_{int(row['movieId'])}"
        G.add_edge(u_node, m_node, weight=float(row["rating"]))

    # Build helper mappings explicitly (clear and easy to follow)
    user_id_to_node = {}
    node_to_user_id = {}
    movie_id_to_node = {}
    node_to_movie_id = {}

    for node, data in G.nodes(data=True):
        part = data.get("bipartite")
        if part == "user":
            # node is like 'u_123' -> extract numeric id
            uid = int(node.split("_")[1])
            user_id_to_node[uid] = node
            node_to_user_id[node] = uid
        elif part == "movie":
            mid = int(node.split("_")[1])
            movie_id_to_node[mid] = node
            node_to_movie_id[node] = mid

    mappings = {
        "user_id_to_node": user_id_to_node,
        "node_to_user_id": node_to_user_id,
        "movie_id_to_node": movie_id_to_node,
        "node_to_movie_id": node_to_movie_id,
    }

    return G, mappings

# Validate bipartite graph structure
def validate_graph(G: nx.Graph) -> None:

    user_nodes = []
    movie_nodes = []
    for n, d in G.nodes(data=True):
        if d.get("bipartite") == "user":
            user_nodes.append(n)
        elif d.get("bipartite") == "movie":
            movie_nodes.append(n)

    ucount = len(user_nodes)
    mcount = len(movie_nodes)
    ecount = G.number_of_edges()
    print(f"Graph: {ucount} users, {mcount} movies, {ecount} edges")

    # Ensure edges only connect user <-> movie
    for u, v in G.edges():
        bu = G.nodes[u].get("bipartite")
        bv = G.nodes[v].get("bipartite")
        if bu == bv:
            raise AssertionError(f"Edge between same partition: {u}({bu}) - {v}({bv})")


# Precompute movie likers cache for performance
def precompute_movie_likers(G: nx.Graph) -> Dict[str, set]:
    cache = {}
    movie_nodes = [n for n, d in G.nodes(data=True) if d.get("bipartite") == "movie"]
    for m_node in movie_nodes:
        cache[m_node] = {nbr for nbr in G.neighbors(m_node) if G.nodes[nbr].get("bipartite") == "user"}
    return cache


# Precompute movie genres cache
def precompute_movie_genres(G: nx.Graph) -> Dict[str, set]:
    cache = {}
    movie_nodes = [n for n, d in G.nodes(data=True) if d.get("bipartite") == "movie"]
    for m_node in movie_nodes:
        genres_str = G.nodes[m_node].get("genres")
        if genres_str:
            cache[m_node] = set(str(genres_str).split('|'))
        else:
            cache[m_node] = set()
    return cache


# Build memory-efficient ratings lookup (only for filtered users)
def precompute_ratings_lookup(ratings_df: pd.DataFrame) -> Dict[int, Dict[int, float]]:
    print("      - Building ratings lookup cache...", flush=True)
    ratings_lookup = {}
    
    for user_id in ratings_df['userId'].unique():
        user_ratings = ratings_df[ratings_df['userId'] == user_id]
        ratings_lookup[user_id] = dict(zip(user_ratings['movieId'], user_ratings['rating']))
    
    return ratings_lookup
