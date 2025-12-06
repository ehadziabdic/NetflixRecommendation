import os
import sys
from typing import List, Dict, Any, Optional

from flask import Flask, render_template, request

# Ensure we can import project modules from the `src` folder
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.join(REPO_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

import graph as graph_mod
import scoring as scoring_mod
import networkx as nx

app = Flask(__name__)


# Load data and build graph once at startup
RATINGS_PATH = os.path.join(REPO_ROOT, "res", "ratings.csv")
MOVIES_PATH = os.path.join(REPO_ROOT, "res", "movies.csv")

# Load CSVs
RATINGS_DF, MOVIES_DF = graph_mod.load_data(RATINGS_PATH, MOVIES_PATH)

# Optionally downsample to keep web app responsive
RATINGS_FILTERED = graph_mod.downsample_users(RATINGS_DF, min_likes=10, threshold=3.5, sample_n=500)

# Build bipartite graph and mappings
G, MAPPINGS = graph_mod.build_bipartite_graph(RATINGS_FILTERED, MOVIES_DF, threshold=3.5)
graph_mod.validate_graph(G)

# Prepare lists for form selects
USER_NODES = [n for n, d in G.nodes(data=True) if d.get("bipartite") == "user"]
MOVIE_NODES = [n for n, d in G.nodes(data=True) if d.get("bipartite") == "movie"]

# Helper to get likers of a movie node
def get_likers_of_movie_node(m_node: str) -> set:
    return {nbr for nbr in G.neighbors(m_node) if G.nodes[nbr].get("bipartite") == "user"}


# Recommend movies for a given user node
def get_recommendations_for_user_node(user_node: str, top_n: int = 10, genre_filter: Optional[str] = None) -> List[Dict[str, Any]]:
    # Movies already seen
    seen = set()
    for nbr in G.neighbors(user_node):
        if G.nodes[nbr].get("bipartite") == "movie":
            seen.add(nbr)

    # Candidate movies
    candidates = []
    for m in MOVIE_NODES:
        if m not in seen:
            # optional genre filter
            if genre_filter and genre_filter != "All":
                genres = G.nodes[m].get("genres")
                if not genres or genre_filter not in str(genres):
                    continue
            candidates.append(m)

    results = []

    # Users at distance 2 from the user
    lengths_u = nx.single_source_shortest_path_length(G, source=user_node, cutoff=2)
    two_hop_users = {n for n, dist in lengths_u.items() if dist == 2 and G.nodes[n].get("bipartite") == "user"}

    for m in candidates:
        # Jaccard score
        jacc = scoring_mod.jaccard_2hop_score(user_node, m, G)

        # Supporters: users who liked the movie and intersection with two-hop users
        likers = get_likers_of_movie_node(m)
        supporters = two_hop_users & likers
        common_2hop = len(supporters)

        # Average rating among supporters
        movie_id = MAPPINGS["node_to_movie_id"].get(m)
        avg_rating = None
        if movie_id is not None and supporters:
            supporter_ids = [MAPPINGS["node_to_user_id"].get(u) for u in supporters]
            rows = RATINGS_DF[(RATINGS_DF["movieId"] == movie_id) & (RATINGS_DF["userId"].isin(supporter_ids))]
            if not rows.empty:
                avg_rating = float(rows["rating"].mean())

        results.append({
            "movie_node": m,
            "movie_id": movie_id,
            "title": G.nodes[m].get("title"),
            "genres": G.nodes[m].get("genres"),
            "jaccard": jacc,
            "common_2hop": common_2hop,
            "avg_rating": avg_rating,
        })

    # Sort by jaccard then common_2hop
    def _sort_key(item: Dict[str, Any]):
        primary = item.get("jaccard") or 0.0
        secondary = item.get("common_2hop") or 0
        return (-primary, -secondary)

    results.sort(key=_sort_key)

    return results[:top_n]


def get_recommendations_for_liked_movies(liked_movie_ids: List[int], top_n: int = 10, genre_filter: Optional[str] = None) -> List[Dict[str, Any]]:
    # Build union of likers for selected movies
    selected_movie_nodes = [f"m_{mid}" for mid in liked_movie_ids if f"m_{mid}" in MOVIE_NODES]
    user_set = set()
    for m_node in selected_movie_nodes:
        for nbr in G.neighbors(m_node):
            if G.nodes[nbr].get("bipartite") == "user":
                user_set.add(nbr)

    # Candidate movies exclude the selected ones
    candidates = [m for m in MOVIE_NODES if m not in selected_movie_nodes]
    results = []
    for m in candidates:
        # genre filter
        if genre_filter and genre_filter != "All":
            genres = G.nodes[m].get("genres")
            if not genres or genre_filter not in str(genres):
                continue

        # likers of candidate
        likers = set(get_likers_of_movie_node(m))

        # compute Jaccard between user_set and likers
        intersection = user_set.intersection(likers)
        union = user_set.union(likers)
        jacc = 0.0
        if len(union) > 0:
            jacc = len(intersection) / len(union)

        # average rating among intersection
        avg_rating = None
        if intersection:
            supporter_ids = [MAPPINGS["node_to_user_id"].get(u) for u in intersection]
            movie_id = MAPPINGS["node_to_movie_id"].get(m)
            if movie_id is not None:
                rows = RATINGS_DF[(RATINGS_DF["movieId"] == movie_id) & (RATINGS_DF["userId"].isin(supporter_ids))]
                if not rows.empty:
                    avg_rating = float(rows["rating"].mean())

        # Ensure result dict includes same keys as user-based recommendations
        results.append({
            "movie_node": m,
            "movie_id": MAPPINGS["node_to_movie_id"].get(m),
            "title": G.nodes[m].get("title"),
            "genres": G.nodes[m].get("genres"),
            "jaccard": jacc,
            "common_2hop": len(intersection),
            "avg_rating": avg_rating,
        })

    # sort by jaccard desc
    results.sort(key=lambda x: -x["jaccard"])
    return results[:top_n]


@app.route("/", methods=["GET"])
def index():
    # Provide simple lists for form selects; movies list uses (id, title)
    movie_options = []
    for m in MOVIE_NODES:
        mid = MAPPINGS["node_to_movie_id"].get(m)
        title = G.nodes[m].get("title")
        movie_options.append((mid, title))

    user_options = []
    for u in USER_NODES:
        uid = MAPPINGS["node_to_user_id"].get(u)
        user_options.append(uid)

    # Simple genre list (collect a few from movies_df)
    genres = ["All"]
    if MOVIES_DF is not None and "genres" in MOVIES_DF.columns:
        # collect unique genre tokens (split on |) — simple heuristic
        gset = set()
        for val in MOVIES_DF["genres"].dropna().unique():
            parts = str(val).split("|")
            for p in parts:
                gset.add(p)
        genres.extend(sorted(gset))

    return render_template("index.html", users=user_options, movies=movie_options, genres=genres, results=None)


@app.route("/recommend", methods=["POST"])
def recommend():
    # Read form values
    user_id = request.form.get("user_id")
    top_n = int(request.form.get("top_n", 10))
    genre = request.form.get("genre", "All")

    # Multi-select liked movies comes as list of strings of numeric ids
    liked = request.form.getlist("liked_movies")
    liked_movie_ids = [int(x) for x in liked if x]

    results = None
    if user_id:
        try:
            user_node = f"u_{int(user_id)}"
            results = get_recommendations_for_user_node(user_node, top_n=top_n, genre_filter=genre)
        except Exception as e:
            results = []
    elif liked_movie_ids:
        results = get_recommendations_for_liked_movies(liked_movie_ids, top_n=top_n, genre_filter=genre)
    else:
        results = []

    # Render same index template with results
    movie_options = [(MAPPINGS["node_to_movie_id"].get(m), G.nodes[m].get("title")) for m in MOVIE_NODES]
    user_options = [MAPPINGS["node_to_user_id"].get(u) for u in USER_NODES]
    genres = ["All"]
    if MOVIES_DF is not None and "genres" in MOVIES_DF.columns:
        gset = set()
        for val in MOVIES_DF["genres"].dropna().unique():
            parts = str(val).split("|")
            for p in parts:
                gset.add(p)
        genres.extend(sorted(gset))

    return render_template("index.html", users=user_options, movies=movie_options, genres=genres, results=results)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
