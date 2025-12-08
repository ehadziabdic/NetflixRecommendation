import os
import sys
from typing import List, Dict, Any, Optional

from flask import Flask, render_template, request, session

# Ensure we can import project modules from the `src` folder
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.join(REPO_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

import graph as graph_mod
import scoring as scoring_mod
import graphvis
import networkx as nx

app = Flask(__name__)
# Use environment variable for secret key in production
app.secret_key = os.environ.get('SECRET_KEY', 'dev_key_change_in_production')


# Load data and build graph once at startup
RATINGS_PATH = os.path.join(REPO_ROOT, "res", "ratings.csv")
MOVIES_PATH = os.path.join(REPO_ROOT, "res", "movies.csv")

# Load CSVs
RATINGS_DF, MOVIES_DF = graph_mod.load_data(RATINGS_PATH, MOVIES_PATH)

# Use all users (no downsampling)
RATINGS_FILTERED = graph_mod.downsample_users(RATINGS_DF, min_likes=10, threshold=3.5, sample_n=None)

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


# Recommend movies for a given user node or liked movies
def get_recommendations_for_liked_movies(
    liked_movie_ids: List[int], 
    top_n: int = 10, 
    genre_filter: Optional[str] = None,
    rating_limit: float = 0.0,
    algorithm: str = "jaccard",
    prioritize_rating: bool = False
) -> List[Dict[str, Any]]:
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
        # Genre filter
        if genre_filter and genre_filter != "All":
            genres = G.nodes[m].get("genres")
            if not genres or genre_filter not in str(genres):
                continue

        # Likers of candidate
        likers = set(get_likers_of_movie_node(m))

        # Compute score based on algorithm
        intersection = user_set.intersection(likers)
        union = user_set.union(likers)
        
        if algorithm == "cn":
            # Common neighbors
            score = len(intersection)
        else:
            # Jaccard (default)
            score = len(intersection) / len(union) if len(union) > 0 else 0.0

        # Average rating among intersection
        avg_rating = None
        if intersection:
            supporter_ids = [MAPPINGS["node_to_user_id"].get(u) for u in intersection]
            movie_id = MAPPINGS["node_to_movie_id"].get(m)
            if movie_id is not None:
                rows = RATINGS_DF[(RATINGS_DF["movieId"] == movie_id) & (RATINGS_DF["userId"].isin(supporter_ids))]
                if not rows.empty:
                    avg_rating = float(rows["rating"].mean())

        # Apply rating limit filter
        if avg_rating is not None and avg_rating < rating_limit:
            continue

        # Apply rating weight if prioritize_rating is True
        final_score = score
        if prioritize_rating and avg_rating is not None:
            # Weight score by normalized rating (0-1 scale)
            rating_weight = avg_rating / 5.0
            final_score = score * rating_weight

        results.append({
            "movie_node": m,
            "movie_id": MAPPINGS["node_to_movie_id"].get(m),
            "title": G.nodes[m].get("title"),
            "genres": G.nodes[m].get("genres"),
            "jaccard": score if algorithm == "jaccard" else None,
            "common_2hop": len(intersection),
            "avg_rating": avg_rating,
            "score": final_score
        })

    # Sort by final score descending
    results.sort(key=lambda x: -x["score"])
    return results[:top_n]


@app.route("/", methods=["GET"])
def index():
    # Provide movie list
    movie_options = []
    for m in MOVIE_NODES:
        mid = MAPPINGS["node_to_movie_id"].get(m)
        title = G.nodes[m].get("title")
        movie_options.append((mid, title))

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

    # Restore liked movies from session if returning from results/graph page
    liked_movies_from_session = session.get('liked_movies', [])
    
    return render_template("index.html", movies=movie_options, genres=genres, 
                         results=None, liked_movies=liked_movies_from_session)


@app.route("/recommend", methods=["GET", "POST"])
def recommend():
    if request.method == "POST":
        # Read form values
        top_n = int(request.form.get("top_n", 10))
        genre = request.form.get("genre", "All")
        rating_limit = float(request.form.get("rating_limit", 0.0))
        algorithm = request.form.get("algorithm", "jaccard")
        # Checkbox returns "yes" if checked, None if unchecked
        prioritize_rating = request.form.get("prioritize_rating") == "yes"
        
        # Get liked movies from comma-separated string
        liked_movies_str = request.form.get("liked_movies", "")
        liked_movie_ids = []
        if liked_movies_str:
            liked_movie_ids = [int(x.strip()) for x in liked_movies_str.split(",") if x.strip()]

        # Store in session for graph visualization and back navigation
        session['liked_movies'] = liked_movie_ids
        session['top_n'] = top_n
        session['genre'] = genre
        session['rating_limit'] = rating_limit
        session['algorithm'] = algorithm
        session['prioritize_rating'] = prioritize_rating

        results = []
        if liked_movie_ids:
            results = get_recommendations_for_liked_movies(
                liked_movie_ids, 
                top_n=top_n, 
                genre_filter=genre,
                rating_limit=rating_limit,
                algorithm=algorithm,
                prioritize_rating=prioritize_rating
            )
        
        # Store results in session for graph page
        session['results'] = results

        # Render recommendations page
        return render_template("recommendations.html", results=results)
    
    else:  # GET request - retrieve from session
        results = session.get('results', [])
        if not results:
            # No session data, redirect to home
            return render_template("recommendations.html", results=[], error="No recommendations found. Please select movies first.")
        
        return render_template("recommendations.html", results=results)


@app.route("/graph", methods=["GET"])
def graph():
    # Get data from session
    liked_movie_ids = session.get('liked_movies', [])
    results = session.get('results', [])
    
    if not liked_movie_ids or not results:
        return render_template("graph.html", error="No recommendation data found. Please generate recommendations first.")
    
    # Generate interactive graph
    fig, similar_users_details = graphvis.create_bipartite_graph(
        G, MAPPINGS, MOVIES_DF, liked_movie_ids, results, top_n_similar=5
    )
    
    # Convert to HTML
    graph_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
    
    return render_template("graph.html", graph_html=graph_html, similar_users=similar_users_details)


if __name__ == "__main__":
    # Debug mode should be False in production
    app.run(debug=os.environ.get('FLASK_DEBUG', 'False') == 'True', 
            host='0.0.0.0',
            port=int(os.environ.get('PORT', 5000)))
