import os
import sys
import networkx as nx
from typing import List, Dict, Any, Optional
from flask import Flask, render_template, request, session

from src import graph as graph_mod
from src import scoring as scoring_mod
from src import graphvis

# Force stdout/stderr to flush immediately for debugging
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

app = Flask(__name__)
# Use environment variable for secret key in production
app.secret_key = os.environ.get('SECRET_KEY', 'dev_key_change_in_production')

print("[DEBUG] Flask app created", flush=True)

# Load data and build graph once at startup
print("="*80, flush=True)
print("🎬 NETFLIX MOVIE RECOMMENDER - INITIALIZING", flush=True)
print("="*80, flush=True)

import psutil
process = psutil.Process(os.getpid())
print(f"[DEBUG] Initial memory: {process.memory_info().rss / 1024 / 1024:.1f} MB", flush=True)

RATINGS_PATH = "res/ratings_netflix.csv"
MOVIES_PATH = "res/movies_netflix.csv"

print("\n[1/5] 📂 Loading data...", flush=True)
RATINGS_DF, MOVIES_DF = graph_mod.load_data(RATINGS_PATH, MOVIES_PATH)
print(f"      ✓ Loaded {len(RATINGS_DF):,} ratings and {len(MOVIES_DF):,} movies", flush=True)
print(f"[DEBUG] Memory after load: {process.memory_info().rss / 1024 / 1024:.1f} MB", flush=True)

print("\n[2/5] 👥 Filtering users (min 10 ratings ≥3.5)...", flush=True)
RATINGS_FILTERED = graph_mod.downsample_users(RATINGS_DF, min_likes=10, threshold=3.5, sample_n=None)
print(f"      ✓ Filtered to {len(RATINGS_FILTERED):,} positive ratings", flush=True)
print(f"[DEBUG] Memory after filter: {process.memory_info().rss / 1024 / 1024:.1f} MB", flush=True)

print("\n[3/5] 🔗 Building bipartite graph...")
G, MAPPINGS = graph_mod.build_bipartite_graph(RATINGS_FILTERED, MOVIES_DF, threshold=3.5)
print(f"      ✓ Graph created with {G.number_of_nodes():,} nodes and {G.number_of_edges():,} edges")
graph_mod.validate_graph(G)

# Free memory: delete filtered ratings DataFrame (no longer needed)
del RATINGS_FILTERED
import gc
gc.collect()
print("      ✓ Freed memory from filtered DataFrame", flush=True)
print(f"[DEBUG] Memory after cleanup: {process.memory_info().rss / 1024 / 1024:.1f} MB", flush=True)

# Prepare lists for form selects
USER_NODES = [n for n, d in G.nodes(data=True) if d.get("bipartite") == "user"]
MOVIE_NODES = [n for n, d in G.nodes(data=True) if d.get("bipartite") == "movie"]

print("\n[4/5] ⚡ Building performance caches...", flush=True)

# Cache: movie_node -> set of user nodes who liked it
print("      - Building movie likers cache...", flush=True)
MOVIE_LIKERS_CACHE = graph_mod.precompute_movie_likers(G)
print(f"      ✓ Cached likers for {len(MOVIE_LIKERS_CACHE):,} movies", flush=True)

# Cache: movie_node -> genres set for faster filtering
print("      - Building genres cache...", flush=True)
MOVIE_GENRES_CACHE = graph_mod.precompute_movie_genres(G)
print(f"      ✓ Cached genres for {len(MOVIE_GENRES_CACHE):,} movies", flush=True)

# Cache: userId -> {movieId: rating} for fast lookups
RATINGS_LOOKUP = graph_mod.precompute_ratings_lookup(RATINGS_DF)
print(f"      ✓ Cached ratings for {len(RATINGS_LOOKUP):,} users", flush=True)

print(f"[DEBUG] Memory after caches: {process.memory_info().rss / 1024 / 1024:.1f} MB", flush=True)

print("\n[5/5] 🎯 Preparing movie and genre lists...")
print(f"      ✓ Ready to serve recommendations!\n")
print("="*80)
print("✅ SERVER INITIALIZATION COMPLETE")
print("="*80)
print()

# Helper to get likers of a movie node (uses cache)
def get_likers_of_movie_node(m_node: str) -> set:
    return MOVIE_LIKERS_CACHE.get(m_node, set())


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


# Recommend movies for a given user node or liked movies (OPTIMIZED)
def get_recommendations_for_liked_movies(
    liked_movie_ids: List[int], 
    top_n: int = 10, 
    genre_filter: Optional[str] = None,
    rating_limit: float = 0.0,
    algorithm: str = "jaccard",
    prioritize_rating: bool = False
) -> List[Dict[str, Any]]:
    # Build union of likers for selected movies (using cache)
    selected_movie_nodes = [f"m_{mid}" for mid in liked_movie_ids if f"m_{mid}" in MOVIE_LIKERS_CACHE]
    
    # Fast union using set operations
    user_set = set()
    for m_node in selected_movie_nodes:
        user_set.update(MOVIE_LIKERS_CACHE[m_node])

    # Early exit if no users found
    if not user_set:
        return []

    # Build candidate list with genre pre-filtering
    selected_set = set(selected_movie_nodes)
    if genre_filter and genre_filter != "All":
        candidates = [m for m in MOVIE_NODES 
                     if m not in selected_set and genre_filter in MOVIE_GENRES_CACHE.get(m, set())]
    else:
        candidates = [m for m in MOVIE_NODES if m not in selected_set]
    
    # Pre-convert user nodes to IDs for rating lookups
    user_ids = {MAPPINGS["node_to_user_id"][u] for u in user_set if u in MAPPINGS["node_to_user_id"]}
    
    results = []
    
    for m in candidates:
        # Get likers from cache
        likers = MOVIE_LIKERS_CACHE.get(m, set())
        
        # Compute intersection once
        intersection = user_set & likers
        
        # Skip if no intersection
        if not intersection:
            if algorithm == "jaccard":
                score = 0.0
            else:
                score = 0
            # Skip low scores early
            if score == 0:
                continue
        else:
            # Compute score based on algorithm
            if algorithm == "cn":
                score = len(intersection)
            else:
                # Jaccard
                union_size = len(user_set) + len(likers) - len(intersection)
                score = len(intersection) / union_size if union_size > 0 else 0.0
        
        # Fast rating calculation using lookup cache
        avg_rating = None
        movie_id = MAPPINGS["node_to_movie_id"].get(m)
        
        if movie_id and intersection:
            supporter_ids = [MAPPINGS["node_to_user_id"][u] for u in intersection if u in MAPPINGS["node_to_user_id"]]
            
            # Use memory-efficient ratings cache
            ratings_list = []
            for uid in supporter_ids:
                if uid in RATINGS_LOOKUP and movie_id in RATINGS_LOOKUP[uid]:
                    ratings_list.append(RATINGS_LOOKUP[uid][movie_id])
            
            if ratings_list:
                avg_rating = sum(ratings_list) / len(ratings_list)
        
        # Apply rating limit filter
        if rating_limit > 0.0 and (avg_rating is None or avg_rating < rating_limit):
            continue
        
        # Apply rating weight if prioritize_rating is True
        final_score = score
        if prioritize_rating and avg_rating is not None:
            rating_weight = avg_rating / 5.0
            final_score = score * rating_weight
        
        results.append({
            "movie_node": m,
            "movie_id": movie_id,
            "title": G.nodes[m].get("title"),
            "genres": G.nodes[m].get("genres"),
            "jaccard": score if algorithm == "jaccard" else None,
            "common_2hop": len(intersection),
            "avg_rating": avg_rating,
            "score": final_score
        })
    
    # Sort by final score descending and return top N
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
