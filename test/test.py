import os
import sys
from typing import List, Dict, Any, Optional

# Ensure we can import project modules from the `src` folder
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.join(REPO_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

import os
import graph
import recommend as rec

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ratings_path = os.path.join(repo_root, "res", "ratings.csv")
movies_path = os.path.join(repo_root, "res", "movies.csv")

ratings_df, movies_df = graph.load_data(ratings_path, movies_path)
ratings_filtered = graph.downsample_users(ratings_df, min_likes=10, threshold=3.5, sample_n=500)
G, mappings = graph.build_bipartite_graph(ratings_filtered, movies_df, threshold=3.5)
graph.validate_graph(G)

# pick a user (first user node)
user_node = next(n for n, d in G.nodes(data=True) if d.get("bipartite") == "user")
top = rec.recommend_for_user(G, mappings, ratings_df, movies_df, user_node, top_n=10)

print(f"Top recommendations for {user_node}:")
for i, r in enumerate(top, start=1):
	title = r.get("title") or r["movie_node"]
	mid = r.get("movie_id")
	print(
		f"{i}. {title} (id={mid}): jaccard={r['jaccard']:.4f}, common2={r['common_2hop']}, avg_rating={r.get('avg_rating')}"
	)