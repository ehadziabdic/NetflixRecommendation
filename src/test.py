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
top = rec.recommend_for_user(G, mappings, ratings_df, movies_df, user_node, top_n=10, use_rwr=True)

print(f"Top recommendations for {user_node}:")
for i, r in enumerate(top, start=1):
	title = r.get("title") or r["movie_node"]
	mid = r.get("movie_id")
	print(
		f"{i}. {title} (id={mid}): jaccard={r['jaccard']:.4f}, rwr={r['rwr_prob']:.6f}, common2={r['common_2hop']}, avg_rating_similar={r['avg_rating_similar']}"
	)