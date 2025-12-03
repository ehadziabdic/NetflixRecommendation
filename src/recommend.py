import scoring as score
import networkx as nx
from typing import List, Dict, Any

# Recommend movies for a given user node
def recommend_for_user(
	G,
	mappings: Dict[str, Dict],
	ratings_df,
	movies_df,
	user_node: str,
	top_n: int = 10,
	use_rwr: bool = True,
	rwr_kwargs: Dict = None,
) -> List[Dict[str, Any]]:

	# Default parameters for RWR
	if rwr_kwargs is None:
		rwr_kwargs = {"alpha": 0.15, "max_iter": 200, "tol": 1e-6}

	# Build set of movies already seen by the user (neighbors in the bipartite graph)
	seen = set()
	for nbr in G.neighbors(user_node):
		if G.nodes[nbr].get("bipartite") == "movie":
			seen.add(nbr)

	# Build a list of all movie nodes
	all_movies = []
	for n, d in G.nodes(data=True):
		if d.get("bipartite") == "movie":
			all_movies.append(n)

	# Candidates are movies the user has not seen
	candidates = []
	for m in all_movies:
		if m not in seen:
			candidates.append(m)

	# Optionally precompute RWR probabilities for this user
	rwr_probs = None
	if use_rwr:
		pvec, nodes_order = score.rwr_scores_for_user(user_node, G, **rwr_kwargs)
		rwr_probs = {}
		for i, node in enumerate(nodes_order):
			rwr_probs[node] = float(pvec[i])

	# Users at distance 2 from the user (user -> movie -> user)
	lengths_u = nx.single_source_shortest_path_length(G, source=user_node, cutoff=2)
	gamma2_u = set()
	for n, dist in lengths_u.items():
		if dist == 2 and G.nodes[n].get("bipartite") == "user":
			gamma2_u.add(n)

	# Prepare movies lookup by numeric id
	movies_index = None
	if movies_df is not None:
		movies_index = movies_df.set_index("movieId")

	results = []

	# Evaluate each candidate movie one-by-one and collect stats
	for m in candidates:
		# Jaccard score (user-based)
		jacc = score.jaccard_2hop_score(user_node, m, G)

		# Users who liked the movie (distance 1 from movie node)
		lengths_m1 = nx.single_source_shortest_path_length(G, source=m, cutoff=1)
		gamma1_m = set()
		for n, dist in lengths_m1.items():
			if dist == 1 and G.nodes[n].get("bipartite") == "user":
				gamma1_m.add(n)

		# Intersection = users similar to `user_node` who also liked the movie
		intersection = set()
		for u in gamma2_u:
			if u in gamma1_m:
				intersection.add(u)
		inter_size = len(intersection)

		# Similar user ids (numeric) for rating lookups
		similar_user_ids = []
		for u_node in intersection:
			uid = mappings["node_to_user_id"].get(u_node)
			if uid is not None:
				similar_user_ids.append(uid)

		# Map movie node to numeric movie id
		movie_id = mappings["node_to_movie_id"].get(m)

		# Compute average rating among similar users for this movie (using full ratings_df)
		avg_rating = None
		if movie_id is not None and similar_user_ids:
			rows = ratings_df[(ratings_df["movieId"] == movie_id) & (ratings_df["userId"].isin(similar_user_ids))]
			if not rows.empty:
				avg_rating = float(rows["rating"].mean())

		# Lookup movie title if available
		title = None
		if movie_id is not None and movies_index is not None and movie_id in movies_index.index:
			title = movies_index.loc[movie_id].get("title")

		# RWR probability for this movie (if computed)
		rwr_p = None
		if rwr_probs is not None and m in rwr_probs:
			rwr_p = rwr_probs[m]

		# Append result dict
		result = {
			"movie_node": m,
			"movie_id": movie_id,
			"title": title,
			"jaccard": jacc,
			"rwr_prob": rwr_p,
			"common_2hop": inter_size,
			"avg_rating_similar": avg_rating,
		}
		results.append(result)

	# Sort results primarily by jaccard descending, secondarily by rwr_prob descending
	def sort_key(item: Dict[str, Any]):
		# Primary: jaccard (higher first)
		primary = item.get("jaccard") or 0.0
		# Secondary: rwr probability (higher first); treat None as 0
		secondary = item.get("rwr_prob") or 0.0
		# We return a tuple that sorted() will use; negate values to sort descending
		return (-primary, -secondary)

	results_sorted = sorted(results, key=sort_key)

	# Return top-N
	return results_sorted[:top_n]