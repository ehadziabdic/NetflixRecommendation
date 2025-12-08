# =============================================================================
# GRAPH VISUALIZATION & RECOMMENDATION TEST
# =============================================================================
# This is the main test file that combines:
# - User recommendation engine
# - Similar user detection
# - Bipartite graph visualization
# All functionality previously in test.py and recommend.py is included here
# =============================================================================

# Default Imports
import matplotlib.pyplot as plt
import networkx as nx
import sys
import os

# Getting src/
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.join(REPO_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

# File Imports
import graph as graph_mod
import scoring as scoring_mod

# Load data and build graph
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ratings_path = os.path.join(repo_root, "res", "ratings.csv")
movies_path = os.path.join(repo_root, "res", "movies.csv")

# Use all users
ratings_df, movies_df = graph_mod.load_data(ratings_path, movies_path)
ratings_filtered = graph_mod.downsample_users(ratings_df, min_likes=10, threshold=3.5, sample_n=None)
G, mappings = graph_mod.build_bipartite_graph(ratings_filtered, movies_df, threshold=3.5)

print(f"Graph contains {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

# Get user and movie nodes
user_nodes = [n for n, d in G.nodes(data=True) if d.get('bipartite') == 'user']
movie_nodes = [n for n, d in G.nodes(data=True) if d.get('bipartite') == 'movie']
print(f"Users: {len(user_nodes)}, Movies: {len(movie_nodes)}")

# Get movie title
def get_movie_title(movie_node):
    movie_id = mappings['node_to_movie_id'].get(movie_node)
    if movie_id and movie_id in movies_df['movieId'].values:
        return movies_df[movies_df['movieId'] == movie_id]['title'].values[0]
    return movie_node

# Find similar users
def find_similar_users(user_node, top_n=5):
    # Get movies liked by user
    user_movies = [m for m in G.neighbors(user_node) if G.nodes[m].get('bipartite') == 'movie']
    
    # Find users at distance 2 (via shared movies)
    user_similarity = {}
    for movie in user_movies:
        for other_user in G.neighbors(movie):
            if other_user != user_node and G.nodes[other_user].get('bipartite') == 'user':
                if other_user not in user_similarity:
                    user_similarity[other_user] = 0
                user_similarity[other_user] += 1
    
    # Sort by number of shared movies
    similar = sorted(user_similarity.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return similar

# Get recommendations
def get_recommendations(user_node, top_k=10):
    # Movies already seen
    seen = {m for m in G.neighbors(user_node) if G.nodes[m].get('bipartite') == 'movie'}
    
    # Candidate movies
    candidates = [m for m in movie_nodes if m not in seen]
    
    # Calculate scores
    results = []
    for m in candidates:
        jacc = scoring_mod.jaccard_2hop_score(user_node, m, G)
        cn = scoring_mod.common_neighbors_count(user_node, m, G)
        results.append({
            'movie_node': m,
            'jaccard': jacc,
            'common_neighbors': cn,
            'title': get_movie_title(m)
        })
    
    # Sort by jaccard then common neighbors
    results.sort(key=lambda x: (-x['jaccard'], -x['common_neighbors']))
    return results[:top_k]

# Display user info
def display_user_info(user_node='u_1', top_k=10):
    if user_node not in G:
        print(f"Error: {user_node} not found in graph!")
        print(f"Available users: {user_nodes}")
        return
    
    user_id = mappings['node_to_user_id'].get(user_node)
    print("\n" + "="*80)
    print(f"USER ANALYSIS: {user_node} (ID: {user_id})")
    print("="*80)
    
    # Show neighbors (movies user liked)
    neighbors = [m for m in G.neighbors(user_node) if G.nodes[m].get('bipartite') == 'movie']
    print(f"\n📽️  MOVIES LIKED BY USER ({len(neighbors)} total):")
    print("-" * 80)
    for i, movie in enumerate(neighbors[:15], 1):  # Show first 15
        title = get_movie_title(movie)
        print(f"  {i:2d}. {title}")
    if len(neighbors) > 15:
        print(f"  ... and {len(neighbors) - 15} more")
    
    # Show similar users
    similar = find_similar_users(user_node, top_n=5)
    print(f"\n👥 SIMILAR USERS (sharing common movie interests):")
    print("-" * 80)
    for i, (other_user, shared_count) in enumerate(similar, 1):
        other_id = mappings['node_to_user_id'].get(other_user)
        print(f"  {i}. {other_user} (ID: {other_id}) - {shared_count} shared movies")
        # Show some shared movies
        user_movies = set(m for m in G.neighbors(user_node) if G.nodes[m].get('bipartite') == 'movie')
        other_movies = set(m for m in G.neighbors(other_user) if G.nodes[m].get('bipartite') == 'movie')
        shared = user_movies & other_movies
        shared_titles = [get_movie_title(m) for m in list(shared)[:3]]
        print(f"     Shared: {', '.join(shared_titles)}")
    
    # Show recommendations
    recommendations = get_recommendations(user_node, top_k=top_k)
    print(f"\n⭐ TOP-{top_k} RECOMMENDED MOVIES:")
    print("-" * 80)
    print(f"{'Rank':<6} {'Title':<50} {'Jaccard':<10} {'Common Users'}")
    print("-" * 80)
    for i, rec in enumerate(recommendations, 1):
        title = rec['title'][:47] + '...' if len(rec['title']) > 50 else rec['title']
        print(f"{i:<6} {title:<50} {rec['jaccard']:.4f}    {rec['common_neighbors']}")
    
    # Visualization
    print("\n📊 Generating visualization...")
    visualize_user_neighborhood(user_node, recommendations)

# Visualize user neighborhood and recommendations
def visualize_user_neighborhood(user_node, top_recommendations):
    # Get watched movies
    watched_movies = [m for m in G.neighbors(user_node) if G.nodes[m].get('bipartite') == 'movie']
    
    # Get recommended movie nodes
    recommended_movie_nodes = [rec['movie_node'] for rec in top_recommendations]
    
    # Get similar users
    similar_users_list = find_similar_users(user_node, top_n=5)
    similar_user_nodes = [u for u, _ in similar_users_list]
    
    # Collect all nodes for the bipartite graph
    all_user_nodes = [user_node] + similar_user_nodes
    all_movie_nodes = watched_movies + recommended_movie_nodes
    
    # Create subgraph
    subgraph_nodes = all_user_nodes + all_movie_nodes
    subgraph = G.subgraph([n for n in subgraph_nodes if n in G])
    
    # Add edges for recommended movies (connect to similar users who liked them)
    edges_to_add = []
    for rec_movie in recommended_movie_nodes:
        if rec_movie in G:
            for similar_user in similar_user_nodes:
                if similar_user in G and G.has_edge(similar_user, rec_movie):
                    edges_to_add.append((similar_user, rec_movie))
    
    # Create a new graph with all nodes and edges
    viz_graph = nx.Graph()
    viz_graph.add_nodes_from(subgraph.nodes(data=True))
    viz_graph.add_edges_from(subgraph.edges())
    viz_graph.add_edges_from(edges_to_add)
    
    # Create bipartite layout
    pos = nx.bipartite_layout(viz_graph, all_user_nodes, align='vertical', scale=2)
    
    # Create figure
    plt.figure(figsize=(20, 12))
    
    # Draw selected user (yellow circle)
    nx.draw_networkx_nodes(viz_graph, pos, nodelist=[user_node], node_color='lightgreen', node_size=600, 
                          label='Selected User', node_shape='o', edgecolors='black', linewidths=2)
    
    # Draw similar users (green circles)
    nx.draw_networkx_nodes(viz_graph, pos, nodelist=similar_user_nodes, node_color='lightblue', node_size=600, 
                          label='Similar Users', node_shape='o', edgecolors='black', linewidths=2)
    
    # Draw watched movies (blue squares)
    nx.draw_networkx_nodes(viz_graph, pos, nodelist=watched_movies, node_color='gray', node_size=500, 
                          label='Already Watched', node_shape='s', edgecolors='black', linewidths=2)
    
    # Draw recommended movies (green squares)
    nx.draw_networkx_nodes(viz_graph, pos, nodelist=recommended_movie_nodes, node_color='red', node_size=500, 
                          label='Recommended', node_shape='s', edgecolors='black', linewidths=2)
    
    # Draw edges
    nx.draw_networkx_edges(viz_graph, pos, alpha=0.3, width=1.5)
    
    # Add labels
    labels = {user_node: user_node}
    
    # Label similar users
    for u in similar_user_nodes:
        labels[u] = u
    
    # Label movies
    for m in watched_movies + recommended_movie_nodes:
        title = get_movie_title(m)
        labels[m] = title[:20] + '...' if len(title) > 20 else title
    
    nx.draw_networkx_labels(viz_graph, pos, labels, font_size=8, font_weight='bold')
    
    user_id = mappings['node_to_user_id'].get(user_node)
    plt.title(f'Bipartite Recommendation Graph for {user_node} (ID: {user_id})\n' + 
              f'{len(watched_movies)} Watched | {len(recommended_movie_nodes)} Recommended | {len(similar_user_nodes)} Similar Users',
              fontsize=16, fontweight='bold', pad=20)
    plt.legend(loc='upper right', fontsize=12, frameon=True, shadow=True)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


# Main execution
if __name__ == "__main__":
    display_user_info('u_1', top_k=10)
