# Graph Visualization Module for Web Application
import plotly.graph_objects as go
import networkx as nx
from typing import List, Dict, Any, Tuple

def create_bipartite_graph(
    G,
    mappings: Dict[str, Dict],
    movies_df,
    liked_movie_ids: List[int],
    recommended_movies: List[Dict[str, Any]],
    top_n_similar: int = 5
) -> Tuple[go.Figure, List[Dict[str, Any]]]:
    """
    Create an interactive Plotly bipartite graph visualization.
    
    Returns:
        - Plotly Figure object
        - List of similar users with details
    """
    
    # Get movie titles helper
    def get_movie_title(movie_node):
        movie_id = mappings['node_to_movie_id'].get(movie_node)
        if movie_id and movie_id in movies_df['movieId'].values:
            return movies_df[movies_df['movieId'] == movie_id]['title'].values[0]
        return movie_node
    
    # Convert liked movie IDs to nodes
    liked_movie_nodes = [f"m_{mid}" for mid in liked_movie_ids if f"m_{mid}" in G]
    
    # Find users who liked these movies (similar users)
    user_set = set()
    for m_node in liked_movie_nodes:
        if m_node in G:
            for nbr in G.neighbors(m_node):
                if G.nodes[nbr].get("bipartite") == "user":
                    user_set.add(nbr)
    
    # Calculate similarity scores for users
    user_similarity = {}
    for user in user_set:
        shared_count = 0
        for m_node in liked_movie_nodes:
            if m_node in G and G.has_edge(user, m_node):
                shared_count += 1
        if shared_count > 0:
            user_similarity[user] = shared_count
    
    # Get top N similar users
    similar_users_sorted = sorted(user_similarity.items(), key=lambda x: x[1], reverse=True)[:top_n_similar]
    similar_user_nodes = [u for u, _ in similar_users_sorted]
    
    # Get recommended movie nodes
    recommended_movie_nodes = [rec['movie_node'] for rec in recommended_movies[:10]]  # Top 10 for visualization
    
    # Create virtual "You" node
    virtual_user = "virtual_you"
    
    # Build visualization graph
    viz_graph = nx.Graph()
    
    # Add virtual user node
    viz_graph.add_node(virtual_user, bipartite='user', label='You')
    
    # Add similar users
    for u in similar_user_nodes:
        user_id = mappings['node_to_user_id'].get(u)
        viz_graph.add_node(u, bipartite='user', label=f"User {user_id}")
    
    # Add liked movies
    for m in liked_movie_nodes:
        if m in G:
            title = get_movie_title(m)
            viz_graph.add_node(m, bipartite='movie', label=title, movie_type='liked')
            viz_graph.add_edge(virtual_user, m)  # Connect virtual user to liked movies
    
    # Add recommended movies
    for m in recommended_movie_nodes:
        if m in G:
            title = get_movie_title(m)
            viz_graph.add_node(m, bipartite='movie', label=title, movie_type='recommended')
    
    # Add edges from similar users to all movies (liked and recommended)
    for u in similar_user_nodes:
        for m in liked_movie_nodes + recommended_movie_nodes:
            if m in G and G.has_edge(u, m):
                viz_graph.add_edge(u, m)
    
    # Create bipartite layout
    all_user_nodes = [virtual_user] + similar_user_nodes
    all_movie_nodes = liked_movie_nodes + recommended_movie_nodes
    
    pos = nx.bipartite_layout(viz_graph, all_user_nodes, align='vertical', scale=2)
    
    # Prepare Plotly traces
    edge_trace = []
    
    # Draw edges
    for edge in viz_graph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace.append(
            go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(width=1, color='rgba(150, 150, 150, 0.3)'),
                hoverinfo='none',
                showlegend=False
            )
        )
    
    # Prepare node traces
    # Virtual user (You)
    x_you, y_you = pos[virtual_user]
    you_trace = go.Scatter(
        x=[x_you],
        y=[y_you],
        mode='markers+text',
        marker=dict(size=30, color='#ffcc00', symbol='circle', line=dict(width=3, color='black')),
        text=['You'],
        textposition='top center',
        textfont=dict(size=14, color='white', family='Bebas Neue'),
        name='You',
        hoverinfo='text',
        hovertext='Virtual User (You)'
    )
    
    # Similar users
    x_similar = [pos[u][0] for u in similar_user_nodes]
    y_similar = [pos[u][1] for u in similar_user_nodes]
    labels_similar = [viz_graph.nodes[u]['label'] for u in similar_user_nodes]
    similar_trace = go.Scatter(
        x=x_similar,
        y=y_similar,
        mode='markers+text',
        marker=dict(size=20, color='#90ee90', symbol='circle', line=dict(width=2, color='black')),
        text=labels_similar,
        textposition='top center',
        textfont=dict(size=10, color='white', family='Arial'),
        name='Similar Users',
        hoverinfo='text',
        hovertext=labels_similar
    )
    
    # Liked movies
    x_liked = [pos[m][0] for m in liked_movie_nodes if m in pos]
    y_liked = [pos[m][1] for m in liked_movie_nodes if m in pos]
    labels_liked = [viz_graph.nodes[m]['label'][:25] + '...' if len(viz_graph.nodes[m]['label']) > 25 
                    else viz_graph.nodes[m]['label'] for m in liked_movie_nodes if m in pos]
    liked_trace = go.Scatter(
        x=x_liked,
        y=y_liked,
        mode='markers+text',
        marker=dict(size=15, color='#87ceeb', symbol='square', line=dict(width=2, color='black')),
        text=labels_liked,
        textposition='bottom center',
        textfont=dict(size=8, color='white', family='Arial'),
        name='Liked Movies',
        hoverinfo='text',
        hovertext=labels_liked
    )
    
    # Recommended movies
    x_rec = [pos[m][0] for m in recommended_movie_nodes if m in pos]
    y_rec = [pos[m][1] for m in recommended_movie_nodes if m in pos]
    labels_rec = [viz_graph.nodes[m]['label'][:25] + '...' if len(viz_graph.nodes[m]['label']) > 25 
                  else viz_graph.nodes[m]['label'] for m in recommended_movie_nodes if m in pos]
    rec_trace = go.Scatter(
        x=x_rec,
        y=y_rec,
        mode='markers+text',
        marker=dict(size=15, color='#90ee90', symbol='square', line=dict(width=2, color='black')),
        text=labels_rec,
        textposition='bottom center',
        textfont=dict(size=8, color='white', family='Arial'),
        name='Recommended Movies',
        hoverinfo='text',
        hovertext=labels_rec
    )
    
    # Create figure
    fig = go.Figure(data=edge_trace + [you_trace, similar_trace, liked_trace, rec_trace])
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f'Recommendation Graph<br><sub>{len(liked_movie_nodes)} Liked | {len(recommended_movie_nodes)} Recommended | {len(similar_user_nodes)} Similar Users</sub>',
            font=dict(size=24, color='#ff1615', family='Bebas Neue'),
            x=0.5,
            xanchor='center'
        ),
        showlegend=True,
        legend=dict(
            x=1,
            y=1,
            bgcolor='rgba(35, 35, 35, 0.8)',
            bordercolor='#ff1615',
            borderwidth=2,
            font=dict(color='white', family='Bebas Neue')
        ),
        plot_bgcolor='#181616',
        paper_bgcolor='#181616',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        hovermode='closest',
        margin=dict(b=20, l=20, r=20, t=80),
        height=700
    )
    
    # Prepare similar users details
    similar_users_details = []
    for user_node, shared_count in similar_users_sorted:
        user_id = mappings['node_to_user_id'].get(user_node)
        
        # Get shared movie titles
        shared_movies = []
        for m_node in liked_movie_nodes:
            if m_node in G and G.has_edge(user_node, m_node):
                shared_movies.append(get_movie_title(m_node))
        
        similar_users_details.append({
            'user_node': user_node,
            'user_id': user_id,
            'shared_count': shared_count,
            'shared_movies': shared_movies[:3]  # First 3 for display
        })
    
    return fig, similar_users_details
