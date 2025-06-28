from sklearn.decomposition import NMF
from sklearn.preprocessing import MinMaxScaler, normalize
import pandas as pd
import streamlit as st
import numpy as np

df = pd.read_excel("top5-players24-25.xlsx") 

X = df[['Gls', 'Ast', 'G+A', 'G-PK', 'PK', 'PKatt', 'CrdY','CrdR',
        'xG', 'npxG', 'xAG', 'npxG+xAG', 'PrgC', 'PrgP', 'PrgR']]

scaler = MinMaxScaler()
X_rescaled = scaler.fit_transform(X)

nmf = NMF(n_components=5)
nmf_features = nmf.fit_transform(X_rescaled)
norm_features = normalize(nmf_features)

players = df["Player"]

def get_similar_players(target_player, norm_features, players, top_n=10):
    try:
        idx = np.where(players == target_player)[0][0]
    except IndexError:
        return []
    player_vec = norm_features[idx].reshape(1, -1)
    similarities = norm_features.dot(player_vec.T).flatten()
    similar_idx = similarities.argsort()[::-1][1:top_n+1]  # skip self
    return [(players[i], similarities[i]) for i in similar_idx]

st.title("Attacking Productivity Similarity Recommender (Europe Top 5 Domestic Leagues: 24/25)")

selected_player = st.selectbox("Select a player:", players)

if selected_player:
    recommendations = get_similar_players(selected_player, norm_features, players, top_n=10)
    st.write(f"Top 10 players similar to **{selected_player}**:")
    for name, score in recommendations:
        st.write(f"- {name} (Similarity: {score:.3f})")
