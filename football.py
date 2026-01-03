from sklearn.decomposition import NMF
from sklearn.preprocessing import MinMaxScaler, normalize
import pandas as pd
import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Caching
st.set_page_config(page_title="Football Recommender", layout="centered")

@st.cache_data
def load_and_process_data():
    df = pd.read_excel("data/top5-players24-25.xlsx") 
    
    df = df.dropna(subset=["Pos"])
    df = df[df["Pos"].str.contains("FW")]
    
    features = ['Gls', 'Ast', 'G+A', 'G-PK', 'PK', 'PKatt', 'xG',
                'npxG', 'xAG', 'npxG+xAG', 'PrgC', 'PrgP', 'PrgR']

    X = df[features].fillna(0)
    
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    nmf = NMF(n_components=10, max_iter=500, random_state=42)
    nmf_features = nmf.fit_transform(X_scaled)
    normalized_features = normalize(nmf_features)

    return df, normalized_features
        
# Load the cached data
try:
    df, normalized_features = load_and_process_data()
    players = df["Player"].reset_index(drop=True)
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Helper function
def get_similar_players(target_player, normalized_features, players, top_n=10):
    indices = np.where(players == target_player)[0]
    
    if len(indices) == 0:
        return []
    
    idx = indices[0]
    
    target_player_features_vector = normalized_features[idx]
    similarities = normalized_features.dot(target_player_features_vector)

    similar_players_indices = similarities.argsort()[::-1][1:top_n+1] 
    
    return [(players.iloc[i], similarities[i]) for i in similar_players_indices]

# UI LAYOUT
st.title("⚽ Attacking Productivity Recommender")
st.subheader("Europe Top 5 Leagues (24/25)")

selected_player = st.selectbox("Select a player:", players, index=0)

if selected_player:
    recommendations = get_similar_players(selected_player, normalized_features, players, top_n=5)
    
    st.markdown(f"### Top 5 Attackers similar to **{selected_player}**:")
    
    for name, score in recommendations:
        st.write(f"- **{name}** (Similarity: {score:.3f})")
    
    recommendations_df = pd.DataFrame(
        [{"name": name, "score": score} for name, score in recommendations]
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=recommendations_df, x="score", y="name", palette="rocket", orient="h", ax=ax)
    ax.set_title(f"Similarity Scores for {selected_player}")
    ax.set_xlabel("Similarity Score (0-1)")
    ax.set_ylabel("")
    st.pyplot(fig)

st.markdown("---")
st.markdown("""
**About this model** This model employs **Non-negative Matrix Factorization (NMF)** to recommend attackers with similar productivity profiles. 
It analyzes metrics like Goals, Assists, xG, and Progression stats to find players who output similar numbers, **regardless of their specific playstyle or position**.
""")
