from sklearn.decomposition import NMF
from sklearn.preprocessing import MinMaxScaler, normalize
import pandas as pd
import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_excel("data/top5-players24-25.xlsx") 

# Only keep Forwards
df = df[df["Pos"].str.contains("FW")]

X = df[['Gls', 'Ast', 'G+A', 'G-PK', 'PK', 'PKatt', 'xG',
        'npxG', 'xAG', 'npxG+xAG', 'PrgC', 'PrgP', 'PrgR']]


scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

nmf = NMF(max_iter=500)
nmf_features = nmf.fit_transform(X_scaled)
normalized_features = normalize(nmf_features)

players = df["Player"]

def get_similar_players(target_player, normalized_features, players, top_n=10):
    try:
        idx = np.where(players == target_player)[0][0]
    except IndexError:
        return []
    
    target_player_features_vector = normalized_features[idx]
    similarities = normalized_features.dot(target_player_features_vector)
    similar_players_indices = similarities.argsort()[::-1][1:top_n+1] # skip self
    
    return [(players.iloc[i], similarities[i]) for i in similar_players_indices]

st.title("Attacking Productivity Similarity Recommender (Europe Top 5 Domestic Leagues: 24/25)")

selected_player = st.selectbox("Select a player:", players, index=979)

if selected_player:
    recommendations = get_similar_players(selected_player, normalized_features, players, top_n=5)
    st.write(f"Top 5 Attackers with similar productivity to **{selected_player}'s**:")
    for name, score in recommendations:
        st.write(f"- {name} (Similarity: {score:.3f})")
   
    recommendations_df = pd.DataFrame({
        "name": name,
        "score": score
        }
        for name, score in recommendations
    )

    fig, ax = plt.subplots()
    ax = sns.barplot(data=recommendations_df, x="score", y="name", palette="rocket", orient="h")
    ax.set_title(f"Top 5 Attackers with similar productivity to {selected_player}'s.")
    st.pyplot(fig)

st.markdown("---")
st.markdown("""
**About this model**  
This model employs **Non-negative Matrix Factorization (NMF)** to recommend attackers with similar productivity profiles from the top five European domestic leagues (Premier League, La Liga, Bundesliga, Serie A, and Ligue 1) for the 2024/25 season.  

It uses the following performance metrics:  
`['Gls', 'Ast', 'G+A', 'G-PK', 'PK', 'PKatt', 'xG', 'npxG', 'xAG', 'npxG+xAG', 'PrgC', 'PrgP', 'PrgR']`

The goal is to help teams find replicate an attacker's final product **without being being necessarily concerned about positions and/or play styles**.
""")
