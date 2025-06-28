# Unsupervised Machine Learning: Footballer Attacking Productivity Clustering & Similarity Recommendation System

![Python](https://img.shields.io/badge/Python-3.11-blue)
![pandas](https://img.shields.io/badge/pandas-1.5.3-blueviolet)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2.2-green)
![streamlit](https://img.shields.io/badge/streamlit-1.24.1-orange)
![matplotlib](https://img.shields.io/badge/matplotlib-3.7.1-red)
![seaborn](https://img.shields.io/badge/seaborn-0.12.2-blue)
![License](https://img.shields.io/badge/license-MIT-brightgreen)

---

## 📖 Project Overview

This project applies unsupervised learning techniques to analyze attacking footballers' productivity metrics from the 2024/25 season across Europe’s Big 5 leagues:

* **Clustering with K-Means**: Players are grouped by similarity in goal contributions, expected goals/assists, and progressive actions to identify meaningful player archetypes.
* **Dimensionality Reduction and Similarity Search with NMF**: Extracts latent feature representations from non-negative attacking stats to recommend players with similar attacking productivity profiles.

The deliverables include:

* An **exploratory data analysis (EDA) Jupyter notebook** covering distributional analysis, team-level attacking efficiency, clustering, and player profiling.
* A **Streamlit web app** that provides an interactive interface to recommend and visualize similar forwards using NMF-based embeddings.

---

## 🗃️ Dataset & Feature Engineering

* Source: Detailed player performance data extracted from Europe’s top five leagues (Premier League, La Liga, Serie A, Bundesliga, Ligue 1) for the 2024/25 season.

* Focus on **forwards only** (`Pos` includes `"FW"`).

* Selected features reflect attacking productivity and contributions, including:

  | Metric     | Description                          |
  | ---------- | ------------------------------------ |
  | `Gls`      | Goals scored                         |
  | `Ast`      | Assists made                         |
  | `G+A`      | Goals plus assists                   |
  | `G-PK`     | Goals excluding penalties            |
  | `PK`       | Penalty goals                        |
  | `PKatt`    | Penalty attempts                     |
  | `xG`       | Expected goals                       |
  | `npxG`     | Non-penalty expected goals           |
  | `xAG`      | Expected assists                     |
  | `npxG+xAG` | Composite expected goal contribution |
  | `PrgC`     | Progressive carries                  |
  | `PrgP`     | Progressive passes                   |
  | `PrgR`     | Progressive receptions               |

* Preprocessing:

  * Applied `MinMaxScaler` to normalize feature ranges \[0,1].
  * For clustering, standardized with `StandardScaler` where appropriate.
  * Dimensionality reduction using `NMF` to leverage non-negativity and parts-based representation.

---

## 🔧 Notebook: Exploratory Data Analysis & Clustering

* **EDA steps:**

  * Univariate histograms reveal strong left skewness typical for attacking metrics, with many low-scoring players and a few high performers.
  * Aggregated team-level attacking outputs via sums of goals, assists, xG, and xAG.
  * Constructed an **Efficiency Score** composite metric, weighted to reflect contributions beyond raw stats.
  * Visualized relationships between actual and expected goals/assists identifying over- and under-performers.

* **K-Means clustering:**

  * Performed on standardized attacking metrics to segment forwards into 4 clusters.
  * Silhouette scores and inertia evaluated cluster separation and compactness.
  * Cluster centroids visualized via radar/spider plots to interpret trait differences.
  * Identified clusters roughly correspond to:

    * Elite “lethal strikers” with dominant finishing and expected goals.
    * Creative, well-rounded attackers combining assists and progressive play.
    * Average contributors with moderate stats.
    * Low-productivity or peripheral forwards.
  * Outlier analysis isolates standout players within clusters (e.g., Mbappé as an outlier in lethal striker cluster).

---

## ⚙️ Streamlit App: NMF-based Similarity Recommender

* **Workflow:**

  1. Scale selected features with `MinMaxScaler`.
  2. Fit NMF (`max_iter=500`) to factorize the player-feature matrix into latent components.
  3. Normalize the resulting latent vectors using cosine normalization.
  4. Compute cosine similarity between the target player and all others in latent space.
  5. Return top-N similar players ranked by similarity score.

---

## 🎯 Applications

* **Player scouting:** Find hidden gems or comparable alternatives by productivity profile, useful when positional data is unavailable or ambiguous.
* **Transfer market:** Data-driven similarity can inform recruitment strategy and mitigate risk.
* **Player development:** Track progression in latent productivity traits over time.
* **Football analytics:** Extends typical metrics to non-negative latent factors that capture attacking style nuances.

---

## 🚀 How to Run

1. Clone the repo.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Open and explore the Jupyter Notebook for detailed EDA and clustering:

   ```bash
   jupyter notebook football_analysis.ipynb
   ```
4. Launch the Streamlit app for real-time recommendations:

   ```bash
   streamlit run football.py
   ```

---

## 📚 Libraries & Tools

* [pandas](https://pandas.pydata.org/) for data manipulation
* [scikit-learn](https://scikit-learn.org/) for scaling, clustering, NMF, and similarity metrics
* [matplotlib](https://matplotlib.org/) and [seaborn](https://seaborn.pydata.org/) for visualization
* [streamlit](https://streamlit.io/) for interactive web app

---

## ⚖️ License

This project is licensed under the **MIT License**.
