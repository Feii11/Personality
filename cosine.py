import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from pyvis.network import Network

# --- Contoh Data OCEAN untuk 10 orang (range 1–7) ---
np.random.seed(42)
people = ["Alice","Bob","Carol","Dave","Eve","Frank","Grace","Hank","Ivy","Jack"]
traits = ["Openness","Conscientiousness","Extraversion","Agreeableness","Neuroticism"]

data = pd.DataFrame(np.random.randint(1,8,(10,5)), columns=traits, index=people)

# --- Hitung Cosine Similarity ---
cos_sim = pd.DataFrame(cosine_similarity(data), index=people, columns=people)

# --- Bentuk pasangan (A,B,similarity) tanpa duplikat diri sendiri ---
edges = []
for i in range(len(people)):
    for j in range(i+1, len(people)):
        edges.append((people[i], people[j], cos_sim.iloc[i,j]))
edges_df = pd.DataFrame(edges, columns=["Person A","Person B","Cosine Similarity"])

# --- Pastikan semua orang punya koneksi: ambil top 1 per orang + top 10 global ---
top_per_person = edges_df.loc[edges_df.groupby("Person A")["Cosine Similarity"].idxmax()]
top_global = edges_df.nlargest(10, "Cosine Similarity")
df_edges = pd.concat([top_per_person, top_global]).drop_duplicates().reset_index(drop=True)

# --- Buat NetworkX graph ---
G = nx.Graph()
for i, row in df_edges.iterrows():
    G.add_edge(row["Person A"], row["Person B"], weight=row["Cosine Similarity"])

# --- Konversi ke Pyvis ---
net = Network(height="650px", width="100%", bgcolor="#ffffff", font_color="black")
net.from_nx(G)

# --- Tambahkan informasi edge ---
for edge in net.edges:
    src = edge["from"]
    dst = edge["to"]

    row = df_edges[
        ((df_edges["Person A"] == src) & (df_edges["Person B"] == dst)) |
        ((df_edges["Person A"] == dst) & (df_edges["Person B"] == src))
    ]

    if not row.empty:
        w = float(row["Cosine Similarity"].values[0])
        edge["value"] = w * 10
        edge["title"] = f"Cosine Similarity: {w:.3f}"

# --- Styling node ---
for node in net.nodes:
    node["size"] = 20 + G.degree(node["id"]) * 4
    node["title"] = f"{node['id']} (Connections: {G.degree(node['id'])})"
    node["color"] = "#90CAF9" if G.degree(node["id"]) > 1 else "#F48FB1"

# --- Simpan hasil ke HTML ---
net.save_graph("network_all_connected.html")

# --- Tampilkan di Streamlit ---
st.title("🔗 Network Plot Cosine Similarity – Semua Orang Terhubung")
with open("network_all_connected.html", "r", encoding="utf-8") as f:
    html = f.read()
st.components.v1.html(html, height=700, scrolling=True)
