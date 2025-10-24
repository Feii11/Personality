import streamlit as st
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go

# --- Setting halaman ---
st.set_page_config(page_title="Cosine Similarity Heatmap CEO", layout="wide")

st.markdown(
    """
    <style>
    /* Biar slider-nya full width dan terlihat elegan */
    .stSlider {
        width: 100% !important;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Judul dan deskripsi ---
st.title("🧠 Heatmap Cosine Similarity Personality CEO")
st.write("""
Visualisasi hubungan antar sektor industri berdasarkan **dimensi kepribadian CEO (Big Five)**  
dengan filter **threshold cosine similarity** untuk menyoroti korelasi yang kuat.
""")

# --- Pilihan pengaturan utama ---
col1, col2 = st.columns([1, 1])
with col1:
    dimensi = st.selectbox("📊 Pilih Tingkat SIC", ["SIC 1-digit (10 sektor)", "SIC 2-digit (63 sektor)"])
with col2:
    seed = st.number_input("🔢 Random Seed", 0, 9999, 42)

# --- Slider threshold selebar halaman ---
st.markdown("### 🎚️ Threshold Cosine Similarity (geser untuk filter hasil)")
threshold = st.slider("", 0.0, 1.0, 0.7, 0.01, key="threshold", label_visibility="collapsed")

# --- Generate data dummy ---
np.random.seed(seed)
n = 10 if "1-digit" in dimensi else 63
data = np.random.rand(n, 5)  # misal 5 dimensi Big Five
cos_sim = cosine_similarity(data)

# --- Terapkan threshold ---
filtered_cos_sim = np.where(cos_sim >= threshold, cos_sim, np.nan)

# --- Label untuk sumbu ---
labels = [f"SIC{i+1}" for i in range(n)]

# --- Buat heatmap interaktif ---
fig = go.Figure(
    data=go.Heatmap(
        z=filtered_cos_sim,
        x=labels,
        y=labels,
        colorscale="YlOrRd",          # 🔴🟠🟡
        text=np.round(cos_sim, 2),
        texttemplate="%{text}",
        hovertemplate="(%{x}, %{y}) = %{z:.2f}<extra></extra>",
        colorbar=dict(title="Cosine Similarity"),
        zmin=0,
        zmax=1,
    )
)

fig.update_layout(
    title=f"Heatmap Cosine Similarity ({dimensi}) — Threshold ≥ {threshold}",
    xaxis=dict(title="Sektor Industri", tickangle=45),
    yaxis=dict(title="Sektor Industri", autorange="reversed"),
    width=1000 if n == 10 else 1600,
    height=1000 if n == 10 else 1600,
    margin=dict(l=80, r=80, t=100, b=80),
)

# --- Tampilkan heatmap ---
st.plotly_chart(fig, use_container_width=True)

# --- Catatan bawah ---
st.markdown("---")
st.caption("💡 Gunakan slider di atas untuk menyesuaikan threshold dan temukan sektor yang paling mirip.")
