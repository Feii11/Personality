import plotly.graph_objects as go
import plotly.express as px
from typing import List
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram
import numpy as np
import plotly.figure_factory as ff

TRAIT_COLS = ['agree', 'consc', 'extra', 'neuro', 'openn']
TRAIT_LABELS = {
    'agree': 'Agreeableness',
    'consc': 'Conscientiousness',
    'extra': 'Extraversion',
    'neuro': 'Neuroticism',
    'openn': 'Openness'
}

def style_common(fig: go.Figure, y_range=(1,7), height=600):
    fig.update_layout(
        xaxis_title="Dimensi Kepribadian",
        yaxis_title="Skor",
        yaxis=dict(range=list(y_range)),
        template="simple_white",
        height=height,
        title=None
    )
    return fig

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def pie_chart_styled(dataframe: pd.DataFrame, column: str = 'SIC_1digit') -> go.Figure:
    """
    Buat donut chart berdasarkan kolom deskripsi (misal SIC_1digit),
    tanpa legend manual, dengan efek pop dan tooltip interaktif.
    """
    # --- siapkan data ---
    counts = dataframe[column].value_counts().reset_index()
    counts.columns = [column, 'count']
    counts = counts.sort_values('count', ascending=False).reset_index(drop=True)
    counts['percent'] = counts['count'] / counts['count'].sum() * 100

    # --- warna pastel lembut ---
    colors = px.colors.qualitative.Pastel
    if len(colors) < len(counts):
        from itertools import cycle, islice
        colors = list(islice(cycle(colors), len(counts)))

    # --- buat donut chart ---
    fig = go.Figure(
        go.Pie(
            labels=counts[column],
            values=counts['count'],
            hole=0.45,
            marker=dict(colors=colors, line=dict(color='white', width=2)),
            textinfo='label+percent',
            textposition='outside',
            pull=[0.05]*len(counts),  # efek pop keluar
            hovertemplate=(
                "<b>%{label}</b><br>"
                "Jumlah: %{value}<br>"
                "Persentase: %{percent}"
            ),
            insidetextorientation='radial'
        )
    )

    # --- layout tampilan ---
    fig.update_layout(
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.05,
            font=dict(size=12)
        ),
        margin=dict(l=40, r=80, t=40, b=40),
        height=420,
        paper_bgcolor="white",
        hoverlabel=dict(bgcolor="white", font_size=13, font_family="Arial"),
    )

    return fig

def pie_chart(dataframe: pd.DataFrame) -> go.Figure:
    """
    Buat pie chart dari dataframe berdasarkan kolom tertentu.
    """
    column = 'SIC_1digit'
    counts = dataframe[column].value_counts().reset_index()
    counts.columns = [column, 'count']

    fig = px.pie(
        counts,
        names=column,
        values='count',
        hole=0.4
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(showlegend=True)
    return fig

def mean_line_chart(rata_df, id_col: str, desc_col: str, selected_ids: List[str] = None) -> go.Figure:
    """
    Buat line chart dari hasil rata-rata.
    - rata_df: dataframe dengan kolom id_col + TRAIT_COLS + desc_col (optional)
    - id_col: 'SIC_1digit' atau 'SIC_2digit'
    - desc_col: kolom yang berisi deskripsi untuk hover
    """
    fig = go.Figure()
    df = rata_df.copy()
    if selected_ids is not None:
        df = df[df[id_col].isin(selected_ids)]

    x = [TRAIT_LABELS.get(t, t) for t in TRAIT_COLS]

    for _id in df[id_col].unique():
        row = df.loc[df[id_col] == _id].iloc[0]
        y = [row[t] for t in TRAIT_COLS]
        desc = row[desc_col] if desc_col in df.columns else str(_id)
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode='lines+markers',
            name=f"{_id}",
            hovertemplate=f"{_id}. {desc}<br>Dimensi: %{{x}}<br>Skor: %{{y:.2f}}<extra></extra>"
        ))

    fig = style_common(fig)
    fig.update_layout(title=" ")  # tidak ada judul
    return fig


def boxplot_from_means(df_long, height=600):
    fig = px.box(
        df_long,
        x="Dimensi_Kepribadian",
        y="Rata_Rata",
        color="Dimensi_Kepribadian",
        hover_data=["SIC_1digit" if "SIC_1digit" in df_long.columns else "SIC_2digit",
                    "Description_1" if "Description_1" in df_long.columns else "Description_2"],
        points="all",
        height=height
    )
    fig.update_traces(marker=dict(size=6, opacity=0.6))
    fig.update_layout(
        showlegend=False,
        xaxis_title=None,
        yaxis_title="Rata-Rata",
        margin=dict(l=20, r=20, t=20, b=20)
    )
    return fig

def errorbar_plot_from_means(df_long, title_suffix="", height=500):
    """
    Membuat grafik batang (bar chart) dengan error bar (Mean ± Std)
    untuk menampilkan variasi standar deviasi setiap dimensi kepribadian.
    """
    # Hitung mean dan std per dimensi
    std_summary = df_long.groupby('Dimensi_Kepribadian')['Rata_Rata'].agg(['mean', 'std']).reset_index()

    # Plot bar dengan error bar
    fig = px.bar(
        std_summary,
        x='Dimensi_Kepribadian',
        y='mean',
        error_y='std',
        color='Dimensi_Kepribadian',
        title=f" ",
        labels={'mean': 'Rata-rata', 'Dimensi_Kepribadian': 'Dimensi Kepribadian'},
        height=height
    )

    # Custom hover agar tampil Mean dan Std rapi
    fig.update_traces(
        hovertemplate=(
            "Dimensi: %{x}<br>"
            "Rata-rata = %{y:.2f}<br>"
            "Standar deviasi = %{error_y.array:.2f}<extra></extra>"
        )
    )

    fig.update_layout(
        showlegend=False,
        xaxis_title=None,
        yaxis_title="Nilai Rata-rata",
        title_x=0.5,
        margin=dict(l=20, r=20, t=40, b=20),
        template="simple_white"
    )

    return fig

import plotly.express as px

def plot_G_bar(df_G, title_suffix=""):
    """
    Create a bar chart for G(Trait) scores.
    
    Parameters:
    -----------
    df_G : pd.DataFrame
        DataFrame with columns 'Personality Trait' and 'G(Trait)'
    title_suffix : str
        Additional text for the title
    
    Returns:
    --------
    plotly.graph_objects.Figure
    """
    label_map = {
        'agree': 'Agreeableness',
        'consc': 'Conscientiousness',
        'extra': 'Extraversion',
        'neuro': 'Neuroticism',
        'openn': 'Openness'
    }
    
    # Map trait names to full labels
    df_plot = df_G.copy()
    df_plot['Personality Trait'] = df_plot['Personality Trait'].map(label_map).fillna(df_plot['Personality Trait'])
    
    fig = px.bar(
        df_plot,
        x='Personality Trait',
        y='G(Trait)',
        color='Personality Trait',
        title=f"G-Index Distribution by Personality Trait {title_suffix}",
        labels={'G(Trait)': 'G-Index (Gini Coefficient)'},
        template='simple_white',
        text_auto='.3f'
    )
    
    fig.update_layout(
        showlegend=False,
        xaxis_title="Personality Trait",
        yaxis_title="G-Index",
        height=500,
        margin=dict(l=40, r=40, t=60, b=40)
    )
    
    fig.update_traces(
        textposition='outside',
        hovertemplate="<b>%{x}</b><br>G-Index: %{y:.4f}<extra></extra>"
    )
    
    return fig

import plotly.express as px

def plot_cosine_heatmap(sim_df: pd.DataFrame, id_col: str, title: str = None, height: int = 700):
    """
    Visualisasikan matriks cosine similarity sebagai heatmap.
    """
    fig = px.imshow(
        sim_df,
        text_auto=True,  # Set to True to show data values in the heatmap
        color_continuous_scale="RdBu_r",
        aspect="equal",  # Changed from "auto" to "equal" to make it square
        origin="upper",
        labels=dict(x=id_col, y=id_col, color="Cosine Similarity"),
    )

    fig.update_layout(
        title=title or f"Cosine Similarity antar {id_col}",
        height=height,
        width=height,  # Added width=height to ensure square shape
        template="simple_white",
        xaxis_tickangle=-45,
        xaxis_side='top',
        margin=dict(l=60, r=20, t=60, b=100)
    )

    fig.update_traces(
        hovertemplate="%{y} vs %{x}<br>Cosine = %{z:.3f}<extra></extra>"
    )

    return fig

def plot_cosine_dendogram(sim_df: pd.DataFrame, id_col: str, title: str = None, height: int = 700):
    """
    Visualisasikan matriks cosine similarity sebagai dendrogram.
    """


    # Konversi similarity menjadi distance
    distance_matrix = 1 - sim_df.values

    # Lakukan hierarchical clustering
    linked = linkage(distance_matrix, method='ward')

    # Buat dendrogram
    fig = ff.create_dendrogram(
        sim_df.values,
        orientation='left',  # Changed to 'left' to put labels on the y-axis
        labels=sim_df.index.tolist(),
        linkagefun=lambda x: linked
    )

    # Rotate the figure 90 degrees
    fig.update_layout(
        title=title or f"Dendrogram Cosine Similarity antar {id_col}",
        height=height,
        template="simple_white",
        margin=dict(l=60, r=20, t=60, b=100),
        xaxis_tickangle=0  # Make labels horizontal
    )

    return fig