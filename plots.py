import plotly.graph_objects as go
import plotly.express as px
from typing import List
import pandas as pd

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

def plot_G_bar(df_G, group_col):
    label_map = {
        'agree': 'Agreeableness',
        'consc': 'Conscientiousness',
        'extra': 'Extraversion',
        'neuro': 'Neuroticism',
        'openn': 'Openness'
    }
    df_G['Dimensi_Kepribadian'] = df_G['Dimensi_Kepribadian'].map(label_map)
    fig = px.bar(
        df_G,
        x=group_col,
        y='G_Value',
        color='Dimensi_Kepribadian',
        barmode='group',
        title=f"G(Personality) berdasarkan {group_col.upper()}",
        labels={'G_Value': 'G (Total Absolute Difference)'},
        template='simple_white'
    )
    return fig

def plot_G_bar(df, group_col):
    fig = px.bar(
        df,
        x="Personality Trait",
        y="G(Trait)",
        color="Personality Trait",
        facet_col=group_col,
        facet_col_wrap=4,
        title=f"G(Trait) berdasarkan {group_col}",
        text_auto=".2f"
    )
    fig.update_layout(height=600, showlegend=False)
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
        aspect="auto",
        origin="upper",
        labels=dict(x=id_col, y=id_col, color="Cosine Similarity"),
    )

    fig.update_layout(
        title=title or f"Cosine Similarity antar {id_col}",
        height=height,
        template="simple_white",
        xaxis_tickangle=-45,
        xaxis_side='top',
        margin=dict(l=60, r=20, t=60, b=100)
    )

    fig.update_traces(
        hovertemplate="%{y} vs %{x}<br>Cosine = %{z:.3f}<extra></extra>"
    )

    return fig