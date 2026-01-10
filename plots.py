import plotly.graph_objects as go
import plotly.express as px
from typing import List
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram
import numpy as np
import plotly.figure_factory as ff
import math
from typing import Optional
from scipy.spatial.distance import squareform

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

# Radar Chart - MEAN
def plot_radar_chart(rata_df: pd.DataFrame, id_col: str, selected_id: str, desc_col: str = None) -> go.Figure:
    # Filter buat SIC terpilih
    row = rata_df[rata_df[id_col].astype(str) == str(selected_id)]
    
    if row.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        return fig
    
    row = row.iloc[0]
    
    values = [row[trait] for trait in TRAIT_COLS]
    labels = [TRAIT_LABELS.get(trait, trait) for trait in TRAIT_COLS]
    
    values_closed = values + [values[0]]
    labels_closed = labels + [labels[0]]
    
    desc = row[desc_col] if desc_col and desc_col in rata_df.columns else ""
    title_text = f"{id_col}: {selected_id}"
    if desc:
        title_text += f" - {desc}"
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=labels_closed,
        fill='toself',
        name=selected_id,
        line=dict(color='rgb(102, 126, 234)', width=2),
        fillcolor='rgba(102, 126, 234, 0.3)',
        hovertemplate="<b>%{theta}</b><br>Score: %{r:.2f}<extra></extra>"
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[1, 7],
                tickmode='linear',
                tick0=1,
                dtick=1,
                showline=True,
                gridcolor='lightgray'
            ),
            angularaxis=dict(
                gridcolor='lightgray'
            )
        ),
        showlegend=False,
        title=dict(
            text=title_text,
            x=0.5,
            xanchor='center',
            font=dict(size=14)
        ),
        height=500,
        template="simple_white",
        margin=dict(l=60, r=60, t=80, b=60)
    )
    
    return fig

# Pie Chart - Industry Distribution
def pie_chart_styled(dataframe: pd.DataFrame, column: str = 'SIC_1digit') -> go.Figure:
    # --- siapkan data ---
    counts = dataframe[column].value_counts().reset_index()
    counts.columns = [column, 'count']
    counts = counts.sort_values('count', ascending=False).reset_index(drop=True)
    counts['percent'] = counts['count'] / counts['count'].sum() * 100

    # # --- warna pastel lembut ---
    # colors = px.colors.qualitative.Pastel
    # if len(colors) < len(counts):
    #     from itertools import cycle, islice
    #     colors = list(islice(cycle(colors), len(counts)))

    # --- buat donut chart ---
    fig = go.Figure(
        go.Pie(
            labels=counts[column],
            values=counts['count'],
            hole=0.45,
            marker=dict( line=dict(color='white', width=2)),
            textinfo='label+percent',
            textposition='outside',
            pull=[0.05]*len(counts),
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

# Line Chart - MEAN
def mean_line_chart(rata_df, id_col: str, desc_col: str, selected_ids: List[str] = None) -> go.Figure:
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

# Errorbar plot - STANDAR DEVIASI
def errorbar_plot_from_means(df_long, title_suffix="", height=500):
    std_summary = df_long.groupby('Dimensi_Kepribadian')['Rata_Rata'].agg(['mean', 'std']).reset_index()

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

# Bar Chart - G-Index Trait SIC 1 Digit
def plot_G_bar(df_G, title_suffix=""):
    label_map = {
        'agree': 'Agreeableness',
        'consc': 'Conscientiousness',
        'extra': 'Extraversion',
        'neuro': 'Neuroticism',
        'openn': 'Openness'
    }
    
    df_plot = df_G.copy()
    df_plot['Personality Trait'] = df_plot['Personality Trait'].map(label_map).fillna(df_plot['Personality Trait'])
    
    fig = px.bar(
        df_plot,
        x='Personality Trait',
        y='G(Trait)',
        color='Personality Trait',
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

# Heatmap - COSINE SIMILARITY & PEARSON CORRELATION
def plot_cosine_heatmap(sim_df: pd.DataFrame, id_col: str, title: str = None, height: int = 700):
    fig = px.imshow(
        sim_df,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        aspect="equal", # Biar heatmapnya kotak
        origin="upper",
        labels=dict(x=id_col, y=id_col, color="Cosine Similarity"),
    )

    fig.update_layout(
        height=height,
        width=height,
        template="simple_white",
        xaxis_tickangle=-45,
        xaxis_side='top',
        margin=dict(l=60, r=20, t=60, b=100)
    )

    fig.update_traces(
        hovertemplate="%{y} vs %{x}<br>Cosine = %{z:.3f}<extra></extra>"
    )

    return fig

# Dendrogram - COSINE SIMILARITY
def plot_cosine_dendogram(sim_df: pd.DataFrame, id_col: str, title: str = None, height: int = 700):
    distance_matrix = 1 - sim_df.values

    linked = linkage(distance_matrix, method='ward')

    fig = ff.create_dendrogram(
        sim_df.values,
        orientation='left',
        labels=sim_df.index.tolist(),
        linkagefun=lambda x: linked
    )

    fig.update_layout(
        height=height,
        template="simple_white",
        margin=dict(l=60, r=20, t=60, b=100),
        xaxis_tickangle=0
    )

    return fig

# Bar Chart - G-INDEX TRAIT SIC 2 DIGIT
def plot_G_per_group_v2(df_Gv2: pd.DataFrame, id_col: str = 'SIC_1digit') -> go.Figure:
    if id_col not in df_Gv2.columns:
        raise ValueError(f"Kolom {id_col} tidak ditemukan pada df_Gv2.")

    trait_cols = [c for c in df_Gv2.columns if c != id_col]
    df_long = df_Gv2.melt(id_vars=[id_col], value_vars=trait_cols, var_name='Trait', value_name='G_i')

    df_long['Trait_Label'] = df_long['Trait'].map(TRAIT_LABELS).fillna(df_long['Trait'])
    trait_order = [TRAIT_LABELS.get(t, t) for t in TRAIT_COLS if t in trait_cols]

    # Color palette
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    trait_palette = {TRAIT_LABELS.get(t, t): c for t, c in zip(TRAIT_COLS, colors)}
    present_traits = [t for t in trait_order]
    color_map = {k: trait_palette[k] for k in present_traits if k in trait_palette}

    fig = px.bar(
        df_long,
        x='Trait_Label',
        y='G_i',
        color='Trait_Label',
        color_discrete_map=color_map,
        category_orders={'Trait_Label': trait_order},
        labels={'Trait_Label': 'Dimensi Kepribadian', 'G_i': 'G_i(Trait)'},
        text_auto='.2f',
        template='simple_white'
    )

    # Atur jarak antar bar dan grup
    fig.update_layout(
        height=600,
        margin=dict(l=40, r=40, t=60, b=40),
        bargap=0.25,
        bargroupgap=0.12,
        showlegend=False
    )

    if trait_order:
        fig.update_xaxes(categoryorder='array', categoryarray=trait_order)

    fig.update_traces(
        hovertemplate="<b>%{x}</b><br>G-index: %{y:.4f}<extra></extra>",
        textposition='outside'
    )

    fig.update_xaxes(title=None)
    fig.update_yaxes(title="G_i(Trait)")

    return fig

# Dendrogram - PEARSON SIMILARITY
def plot_pearson_dendrogram(sim_df: pd.DataFrame, id_col: str = None, title: str = None, height: int = 700):
    mat = sim_df.values.astype(float).copy()

    #Korelasi = 1 → jarak = 0 (sangat mirip)
    # Korelasi = 0 → jarak = 1
    # Korelasi = -1 → jarak = 2
    dist = 1.0 - mat

    dist = (dist + dist.T) / 2.0
    np.fill_diagonal(dist, 0.0)

    condensed = squareform(dist, checks=True)

    linked = linkage(condensed, method='average')

    n = mat.shape[0]
    per_item_px = 28
    padding_px = 140
    max_height_px = 3000
    computed_height = max(height, min(max_height_px, int(n * per_item_px + padding_px)))

    fig = ff.create_dendrogram(
        mat,
        orientation='left',
        labels=sim_df.index.tolist(),
        linkagefun=lambda x: linked
    )

    fig.update_layout(
        height=computed_height,
        template="simple_white",
        margin=dict(l=60, r=20, t=60, b=100),
        xaxis_tickangle=0
    )

    return fig

# Horizontal Bar Chart - G-INDEX PAIRS SIC 1 DIGIT
def plot_G_pairs_bar(df_pairs: pd.DataFrame, height: int = 720) -> go.Figure:
    df = df_pairs.copy()
    if 'Personality Pair' not in df.columns or 'G(Pair)' not in df.columns:
        raise ValueError("df_pairs must contain 'Personality Pair' and 'G(Pair)' columns")

    df_sorted = df.sort_values('G(Pair)', ascending=False).reset_index(drop=True)

    fig = px.bar(
        df_sorted,
        x='G(Pair)',
        y='Personality Pair',
        color='G(Pair)',
        orientation='h',
        color_continuous_scale='Blues',
        text='G(Pair)',
        height=height
    )

    fig.update_traces(
        texttemplate='%{x:.3f}',
        textposition='outside',
        hovertemplate="<b>%{y}</b><br>G(Pair) = %{x:.4f}<extra></extra>"
    )

    fig.update_layout(
        showlegend=False,
        margin=dict(l=220, r=40, t=40, b=40),
        template='simple_white',
        xaxis_title="G(Pair)",
        uniformtext_minsize=8,
        uniformtext_mode='hide'
    )

    fig.update_yaxes(categoryorder='array', categoryarray=list(df_sorted['Personality Pair'][::-1]), automargin=True)
    fig.update_xaxes(automargin=True)

    return fig

# --- Wrapper moved here so every import of plot_radar_chart uses it ---
def enable_radar_wrapper(min_height: int = 520):
    """
    Wrap the module's plot_radar_chart so returned figures enforce a minimum height
    and sensible margins to avoid label clipping. Called at import time.
    """
    global plot_radar_chart
    _orig = plot_radar_chart

    def _wrapped(*args, **kwargs):
        fig = _orig(*args, **kwargs)
        # preserve original update_layout to enforce defaults
        _orig_update = fig.update_layout

        def _enforce_min_height_and_margins(*a, **kw):
            if 'height' in kw:
                kw['height'] = max(kw['height'], min_height)
            else:
                kw.setdefault('height', min_height)
            kw.setdefault('margin', dict(l=80, r=80, t=80, b=80))
            kw.setdefault('autosize', False)
            return _orig_update(*a, **kw)

        fig.update_layout = _enforce_min_height_and_margins

        try:
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(tickfont=dict(size=10)),
                    angularaxis=dict(tickfont=dict(size=11))
                ),
                legend=dict(orientation='h', y=-0.15, x=0.5, xanchor='center')
            )
        except Exception:
            pass

        return fig

    plot_radar_chart = _wrapped

enable_radar_wrapper(min_height=520)