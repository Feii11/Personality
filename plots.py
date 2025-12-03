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

def plot_radar_chart(rata_df: pd.DataFrame, id_col: str, selected_id: str, desc_col: str = None) -> go.Figure:
    """
    Create a radar chart for a single SIC group showing personality trait scores.
    
    Parameters:
    -----------
    rata_df : pd.DataFrame
        DataFrame with mean scores per group
    id_col : str
        Column name for group ID ('SIC_1digit' or 'SIC_2digit')
    selected_id : str
        The specific group ID to visualize
    desc_col : str, optional
        Column name for description
    
    Returns:
    --------
    plotly.graph_objects.Figure
    """
    # Filter for selected group
    row = rata_df[rata_df[id_col].astype(str) == str(selected_id)]
    
    if row.empty:
        # Return empty figure if no data
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        return fig
    
    row = row.iloc[0]
    
    # Get trait values
    values = [row[trait] for trait in TRAIT_COLS]
    labels = [TRAIT_LABELS.get(trait, trait) for trait in TRAIT_COLS]
    
    # Close the radar chart by repeating first value
    values_closed = values + [values[0]]
    labels_closed = labels + [labels[0]]
    
    # Get description if available
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
        # title=f"G-Index Distribution by Personality Trait {title_suffix}",
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
        height=height,
        template="simple_white",
        margin=dict(l=60, r=20, t=60, b=100),
        xaxis_tickangle=0  # Make labels horizontal
    )

    return fig

def plot_G_per_group_v2(df_Gv2: pd.DataFrame, id_col: str = 'SIC_1digit') -> go.Figure:
    """
    Plot G_i(trait) per group (output of compute_G_scores_v2).

    Parameters
    ----------
    df_Gv2 : DataFrame
        Wide DataFrame with first column = id_col and remaining columns = traits.
    id_col : str
        Group identifier column name.

    Returns
    -------
    plotly.graph_objects.Figure
        Grouped bar chart: x = group, y = G_i(trait), color = trait.
    """
    if id_col not in df_Gv2.columns:
        raise ValueError(f"Kolom {id_col} tidak ditemukan pada df_Gv2.")

    trait_cols = [c for c in df_Gv2.columns if c != id_col]
    # Melt to long format
    df_long = df_Gv2.melt(id_vars=[id_col], value_vars=trait_cols, var_name='Trait', value_name='G_i')
    df_long['Trait'] = df_long['Trait'].map(TRAIT_LABELS).fillna(df_long['Trait'])

    fig = px.bar(
        df_long,
        x=id_col,
        y='G_i',
        color='Trait',
        barmode='group',
        labels={id_col: id_col, 'G_i': 'G_i(Trait)'},
        text_auto='.2f',
        template='simple_white'
    )
    fig.update_layout(
        height=600,
        margin=dict(l=40, r=40, t=60, b=40),
        legend_title="Trait"
    )
    fig.update_traces(
        hovertemplate="<b>%{x}</b><br>Trait: %{legendgroup}<br>G_i: %{y:.4f}<extra></extra>",
        textposition='outside'
    )
    return fig

def plot_anova_bar(anova_df: pd.DataFrame, p_thresh: float = 0.05) -> go.Figure:
    """
    Bar chart of F-statistics per trait with significance coloring based on p_thresh.
    Expects anova_df with columns: 'Trait', 'F', 'p'
    """
    df = anova_df.copy()
    df['-log10(p)'] = df['p'].apply(lambda x: -math.log10(x) if pd.notna(x) and x > 0 else float("nan"))
    df['sig'] = df['p'] < p_thresh

    colors = ['#667eea', '#F56565']  # sig / non-sig palette (primary + alert)
    df['color'] = df['sig'].map({True: colors[0], False: colors[1]})

    fig = px.bar(
        df.sort_values('F', ascending=False),
        x='Trait',
        y='F',
        color='sig',
        color_discrete_map={True: colors[0], False: colors[1]},
        labels={'F': 'F-statistic', 'Trait': 'Trait'},
        height=380,
        title="ANOVA: F-statistic per Trait"
    )
    # add p-value as hover
    fig.update_traces(hovertemplate="<b>%{x}</b><br>F = %{y:.3f}<br>p = %{customdata[0]:.4g}<extra></extra>",
                      customdata=df[['p']].values)
    fig.update_layout(showlegend=False, margin=dict(l=20, r=20, t=40, b=20), template="simple_white")
    return fig

def plot_anova_boxplots(df: pd.DataFrame, group_col: str, traits: list = TRAIT_COLS, facet_col_wrap: int = 3) -> go.Figure:
    """
    Faceted boxplots for all traits grouped by group_col.
    Returns a Plotly Figure with facets (small multiples).
    """
    df_long = df[[group_col] + traits].melt(id_vars=[group_col], value_vars=traits, var_name='Trait', value_name='Score')
    # map trait labels if available
    try:
        df_long['Trait_Label'] = df_long['Trait'].map(TRAIT_LABELS)
    except Exception:
        df_long['Trait_Label'] = df_long['Trait']

    fig = px.box(
        df_long,
        x=group_col,
        y='Score',
        color=group_col,
        facet_col='Trait_Label',
        facet_col_wrap=facet_col_wrap,
        points='all',
        height=520,
        color_discrete_sequence=px.colors.qualitative.Plotly
    )
    fig.update_layout(showlegend=False, margin=dict(l=20, r=20, t=40, b=20), template="simple_white")
    # tighten facet axes
    fig.for_each_xaxis(lambda a: a.update(title=''))
    fig.for_each_yaxis(lambda a: a.update(title='Score', range=[1,7]))
    return fig

def plot_pearson_dendrogram(sim_df: pd.DataFrame, id_col: str = None, title: str = None, height: int = 700):
    # defensive copy / numpy array
    mat = sim_df.values.astype(float).copy()

    # Convert correlation to distance
    dist = 1.0 - mat

    # Ensure symmetry and zero diagonal
    dist = (dist + dist.T) / 2.0
    np.fill_diagonal(dist, 0.0)

    # Condensed distance vector required by linkage
    condensed = squareform(dist, checks=True)

    # Build hierarchical clustering linkage from condensed distances
    linked = linkage(condensed, method='average')

    # Create dendrogram using the precomputed linkage
    fig = ff.create_dendrogram(
        mat,  # data passed only so labels/order are preserved by the factory
        orientation='left',
        labels=sim_df.index.tolist(),
        linkagefun=lambda x: linked
    )

    # Update layout to show distance scale that matches 1 - Pearson (max ≈ 2)
    fig.update_layout(
        height=height,
        template="simple_white",
        margin=dict(l=60, r=20, t=60, b=100),
        xaxis_tickangle=0
    )

    return fig

def plot_G_pairs_bar(df_pairs: pd.DataFrame, height: int = 720) -> go.Figure:
    """
    Horizontal bar chart for compute_G_scores_pairs output.
    Expects df_pairs with columns: 'Personality Pair' and 'G(Pair)' columns.
    Sorted so highest G appears at the top.
    """
    df = df_pairs.copy()
    if 'Personality Pair' not in df.columns or 'G(Pair)' not in df.columns:
        raise ValueError("df_pairs must contain 'Personality Pair' and 'G(Pair)' columns")

    # sort so largest G appears first
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
        margin=dict(l=220, r=40, t=40, b=40),  # larger left margin for long category labels
        template='simple_white',
        xaxis_title="G(Pair)",
        uniformtext_minsize=8,
        uniformtext_mode='hide'
    )

    # enforce category order to match sorted df with highest value shown at the top
    # Plotly renders categorical y-axis from bottom-to-top following the provided array,
    # so reverse the sorted list to put largest at the top.
    fig.update_yaxes(categoryorder='array', categoryarray=list(df_sorted['Personality Pair'][::-1]), automargin=True)
    fig.update_xaxes(automargin=True)

    return fig
