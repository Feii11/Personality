# app.py
from typing import TYPE_CHECKING

# Help static analyzers / type-checkers find streamlit without affecting runtime:
if TYPE_CHECKING:
    import streamlit as st   # type: ignore

try:
    import streamlit as st
except Exception:
    # Minimal stub to allow static analysis / non-UI execution when streamlit is not available.
    from types import SimpleNamespace

    def _noop(*a, **k):
        return None

    class _CM:
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc, tb):
            return False

    def _columns(n=1, **kwargs):
        return [_CM() for _ in range(n if isinstance(n, int) else 1)]

    def _tabs(items):
        return [_CM() for _ in items]

    # cache_data should be a decorator; here it is an identity decorator
    def _cache_data(func=None):
        if func is None:
            def _decorator(f):
                return f
            return _decorator
        return func

    st = SimpleNamespace(
        set_page_config=_noop,
        markdown=_noop,
        warning=print,
        error=print,
        file_uploader=lambda *a, **k: None,
        query_params={},
        plotly_chart=_noop,
        metric=_noop,
        columns=_columns,
        tabs=_tabs,
        selectbox=lambda *a, **k: (a[1][0] if len(a) > 1 and a[1] else None),
        radio=lambda *a, **k: (a[1][0] if len(a) > 1 and a[1] else None),
        info=print,
        stop=_noop,
        cache_data=_cache_data
    )

import pandas as pd
from urllib.parse import urlencode
from pathlib import Path

from sic_utils import (
    get_mean_sic1, get_mean_sic2, prepare_data,
    compute_cosine_similarity, compute_pearson_similarity,
    compute_G_scores, compute_G_scores_v2, compute_G_scores_pairs,
    TRAIT_COLS
)
from plots import (
    mean_line_chart, errorbar_plot_from_means,
    plot_cosine_heatmap, plot_cosine_dendogram,
    plot_G_bar, pie_chart_styled, TRAIT_LABELS,
    plot_G_per_group_v2, plot_radar_chart,
    plot_pearson_dendrogram, plot_G_pairs_bar,
)

st.set_page_config(
    page_title="CEO Personality Analytics",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Simplified CSS loader (reads external styles.css)
def inject_css():
    css_path = Path(__file__).parent / "styles.css"
    if css_path.exists():
        css_text = css_path.read_text(encoding="utf-8")
        st.markdown(f"<style>{css_text}</style>", unsafe_allow_html=True)
    else:
        st.warning(f"styles.css not found: {css_path}")

inject_css()

# Global registry so any chart rendered with show_chart_with_expand is available for expand handling
CHART_REGISTRY = {}

# -------------------------
# Utility helpers for expand behavior
# -------------------------
def _current_qs():
    # Use st.query_params and return a shallow copy so we don't mutate Streamlit's internal state
    params = dict(st.query_params)
    # keep as dict of lists for urlencode(doseq=True)
    return params

def _make_qs(params):
    # params expected as dict where values are lists
    flat = {}
    for k, v in params.items():
        # if value already list, pass through
        flat[k] = v
    return "?" + urlencode(flat, doseq=True) if flat else ""

def build_expand_url(chart_id):
    params = _current_qs()
    params['expand'] = [chart_id]
    return _make_qs(params)

def build_close_url():
    params = _current_qs()
    params.pop('expand', None)
    return _make_qs(params)

def show_chart_with_expand(fig, chart_id, height=None, config=None):
    # Register the figure so expand logic can find it deterministically
    try:
        CHART_REGISTRY[chart_id] = fig
    except Exception:
        # non-fatal; continue to render
        pass

    # Render chart
    kwargs = {}
    if height:
        fig.update_layout(height=height)
    if config is None:
        config = {'displayModeBar': False}
    st.plotly_chart(fig, use_container_width=True, config=config)
    # Render hover-only expand link placed right after the chart (CSS uses sibling selector)
    href = build_expand_url(chart_id)
    html = f'<a class="expand-link" href="{href}" target="_self">🔍 Expand</a>'
    st.markdown(html, unsafe_allow_html=True)

# -------------------------
# Cache data loading
# -------------------------
@st.cache_data
def load_file(uploaded_file):
    fname = uploaded_file.name.lower()
    try:
        if fname.endswith(".csv"):
            return pd.read_csv(uploaded_file)
        else:
            return pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        return None

@st.cache_data
def prepare_and_compute(upload):
    dfCEO = load_file(upload)
    dfSIC = pd.read_excel('data/2 digit.xlsx')
    dfCEO, dfSIC = prepare_data(dfCEO, dfSIC)

    # Ensure trait columns are numeric to prevent operations on datetimes/objects
    if dfCEO is None:
        raise ValueError("Uploaded file could not be read")
    dfCEO[TRAIT_COLS] = dfCEO[TRAIT_COLS].apply(pd.to_numeric, errors='coerce')

    rata_rata1 = get_mean_sic1(dfCEO, dfSIC)
    rata_rata2 = get_mean_sic2(dfCEO, dfSIC)

    # Fill missing trait values in the aggregated dataframes with column means to avoid downstream NaN errors
    for agg_df in (rata_rata1, rata_rata2):
        if agg_df is None:
            continue
        for col in TRAIT_COLS:
            if col in agg_df.columns:
                if agg_df[col].isna().any():
                    col_mean = agg_df[col].mean(skipna=True)
                    if pd.isna(col_mean):
                        col_mean = 0.0
                    agg_df[col] = agg_df[col].fillna(col_mean)

    # Menghitung overall mean untuk semua CEO
    overall_mean = dfCEO[TRAIT_COLS].mean(numeric_only=True).to_frame().T
    
    # Menambahkan kolom SIC_1digit dan Description_1 untuk overall_mean
    overall_mean.insert(0, 'SIC_1digit', 'All')
    overall_mean.insert(1, 'Description_1', 'All CEOs Combined')

    cosine_sim1 = compute_cosine_similarity(rata_rata1, 'SIC_1digit')
    cosine_sim2 = compute_cosine_similarity(rata_rata2, 'SIC_2digit')
    pearson1 = compute_pearson_similarity(rata_rata1, 'SIC_1digit')
    pearson2 = compute_pearson_similarity(rata_rata2, 'SIC_2digit')

    df_G1 = compute_G_scores(rata_rata1, 'SIC_1digit')
    df_G2 = compute_G_scores(rata_rata2, 'SIC_2digit')
    df_Gv2_sic1 = compute_G_scores_v2(rata_rata2)

    return (dfCEO, dfSIC, rata_rata1, rata_rata2, overall_mean,
            cosine_sim1, cosine_sim2, pearson1, pearson2,
            df_G1, df_G2, df_Gv2_sic1)

# -------------------------
# Header
# -------------------------
st.markdown("""
<div class="dashboard-header">
    <h1 class="dashboard-title">CEO Personality Analytics Dashboard</h1>
    <p class="dashboard-subtitle">Comprehensive personality analysis based on SIC classification</p>
</div>
""", unsafe_allow_html=True)

# UPLOAD FILE
upload = st.file_uploader(
    "📁 Upload CEO Data (CSV / Excel)",
    type=["csv", "xls", "xlsx"],
    help="Upload your CEO personality dataset"
)


if upload is not None:
    (dfCEO, dfSIC, rata_rata1, rata_rata2, overall_mean,
     cosine_sim1, cosine_sim2, pearson1, pearson2,
     df_G1, df_G2, df_Gv2_sic1) = prepare_and_compute(upload)
    
    figpie = pie_chart_styled(dfCEO, column='SIC_1digit')
    min_height = 520
    fig1 = mean_line_chart(rata_rata1, id_col='SIC_1digit', desc_col='Description_1', selected_ids=None)

    _orig_update_fig1 = fig1.update_layout
    def _enforce_min_height_fig1(*args, **kwargs):
        if 'height' in kwargs:
            kwargs['height'] = max(kwargs['height'], min_height)
        else:
            kwargs.setdefault('height', min_height)
        return _orig_update_fig1(*args, **kwargs)
    fig1.update_layout = _enforce_min_height_fig1

    # ensure fig1 reserves enough space for radar labels
    fig1.update_layout(height=min_height, margin=dict(l=80, r=80, t=80, b=80), autosize=False)

    # Radar (tab1)
    # Note: tab-specific selection will still work; initial radar default is 'All'
    fig_radar_all = plot_radar_chart(overall_mean, id_col='SIC_1digit', selected_id='All', desc_col='Description_1')

    # Error bars
    rr1_long = rata_rata1.melt(
        id_vars=['SIC_1digit', 'Description_1'],
        value_vars=TRAIT_COLS,
        var_name='Dimensi_Kepribadian',
        value_name='Rata_Rata'
    )
    rr1_long['Dimensi_Kepribadian'] = rr1_long['Dimensi_Kepribadian'].map(TRAIT_LABELS)

    fig_err1 = errorbar_plot_from_means(rr1_long, title_suffix="")
    fig_err1.update_layout(height=520, margin=dict(l=20, r=20, t=20, b=20))
    fig_cos_default_1 = plot_cosine_heatmap(cosine_sim1, id_col='SIC_1digit')
    fig_cos_default_1.update_layout(height=340, margin=dict(l=20, r=20, t=20, b=20))
    fig_pear_default_1 = plot_cosine_heatmap(pearson1, id_col='SIC_1digit')
    fig_pear_default_1.update_layout(height=340, margin=dict(l=20, r=20, t=20, b=20))
    fig_g1 = plot_G_bar(df_G1, title_suffix="")
    fig_g1.update_layout(height=520, margin=dict(l=20, r=20, t=30, b=20))

    # G-Pairs moved to be displayed inside the SIC 1 tab (below G-Index) to avoid duplicate rendering

    # Tab2 figures
    # default: show all SIC2 aggregated trends; filtering applied at render-time
    fig2_default = mean_line_chart(rata_rata2, id_col='SIC_2digit', desc_col='Description_2', selected_ids=None)
    # Ensure the line chart uses the same minimum height as radar charts
    fig2_default.update_layout(height=min_height, margin=dict(l=20, r=20, t=20, b=20), autosize=False)

    # Radar tab2
    fig_radar2_default = plot_radar_chart(rata_rata2, id_col='SIC_2digit', selected_id=None, desc_col='Description_2')

    # Errorbars SIC2 default
    grouped_long_default = rata_rata2.melt(
        id_vars=['SIC_2digit', 'SIC_1digit', 'Description_2'],
        value_vars=TRAIT_COLS,
        var_name='Dimensi_Kepribadian',
        value_name='Rata_Rata'
    )
    grouped_long_default['Dimensi_Kepribadian'] = grouped_long_default['Dimensi_Kepribadian'].map(TRAIT_LABELS)
    fig_err2_default = errorbar_plot_from_means(grouped_long_default, title_suffix="")
    fig_err2_default.update_layout(height=520, margin=dict(l=20, r=20, t=20, b=20))

    # Cosine / Pearson SIC2 defaults
    fig_cos_default_2 = plot_cosine_heatmap(cosine_sim2, id_col='SIC_2digit')
    fig_cos_default_2.update_layout(height=340, margin=dict(l=20, r=20, t=20, b=20))
    fig_pear_default_2 = plot_cosine_heatmap(pearson2, id_col='SIC_2digit')
    fig_pear_default_2.update_layout(height=340, margin=dict(l=20, r=20, t=20, b=20))

    # G per group
    # We'll defer building the group-specific bar until selection, but keep a default bar as placeholder
    fig_g2_default = None

    # Build a mapping of chart_id -> figure for expand popups (register defaults)
    chart_map = {
        'pie': figpie,
        'mean_sic1': fig1,
        'radar_sic1_all': fig_radar_all,
        'err_sic1': fig_err1,
        'cos_sic1': fig_cos_default_1,
        'pear_sic1': fig_pear_default_1,
        'g_sic1': fig_g1,
        'mean_sic2': fig2_default,
        'radar_sic2_default': fig_radar2_default,
        'err_sic2': fig_err2_default,
        'cos_sic2': fig_cos_default_2,
        'pear_sic2': fig_pear_default_2
        # g_sic2 will be added dynamically when created per selection
    }
    # ensure registry also knows defaults so expand works when user opens expand URL directly
    CHART_REGISTRY.update(chart_map)

    # If user requested an expanded view via query param, show full-size chart at top and provide a Close link
    q = dict(st.query_params)
    if 'expand' in q:
        chart_to_expand = q.get('expand')[0]
        # First try registry / map
        expanded_fig = CHART_REGISTRY.get(chart_to_expand) or chart_map.get(chart_to_expand)

        if expanded_fig is None:
            # dynamic g_sic2 variations: chart id might be 'g_sic2_<SIC1>'
            if chart_to_expand.startswith('g_sic2_'):
                sic1_key = chart_to_expand.replace('g_sic2_', '')
                df_sic1 = df_Gv2_sic1[df_Gv2_sic1['SIC_1digit'].astype(str) == sic1_key]
                keep_cols = ['SIC_1digit'] + [c for c in TRAIT_COLS if c in df_sic1.columns]
                df_sic1_clean = df_sic1[keep_cols].copy()
                if len(df_sic1_clean) > 1:
                    df_sic1_clean = df_sic1_clean.groupby('SIC_1digit', as_index=False).mean(numeric_only=True)
                if not df_sic1_clean.empty:
                    expanded_fig = plot_G_per_group_v2(df_sic1_clean, id_col='SIC_1digit')
            # radar variations: radar_sic1_<id> or radar_sic2_<id>
            elif chart_to_expand.startswith('radar_sic1_'):
                selected = chart_to_expand.replace('radar_sic1_', '')
                if selected.lower() == 'all' or selected == 'All':
                    expanded_fig = plot_radar_chart(overall_mean, id_col='SIC_1digit', selected_id='All', desc_col='Description_1')
                else:
                    expanded_fig = plot_radar_chart(rata_rata1, id_col='SIC_1digit', selected_id=selected, desc_col='Description_1')
            elif chart_to_expand.startswith('radar_sic2_'):
                selected = chart_to_expand.replace('radar_sic2_', '')
                expanded_fig = plot_radar_chart(rata_rata2, id_col='SIC_2digit', selected_id=selected, desc_col='Description_2')

        if expanded_fig is not None:
            st.markdown(f'<a class="close-expand" href="{build_close_url()}">✖ Close</a>', unsafe_allow_html=True)
            # render big
            expanded_fig.update_layout(height=900)
            st.plotly_chart(expanded_fig, use_container_width=True, config={'displayModeBar': True})
            st.stop()  # stop further rendering of the app while expanded view is shown

    # -------------------------
    # Normal app rendering with helper that adds hover-only expand link after each chart
    # -------------------------
    cols = st.columns([3, 1, 1, 1, 1, 1])
    metric_cols = st.columns(5, gap="large")
    with metric_cols[0]:
        st.metric("Total Records", f"{len(dfCEO):,}")
    with metric_cols[1]:
        st.metric("Total CEOs", f"{dfCEO['EXEC_FULLNAME'].nunique():,}")
    with metric_cols[2]:
        st.metric("SIC 1-Digit", f"{dfCEO['SIC_1digit'].nunique()}")
    with metric_cols[3]:
        st.metric("SIC 2-Digit", f"{dfCEO['SIC_2digit'].nunique()}")
    with metric_cols[4]:
        st.metric("Personality Traits", f"{len(TRAIT_COLS)}")

    st.markdown('<div style="height: 0.75rem;"></div>', unsafe_allow_html=True)

    st.markdown('<div><p class="chart-title">Industry Distribution</p></div>', unsafe_allow_html=True)
    show_chart_with_expand(figpie, 'pie')

    # TWO TABS: SIC 1 and SIC 2
    tab1, tab2 = st.tabs(["🏢 SIC 1 Digit Analysis", "🏭 SIC 2 Digit Analysis"])

    # TAB 1
    with tab1:
        st.markdown(f'<p class="caption-text">✓ Analyzing {rata_rata1["SIC_1digit"].nunique()} categories</p>', unsafe_allow_html=True)

        # Row 1: Personality Profile + Radar
        col1a, col1b = st.columns([2, 1])

        with col1a:
            st.markdown('<p class="chart-title">Rata-Rata Personality Traits</p>', unsafe_allow_html=True)
            show_chart_with_expand(fig1, 'mean_sic1')
            try:
                fig1["layout"].update(height=min_height, margin=dict(l=80, r=80, t=80, b=80), autosize=False)
            except Exception:
                try:
                    fig1.update_layout(height=min_height, margin=dict(l=80, r=80, t=80, b=80), autosize=False)
                except Exception:
                    pass

            CHART_REGISTRY['mean_sic1'] = fig1
        with col1b:
            st.markdown('<p class="chart-title">Detail Kategori</p>', unsafe_allow_html=True)
            list_sic1_all = ['All'] + sorted(rata_rata1['SIC_1digit'].astype(str).unique())
            selected_sic1 = st.selectbox("", options=list_sic1_all, key="tab1_radar", label_visibility="collapsed")

            if selected_sic1 == 'All':
                fig_radar = plot_radar_chart(overall_mean, id_col='SIC_1digit', selected_id='All', desc_col='Description_1')
                show_chart_with_expand(fig_radar, 'radar_sic1_all')
            else:
                fig_radar = plot_radar_chart(rata_rata1, id_col='SIC_1digit', selected_id=selected_sic1, desc_col='Description_1')
                show_chart_with_expand(fig_radar, f'radar_sic1_{selected_sic1}')

        # Row 2: Standard Deviation and G-Index
        
        err_col, g_col = st.columns([1, 1])
        with err_col:
            st.markdown('<p class="chart-title">Standar Deviasi</p>', unsafe_allow_html=True)
            show_chart_with_expand(fig_err1, 'err_sic1')
        with g_col:
            st.markdown('<p class="chart-title">G-Traits Index</p>', unsafe_allow_html=True)
            show_chart_with_expand(fig_g1, 'g_sic1')
 
        # Row 3: G-Pairs (pairwise G for trait pairs)
        st.markdown('<p class="chart-title">G-Traits Index Pairs</p>', unsafe_allow_html=True)
        try:
            df_pairs = compute_G_scores_pairs(rata_rata1, 'SIC_1digit')
            if not df_pairs.empty:
                fig_pairs = plot_G_pairs_bar(df_pairs, height=420)
                show_chart_with_expand(fig_pairs, 'g_pairs_sic1')
        except Exception as e:
            st.warning(f"Unable to compute/plot G-pairs: {e}")
        
        # Row 4: Similarity - Cosine and Pearson
        sim_col_cos, sim_col_pear = st.columns(2)

        with sim_col_cos:
            st.markdown('<div style="font-weight:700; margin-bottom:0.3rem;">Cosine Similarity</div>', unsafe_allow_html=True)
            cos_viz_choice = st.radio("", ["Dendrogram", "Heatmap"], horizontal=True, key="tab1_cos_viz", label_visibility="collapsed")
            if cos_viz_choice == "Dendrogram":
                fig_cos = plot_cosine_dendogram(cosine_sim1, id_col='SIC_1digit')
            else:
                fig_cos = plot_cosine_heatmap(cosine_sim1, id_col='SIC_1digit')
            fig_cos.update_layout(height=340, margin=dict(l=20, r=20, t=20, b=20))
            show_chart_with_expand(fig_cos, 'cos_sic1')

        with sim_col_pear:
            st.markdown('<div style="font-weight:700; margin-bottom:0.3rem;">Pearson Correlation</div>', unsafe_allow_html=True)
            pear_viz_choice = st.radio("", ["Dendrogram", "Heatmap"], horizontal=True, key="tab1_pear_viz", label_visibility="collapsed")
            if pear_viz_choice == "Dendrogram":
                fig_pear = plot_pearson_dendrogram(pearson1, id_col='SIC_1digit')
            else:
                fig_pear = plot_cosine_heatmap(pearson1, id_col='SIC_1digit')
            fig_pear.update_layout(height=340, margin=dict(l=20, r=20, t=20, b=20))
            show_chart_with_expand(fig_pear, 'pear_sic1')

    # TAB 2
    with tab2:
        sic1_filter = st.selectbox(
            "Filter by SIC 1 Digit:",
            ['All'] + sorted(rata_rata2['SIC_1digit'].astype(str).unique()),
            key="tab2_filter"
        )

        if sic1_filter == 'All':
            rata_rata2_filtered = rata_rata2.copy()
        else:
            # Filter by SIC_1digit (was incorrectly filtering SIC_2digit)
            rata_rata2_filtered = rata_rata2[rata_rata2['SIC_1digit'].astype(str) == sic1_filter]

        num_categories = len(rata_rata2_filtered)
        if num_categories > 0:
            ceos_mask = dfCEO['SIC_2digit'].astype(str).isin(rata_rata2_filtered['SIC_2digit'].astype(str))
            num_ceos = int(dfCEO.loc[ceos_mask, 'EXEC_FULLNAME'].nunique())
            num_records = int(dfCEO.loc[ceos_mask].shape[0])
            st.markdown(f'<p class="caption-text">✓ Analyzing {num_categories} categories</p>', unsafe_allow_html=True)
        else:
            st.info("No data available")
            st.stop()

        # Row 1 - MEAN PERSONALITY SIC 2 DIGIT
        col1a, col1b = st.columns([2, 1])

        # MEAN - Line Chart
        with col1a:
            st.markdown('<p class="chart-title">Rata-Rata Personality Traits</p>', unsafe_allow_html=True)
            fig2 = mean_line_chart(rata_rata2_filtered, id_col='SIC_2digit', desc_col='Description_2', selected_ids=None)
            fig2.update_layout(height=min_height, margin=dict(l=20, r=20, t=20, b=20), autosize=False)
            CHART_REGISTRY['mean_sic2'] = fig2
            show_chart_with_expand(fig2, 'mean_sic2')

        # MEAN - Radar Chart
        with col1b:
            st.markdown('<p class="chart-title">Detail Kategori</p>', unsafe_allow_html=True)
            if len(rata_rata2_filtered) > 0:
                list_sic2 = sorted(rata_rata2_filtered['SIC_2digit'].astype(str).unique())
                selected_sic2 = st.selectbox("", options=list_sic2, key="tab2_radar", label_visibility="collapsed")
                fig_radar2 = plot_radar_chart(rata_rata2, id_col='SIC_2digit', selected_id=selected_sic2, desc_col='Description_2')
                fig_radar2.update_layout(height=320, margin=dict(l=20, r=20, t=20, b=20))
                # register radar for expand with deterministic id
                show_chart_with_expand(fig_radar2, f'radar_sic2_{selected_sic2}')

        # Row 2 - STANDAR DEVIASI DAN G-INDEX SIC 2 DIGIT
        col_err2, col_g2 = st.columns([1, 1])

        # STANDAR DEVIASI - Error Bar Chart
        with col_err2:
            st.markdown('<p class="chart-title">Standar Deviasi</p>', unsafe_allow_html=True)
            grouped_long = rata_rata2_filtered.melt(
            id_vars=['SIC_2digit', 'SIC_1digit', 'Description_2'],
            value_vars=TRAIT_COLS,
            var_name='Dimensi_Kepribadian',
            value_name='Rata_Rata'
            )
            grouped_long['Dimensi_Kepribadian'] = grouped_long['Dimensi_Kepribadian'].map(TRAIT_LABELS)
            fig_err2 = errorbar_plot_from_means(grouped_long, title_suffix="")
            fig_err2.update_layout(height=520, margin=dict(l=20, r=20, t=20, b=20))
            # register updated err_sic2 for expand
            CHART_REGISTRY['err_sic2'] = fig_err2
            show_chart_with_expand(fig_err2, 'err_sic2')

        # G-INDEX PER GROUP - Bar Chart
        with col_g2:
            st.markdown('<p class="chart-title">G-Index Per Group Analysis</p>', unsafe_allow_html=True)
            selected_sic1_g = sic1_filter

            # User harus memilih SIC 1 Digit spesifik untuk melihat G-Index per SIC 1 Digit
            if selected_sic1_g == 'All':
                st.info("Select a specific SIC 1 Digit using the 'Filter by SIC 1 Digit' dropdown above to view G-Index per group.")
                df_sic1_clean = pd.DataFrame(columns=['SIC_1digit'] + TRAIT_COLS)
            else:
                df_sic1 = df_Gv2_sic1[df_Gv2_sic1['SIC_1digit'].astype(str) == selected_sic1_g]
                keep_cols = ['SIC_1digit'] + [c for c in TRAIT_COLS if c in df_sic1.columns]
                df_sic1_clean = df_sic1[keep_cols].copy()
        
            if len(df_sic1_clean) > 1:
                df_sic1_clean = df_sic1_clean.groupby('SIC_1digit', as_index=False).mean(numeric_only=True)

            num_sic2 = rata_rata2[rata_rata2['SIC_1digit'].astype(str) == selected_sic1_g]['SIC_2digit'].nunique()

            if not df_sic1_clean.empty:
                fig_bar = plot_G_per_group_v2(df_sic1_clean, id_col='SIC_1digit')
                # Buat atur tingginya
                try:
                    fig_bar.update_layout(height=min_height, margin=dict(l=20, r=20, t=20, b=20), autosize=False)
                except Exception:
                    pass
                # daftarkan dengan ID unik berdasarkan SIC 1 digit yang dipilih
                chart_id_for_g2 = f'g_sic2_{selected_sic1_g}'
                chart_map[chart_id_for_g2] = fig_bar
                CHART_REGISTRY[chart_id_for_g2] = fig_bar
                show_chart_with_expand(fig_bar, chart_id_for_g2)
            else:
                st.info("No data available for this selection")
        
        # Row 3: COSINE SIMILARITY AND PEARSON CORRELATION SIC 2 DIGIT
        st.markdown('<p class="chart-title">Similarity (Cosine vs Pearson)</p>', unsafe_allow_html=True)
        sim_col_cos2, sim_col_pear2 = st.columns(2)

        # COSINE SIMILARITY
        with sim_col_cos2:
            st.markdown('<div style="font-weight:700; margin-bottom:0.3rem;">Cosine Similarity</div>', unsafe_allow_html=True)
            cos_viz_choice2 = st.radio("", ["Dendrogram", "Heatmap"], horizontal=True, key="tab2_cos_viz", label_visibility="collapsed")

            # Try to subset the similarity matrix to only SIC-2 categories in the current SIC-1 filter
            def _subset_sim_matrix(sim_df):
                if sic1_filter == 'All':
                    return sim_df
                try:
                    ids = rata_rata2_filtered['SIC_2digit'].astype(str).unique().tolist()
                    # boolean masks using stringified index/columns to be robust vs int labels
                    row_mask = sim_df.index.map(str).isin(ids)
                    col_mask = sim_df.columns.map(str).isin(ids)
                    sim_sub = sim_df.loc[row_mask, col_mask]
                    # if submatrix has at least one element return it, otherwise fall back to original
                    if sim_sub.shape[0] > 0 and sim_sub.shape[1] > 0:
                        return sim_sub
                except Exception:
                    pass
                return sim_df

            sim_cos_to_plot = _subset_sim_matrix(cosine_sim2)

            if cos_viz_choice2 == "Dendrogram":
                fig_cos2 = plot_cosine_dendogram(sim_cos_to_plot, id_col='SIC_2digit')
            else:
                fig_cos2 = plot_cosine_heatmap(sim_cos_to_plot, id_col='SIC_2digit')

            try:
                fig_cos2.update_layout(height=min_height, margin=dict(l=20, r=20, t=20, b=20), autosize=False)
            except Exception:
                fig_cos2.update_layout(height=520, margin=dict(l=20, r=20, t=20, b=20))

            CHART_REGISTRY['cos_sic2'] = fig_cos2
            show_chart_with_expand(fig_cos2, 'cos_sic2')

        # PEARSON CORRELATION
        with sim_col_pear2:
            st.markdown('<div style="font-weight:700; margin-bottom:0.3rem;">Pearson Similarity</div>', unsafe_allow_html=True)
            pear_viz_choice2 = st.radio("", ["Dendrogram", "Heatmap"], horizontal=True, key="tab2_pear_viz", label_visibility="collapsed")

            # reuse the same subsetting logic
            sim_pear_to_plot = _subset_sim_matrix(pearson2)

            if pear_viz_choice2 == "Dendrogram":
                fig_pear2 = plot_pearson_dendrogram(sim_pear_to_plot, id_col='SIC_2digit')
            else:
                fig_pear2 = plot_cosine_heatmap(sim_pear_to_plot, id_col='SIC_2digit')

            try:
                fig_pear2.update_layout(height=min_height, margin=dict(l=20, r=20, t=20, b=20), autosize=False)
            except Exception:
                fig_pear2.update_layout(height=520, margin=dict(l=20, r=20, t=20, b=20))

            CHART_REGISTRY['pear_sic2'] = fig_pear2
            show_chart_with_expand(fig_pear2, 'pear_sic2')

    # Minimal footer
    st.markdown("""
    <div style='text-align: center; color: white; padding: 0.6rem; background: rgba(255,255,255,0.04); border-radius: 8px;'>
        <p style='margin: 0; font-weight: 600;'>CEO Personality Analytics Dashboard</p>
    </div>
    """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div style='background: white; border-radius: 12px; padding: 1.5rem; text-align: center; box-shadow: 0 8px 24px rgba(0,0,0,0.06);'>
        <h3 style='color: #667eea; margin-bottom: 0.5rem;'> Welcome to CEO Personality Analytics</h3>
        <p style='color: #666; font-size: 0.95rem; margin-bottom: 1rem;'>
            Upload your CEO dataset (CSV or Excel) to begin personality analysis by SIC classification.
        </p>
    </div>
    """, unsafe_allow_html=True)
