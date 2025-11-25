# app.py
import streamlit as st
import pandas as pd
from urllib.parse import urlencode

from sic_utils import (
    get_mean_sic1, get_mean_sic2, prepare_data,
    compute_cosine_similarity, compute_pearson_similarity,
    compute_G_scores, compute_G_scores_v2, TRAIT_COLS,
    compute_anova, compute_tukey
)
from plots import (
    mean_line_chart, errorbar_plot_from_means,
    plot_cosine_heatmap, plot_cosine_dendogram,
    plot_G_bar, pie_chart_styled, TRAIT_LABELS,
    plot_G_per_group_v2, plot_radar_chart,
    plot_anova_bar, plot_anova_boxplots
)

st.set_page_config(
    page_title="CEO Personality Analytics",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Simplified CSS with reduced padding / whitespace + hover-expand styles
def inject_css():
    st.markdown("""
    <style>
        .main { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 0; }
        .block-container { padding: 1rem 1.5rem; max-width: 100%; }
        .dashboard-header { background: white; border-radius: 12px; padding: 1rem 1.5rem; margin-bottom: 1rem; box-shadow: 0 6px 20px rgba(0,0,0,0.06); }
        .dashboard-title { font-size: 1.6rem; font-weight: 800; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0; }
        .dashboard-subtitle { color: #666; font-size: 0.95rem; margin-top: 0.25rem; }
        div[data-testid="stMetricValue"] { font-size: 1.2rem; font-weight: 700; color: #1a1a1a; }
        div[data-testid="stMetricLabel"] { font-size: 0.85rem; color: #666; font-weight: 500; }
        .stTabs [data-baseweb="tab-list"] { gap: 1rem; background: white; padding: 0.6rem 1rem; border-radius: 10px; box-shadow: 0 4px 14px rgba(0,0,0,0.06); }
        .stTabs [data-baseweb="tab"] { height: 44px; padding: 0 1rem; color: #666; font-weight: 600; font-size: 0.95rem; }
        .stTabs [aria-selected="true"] { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white !important; border-radius: 8px; }
        .chart-title { font-size: 1.05rem; font-weight: 700; color: #1a1a1a; margin-bottom: 0.6rem; }
        .caption-text { color: #10b981; font-weight: 500; font-size: 0.85rem; }

        /* Expand button: hidden by default, shown when the chart area is hovered */
        .expand-link {
            display: inline-block;
            float: right;
            margin-top: -40px;
            margin-bottom: 8px;
            padding: 6px 10px;
            background: rgba(0,0,0,0.65);
            color: white !important;
            border-radius: 6px;
            font-weight: 700;
            text-decoration: none;
            opacity: 0;
            transition: opacity 0.15s ease-in-out;
            z-index: 999;
        }
        /* Streamlit renders plotly charts inside elements with class 'stPlotlyChart' */
        .stPlotlyChart:hover + .expand-link,
        .stPlotlyChart:hover ~ .expand-link {
            opacity: 1;
        }

        /* Large view close link */
        .close-expand {
            display: inline-block;
            margin-bottom: 10px;
            padding: 8px 12px;
            background: rgba(0,0,0,0.65);
            color: white !important;
            border-radius: 6px;
            font-weight: 700;
            text-decoration: none;
        }
    </style>
    """, unsafe_allow_html=True)

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

    # compute overall mean using numeric_only to avoid datetime/object arithmetic
    overall_mean = dfCEO[TRAIT_COLS].mean(numeric_only=True).to_frame().T
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

upload = st.file_uploader(
    "📁 Upload CEO Data (CSV / Excel)",
    type=["csv", "xls", "xlsx"],
    help="Upload your CEO personality dataset"
)

# Expand popup handling will happen after computing the figures (we need the figures to exist)
if upload is not None:
    (dfCEO, dfSIC, rata_rata1, rata_rata2, overall_mean,
     cosine_sim1, cosine_sim2, pearson1, pearson2,
     df_G1, df_G2, df_Gv2_sic1) = prepare_and_compute(upload)

    # compute ANOVA results on raw CEO-level data (defensive numeric coercion already applied earlier)
    try:
        anova_sic1 = compute_anova(dfCEO, 'SIC_1digit', trait_cols=TRAIT_COLS)
        anova_sic2 = compute_anova(dfCEO, 'SIC_2digit', trait_cols=TRAIT_COLS)
    except Exception as e:
        anova_sic1 = None
        anova_sic2 = None
        st.warning(f"ANOVA computation failed: {e}")

    # Build all figures first (same as before) but don't immediately plot — we will plot and let helper show expand link
    figpie = pie_chart_styled(dfCEO, column='SIC_1digit')

    # make line chart taller and ensure radar charts are taller too
    min_height = 520
    fig1 = mean_line_chart(rata_rata1, id_col='SIC_1digit', desc_col='Description_1', selected_ids=None)

    # enforce minimum height on this specific fig1 (so later .update_layout calls won't shrink it)
    _orig_update_fig1 = fig1.update_layout
    def _enforce_min_height_fig1(*args, **kwargs):
        if 'height' in kwargs:
            kwargs['height'] = max(kwargs['height'], min_height)
        else:
            kwargs.setdefault('height', min_height)
        return _orig_update_fig1(*args, **kwargs)
    fig1.update_layout = _enforce_min_height_fig1

    # wrap plot_radar_chart so any radar figs created later have a minimum height enforced
    _orig_plot_radar = plot_radar_chart
    def _wrapped_plot_radar(*args, **kwargs):
        fig = _orig_plot_radar(*args, **kwargs)
        _orig_update = fig.update_layout
        def _enforce_min_height(*a, **kw):
            if 'height' in kw:
                kw['height'] = max(kw['height'], min_height)
            else:
                kw.setdefault('height', min_height)
            return _orig_update(*a, **kw)
        fig.update_layout = _enforce_min_height
        return fig
    plot_radar_chart = _wrapped_plot_radar
    fig1.update_layout(height=320, margin=dict(l=20, r=20, t=20, b=20))

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

    # Similarities SIC1
    # Use dendrogram/heatmap generation later but prepare both options when needed
    # We'll store the computed matrices and build fig on selection
    # For default display build heatmaps/dendrograms now
    fig_cos_default_1 = plot_cosine_heatmap(cosine_sim1, id_col='SIC_1digit')
    fig_cos_default_1.update_layout(height=340, margin=dict(l=20, r=20, t=20, b=20))
    fig_pear_default_1 = plot_cosine_heatmap(pearson1, id_col='SIC_1digit')
    fig_pear_default_1.update_layout(height=340, margin=dict(l=20, r=20, t=20, b=20))

    fig_g1 = plot_G_bar(df_G1, title_suffix="")
    # Make G-Index bar chart taller for better readability
    fig_g1.update_layout(height=520, margin=dict(l=20, r=20, t=30, b=20))

    # Tab2 figures
    # default: show all SIC2 aggregated trends; filtering applied at render-time
    fig2_default = mean_line_chart(rata_rata2, id_col='SIC_2digit', desc_col='Description_2', selected_ids=None)
    fig2_default.update_layout(height=320, margin=dict(l=20, r=20, t=20, b=20))

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
            st.markdown('<p class="chart-title">Personality Trends</p>', unsafe_allow_html=True)
            show_chart_with_expand(fig1, 'mean_sic1')

        with col1b:
            st.markdown('<p class="chart-title">Category Detail</p>', unsafe_allow_html=True)
            list_sic1_all = ['All'] + sorted(rata_rata1['SIC_1digit'].astype(str).unique())
            selected_sic1 = st.selectbox("", options=list_sic1_all, key="tab1_radar", label_visibility="collapsed")

            if selected_sic1 == 'All':
                fig_radar = plot_radar_chart(overall_mean, id_col='SIC_1digit', selected_id='All', desc_col='Description_1')
                show_chart_with_expand(fig_radar, 'radar_sic1_all')
            else:
                fig_radar = plot_radar_chart(rata_rata1, id_col='SIC_1digit', selected_id=selected_sic1, desc_col='Description_1')
                show_chart_with_expand(fig_radar, f'radar_sic1_{selected_sic1}')

        # Row 2: Standard Deviation (big, full width)
        st.markdown('<p class="chart-title">Standard Deviation</p>', unsafe_allow_html=True)
        show_chart_with_expand(fig_err1, 'err_sic1')

        # Row 3: Similarity - Cosine and Pearson side-by-side
        st.markdown('<p class="chart-title">Similarity (Cosine vs Pearson)</p>', unsafe_allow_html=True)
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
            st.markdown('<div style="font-weight:700; margin-bottom:0.3rem;">Pearson Similarity</div>', unsafe_allow_html=True)
            pear_viz_choice = st.radio("", ["Dendrogram", "Heatmap"], horizontal=True, key="tab1_pear_viz", label_visibility="collapsed")
            if pear_viz_choice == "Dendrogram":
                fig_pear = plot_cosine_dendogram(pearson1, id_col='SIC_1digit')
            else:
                fig_pear = plot_cosine_heatmap(pearson1, id_col='SIC_1digit')
            fig_pear.update_layout(height=340, margin=dict(l=20, r=20, t=20, b=20))
            show_chart_with_expand(fig_pear, 'pear_sic1')

        # Row 4: G-Index
        st.markdown('<p class="chart-title">G-Index Distribution</p>', unsafe_allow_html=True)
        show_chart_with_expand(fig_g1, 'g_sic1')

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

        # Row 1
        col1a, col1b = st.columns([2, 1])

        with col1a:
            st.markdown('<p class="chart-title">Personality Trends</p>', unsafe_allow_html=True)
            fig2 = mean_line_chart(rata_rata2_filtered, id_col='SIC_2digit', desc_col='Description_2', selected_ids=None)
            fig2.update_layout(height=320, margin=dict(l=20, r=20, t=20, b=20))
            # register the filtered mean chart under the same id so expand shows the currently rendered chart
            CHART_REGISTRY['mean_sic2'] = fig2
            show_chart_with_expand(fig2, 'mean_sic2')

        with col1b:
            st.markdown('<p class="chart-title">Category Detail</p>', unsafe_allow_html=True)
            if len(rata_rata2_filtered) > 0:
                list_sic2 = sorted(rata_rata2_filtered['SIC_2digit'].astype(str).unique())
                selected_sic2 = st.selectbox("", options=list_sic2, key="tab2_radar", label_visibility="collapsed")
                fig_radar2 = plot_radar_chart(rata_rata2, id_col='SIC_2digit', selected_id=selected_sic2, desc_col='Description_2')
                fig_radar2.update_layout(height=320, margin=dict(l=20, r=20, t=20, b=20))
                # register radar for expand with deterministic id
                show_chart_with_expand(fig_radar2, f'radar_sic2_{selected_sic2}')

        # Row 2: Standard Deviation (big, full width) for SIC 2
        st.markdown('<p class="chart-title">Standard Deviation</p>', unsafe_allow_html=True)
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

        # Row 3: Similarity side-by-side for SIC 2
        st.markdown('<p class="chart-title">Similarity (Cosine vs Pearson)</p>', unsafe_allow_html=True)
        sim_col_cos2, sim_col_pear2 = st.columns(2)

        with sim_col_cos2:
            st.markdown('<div style="font-weight:700; margin-bottom:0.3rem;">Cosine Similarity</div>', unsafe_allow_html=True)
            cos_viz_choice2 = st.radio("", ["Dendrogram", "Heatmap"], horizontal=True, key="tab2_cos_viz", label_visibility="collapsed")
            if cos_viz_choice2 == "Dendrogram":
                fig_cos2 = plot_cosine_dendogram(cosine_sim2, id_col='SIC_2digit')
            else:
                fig_cos2 = plot_cosine_heatmap(cosine_sim2, id_col='SIC_2digit')
            fig_cos2.update_layout(height=340, margin=dict(l=20, r=20, t=20, b=20))
            show_chart_with_expand(fig_cos2, 'cos_sic2')

        with sim_col_pear2:
            st.markdown('<div style="font-weight:700; margin-bottom:0.3rem;">Pearson Similarity</div>', unsafe_allow_html=True)
            pear_viz_choice2 = st.radio("", ["Dendrogram", "Heatmap"], horizontal=True, key="tab2_pear_viz", label_visibility="collapsed")
            if pear_viz_choice2 == "Dendrogram":
                fig_pear2 = plot_cosine_dendogram(pearson2, id_col='SIC_2digit')
            else:
                fig_pear2 = plot_cosine_heatmap(pearson2, id_col='SIC_2digit')
            fig_pear2.update_layout(height=340, margin=dict(l=20, r=20, t=20, b=20))
            show_chart_with_expand(fig_pear2, 'pear_sic2')

        # G-Index Per Group
        st.markdown('<p class="chart-title">G-Index Per Group Analysis</p>', unsafe_allow_html=True)
        sic1_options = sorted(df_Gv2_sic1['SIC_1digit'].astype(str).unique())
        selected_sic1_g = st.selectbox("Select SIC 1 Digit Category:", options=sic1_options, key="tab2_g_select")

        df_sic1 = df_Gv2_sic1[df_Gv2_sic1['SIC_1digit'].astype(str) == selected_sic1_g]
        keep_cols = ['SIC_1digit'] + [c for c in TRAIT_COLS if c in df_sic1.columns]
        df_sic1_clean = df_sic1[keep_cols].copy()

        if len(df_sic1_clean) > 1:
            df_sic1_clean = df_sic1_clean.groupby('SIC_1digit', as_index=False).mean(numeric_only=True)

        num_sic2 = rata_rata2[rata_rata2['SIC_1digit'].astype(str) == selected_sic1_g]['SIC_2digit'].nunique()
        st.markdown(f'<p class="caption-text">✓ Processing {num_sic2} SIC 2 Digit categories</p>', unsafe_allow_html=True)

        if not df_sic1_clean.empty:
            fig_bar = plot_G_per_group_v2(df_sic1_clean, id_col='SIC_1digit')
            # create a deterministic chart id including the selected SIC1 so expand can render the same data
            chart_id_for_g2 = f'g_sic2_{selected_sic1_g}'
            chart_map[chart_id_for_g2] = fig_bar  # register for expand handling
            CHART_REGISTRY[chart_id_for_g2] = fig_bar
            show_chart_with_expand(fig_bar, chart_id_for_g2)
        else:
            st.info("No data available for this selection")

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
