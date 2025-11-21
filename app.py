# app.py
import base64
import streamlit as st
import pandas as pd
from pathlib import Path

from sic_utils import get_mean_sic1, get_mean_sic2, prepare_data, compute_cosine_similarity, compute_pearson_similarity, compute_G_scores, compute_G_scores_v2, TRAIT_COLS
from plots import mean_line_chart, errorbar_plot_from_means, plot_cosine_heatmap, plot_cosine_dendogram, plot_G_bar, pie_chart, pie_chart_styled, TRAIT_LABELS, plot_G_per_group_v2, plot_radar_chart

st.set_page_config(page_title="Personality CEO Dashboard", layout="wide", initial_sidebar_state="collapsed")

# Load external CSS
def inject_css():
    css_path = Path(__file__).parent / "style.css"
    if css_path.exists():
        try:
            st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)
        except Exception as e:
            st.warning(f"CSS gagal dimuat: {e}")
    else:
        st.warning("File style.css tidak ditemukan.")

inject_css()

# -------------------------
# Cache data loading + computations
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
        st.error(f"Gagal membaca file: {e}")
        return None

@st.cache_data
def prepare_and_compute(upload):
    dfCEO = load_file(upload)
    dfSIC = pd.read_excel('data/2 digit.xlsx')
    dfCEO, dfSIC = prepare_data(dfCEO, dfSIC)

    rata_rata1 = get_mean_sic1(dfCEO, dfSIC)
    rata_rata2 = get_mean_sic2(dfCEO, dfSIC)
    
    # Compute overall mean across all CEOs
    overall_mean = dfCEO[TRAIT_COLS].mean().to_frame().T
    overall_mean.insert(0, 'SIC_1digit', 'All')
    overall_mean.insert(1, 'Description_1', 'All CEOs Combined')
    
    cosine_sim1 = compute_cosine_similarity(rata_rata1, 'SIC_1digit')
    cosine_sim2 = compute_cosine_similarity(rata_rata2, 'SIC_2digit')

    pearson1 = compute_pearson_similarity(rata_rata1, 'SIC_1digit')
    pearson2 = compute_pearson_similarity(rata_rata2, 'SIC_2digit')
    
    # Compute G scores (original aggregate per trait)
    df_G1 = compute_G_scores(rata_rata1, 'SIC_1digit')
    df_G2 = compute_G_scores(rata_rata2, 'SIC_2digit')
    # Compute per-group G_i(trait) scores (v2)
    df_Gv2_sic1 = compute_G_scores_v2(rata_rata2)

    return dfCEO, dfSIC, rata_rata1, rata_rata2, overall_mean, cosine_sim1, cosine_sim2, pearson1, pearson2, df_G1, df_G2, df_Gv2_sic1

# -------------------------
# Header
# -------------------------
st.markdown("""
<div class="main-header">
    <h1>📊 Dashboard Analisis Personality CEO</h1>
    <p>Analisis komprehensif kepribadian CEO berdasarkan klasifikasi SIC</p>
</div>
""", unsafe_allow_html=True)

upload = st.file_uploader("📁 Upload Data CEO (CSV / Excel)", type=["csv", "xls", "xlsx"])

if upload is not None:
    dfCEO, dfSIC, rata_rata1, rata_rata2, overall_mean, cosine_sim1, cosine_sim2, pearson1, pearson2, df_G1, df_G2, df_Gv2_sic1 = prepare_and_compute(upload)

    # -------------------------
    # OVERVIEW SECTION
    # -------------------------
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Data", len(dfCEO))
    with col2:
        st.metric("Total CEO", dfCEO['EXEC_FULLNAME'].nunique())
    with col3:
        st.metric("Kategori SIC 1 Digit", dfCEO['SIC_1digit'].nunique())
    with col4:
        st.metric("Kategori SIC 2 Digit", dfCEO['SIC_2digit'].nunique())
    with col5:
        st.metric("Dimensi Kepribadian", len(TRAIT_COLS))

    # -------------------------
    # DATA PREVIEW
    # -------------------------
    figpie = pie_chart_styled(dfCEO, column='SIC_1digit')
    st.plotly_chart(figpie, use_container_width=True)

    # -------------------------
    # FILTER SIDEBAR (in main content) - ONLY FOR MEAN ANALYSIS
    # -------------------------
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    col_filter, col_main = st.columns([1, 4])
    
    with col_filter:
        st.markdown('<div class="filter-box">', unsafe_allow_html=True)
        st.markdown("### 🎯 Filter")
        st.caption("Filter ini hanya berlaku untuk bagian Analisis Rata-Rata Kepribadian")
        
        list_sic1 = sorted(rata_rata1['SIC_1digit'].astype(str).unique())
        
        if 'selected_sic' not in st.session_state:
            st.session_state.selected_sic = list_sic1.copy()

        selected = []
        for item in list_sic1:
            checked = st.checkbox(f"SIC {item}", value=(item in st.session_state.selected_sic), key=f"filter_{item}")
            if checked:
                selected.append(item)
        st.session_state.selected_sic = selected
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_main:
        selected_sic = st.session_state.selected_sic or list_sic1
        
        # -------------------------
        # RATA-RATA SECTION (FILTERED)
        # -------------------------
        st.markdown('<div class="section-header"><h2>📈 Analisis Rata-Rata Kepribadian</h2></div>', unsafe_allow_html=True)
        
        st.markdown("#### SIC 1 Digit")
        st.caption(f"✓ {len(rata_rata1[rata_rata1['SIC_1digit'].isin(selected_sic)])} kategori dianalisis")
        
        # Create two columns: line chart and radar chart
        col_line1, col_radar1 = st.columns([3, 2])
        
        with col_line1:
            fig1 = mean_line_chart(rata_rata1, id_col='SIC_1digit', desc_col='Description_1', selected_ids=selected_sic)
            st.plotly_chart(fig1, use_container_width=True)
        
        with col_radar1:
            st.markdown("##### 🎯 Radar Chart Detail")
            # Add "All" option to show overall mean
            list_sic1_all = ['All'] + sorted(rata_rata1['SIC_1digit'].astype(str).unique())
            selected_sic1_radar = st.selectbox(
                "Pilih SIC 1 Digit:",
                options=list_sic1_all,
                key="radar_sic1_select"
            )
            
            # Show radar chart based on selection
            if selected_sic1_radar == 'All':
                fig_radar1 = plot_radar_chart(overall_mean, id_col='SIC_1digit', selected_id='All', desc_col='Description_1')
            else:
                fig_radar1 = plot_radar_chart(rata_rata1, id_col='SIC_1digit', selected_id=selected_sic1_radar, desc_col='Description_1')
            st.plotly_chart(fig_radar1, use_container_width=True)
        
        st.markdown("#### SIC 2 Digit")
        
        # Filter SIC 2 digit based on selected SIC 1 digit from radar dropdown
        if selected_sic1_radar == 'All':
            rata_rata2_filtered = rata_rata2.copy()
            st.caption(f"✓ {len(rata_rata2_filtered)} kategori dianalisis (All SIC 1 Digit)")
        else:
            rata_rata2_filtered = rata_rata2[rata_rata2['SIC_1digit'].astype(str) == selected_sic1_radar]
            st.caption(f"✓ {len(rata_rata2_filtered)} kategori dianalisis (SIC 1 Digit: {selected_sic1_radar})")
        
        # Create two columns: line chart and radar chart
        col_line2, col_radar2 = st.columns([3, 2])
        
        with col_line2:
            fig2 = mean_line_chart(rata_rata2_filtered, id_col='SIC_2digit', desc_col='Description_2', selected_ids=None)
            st.plotly_chart(fig2, use_container_width=True)
        
        with col_radar2:
            st.markdown("##### 🎯 Radar Chart Detail")
            # Only show SIC 2 digit options from filtered data
            if len(rata_rata2_filtered) > 0:
                list_sic2_filtered = sorted(rata_rata2_filtered['SIC_2digit'].astype(str).unique())
                selected_sic2_radar = st.selectbox(
                    "Pilih SIC 2 Digit:",
                    options=list_sic2_filtered,
                    key="radar_sic2_select"
                )
                fig_radar2 = plot_radar_chart(rata_rata2, id_col='SIC_2digit', selected_id=selected_sic2_radar, desc_col='Description_2')
                st.plotly_chart(fig_radar2, use_container_width=True)
            else:
                st.info("Tidak ada data SIC 2 Digit untuk kategori ini.")
    
    # -------------------------
    # STANDAR DEVIASI SECTION (NOT FILTERED - FULL WIDTH)
    # -------------------------
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-header"><h2>📊 Standar Deviasi & Error Analysis</h2></div>', unsafe_allow_html=True)
    
    rr1_long = rata_rata1.melt(
        id_vars=['SIC_1digit', 'Description_1'],
        value_vars=TRAIT_COLS,
        var_name='Dimensi_Kepribadian',
        value_name='Rata_Rata'
    )
    rr1_long['Dimensi_Kepribadian'] = rr1_long['Dimensi_Kepribadian'].map(TRAIT_LABELS)
    
    st.markdown("#### SIC 1 Digit - Rata-rata dan Standar Deviasi")
    st.caption(f"✓ {rata_rata1['SIC_1digit'].nunique()} kategori diproses (semua data)")
    fig_err1 = errorbar_plot_from_means(rr1_long, title_suffix="(SIC 1 Digit)")
    st.plotly_chart(fig_err1, use_container_width=True)
    
    st.markdown("#### SIC 2 Digit - Rata-rata dan Standar Deviasi")
    sic1_options = ['All'] + sorted(rata_rata2['SIC_1digit'].astype(str).unique())
    selected_sic1_filter = st.selectbox(
        "🔍 Filter berdasarkan SIC 1 Digit:",
        sic1_options,
        index=0,
        key="sic2_error_filter"
    )
    
    if selected_sic1_filter == 'All':
        grouped = rata_rata2.copy()
        title_suffix = "Semua Kategori"
    else:
        grouped = rata_rata2[rata_rata2['SIC_1digit'] == selected_sic1_filter]
        title_suffix = f"SIC {selected_sic1_filter}"
    
    grouped_long = grouped.melt(
        id_vars=['SIC_2digit', 'SIC_1digit', 'Description_2'],
        value_vars=TRAIT_COLS,
        var_name='Dimensi_Kepribadian',
        value_name='Rata_Rata'
    )
    grouped_long['Dimensi_Kepribadian'] = grouped_long['Dimensi_Kepribadian'].map(TRAIT_LABELS)
    
    st.caption(f"✓ {grouped['SIC_2digit'].nunique()} kategori diproses")
    fig_err2 = errorbar_plot_from_means(grouped_long, title_suffix=f"{title_suffix} (SIC 2 Digit)")
    st.plotly_chart(fig_err2, use_container_width=True)
    
    # -------------------------
    # SIMILARITY ANALYSIS (NOT FILTERED - FULL WIDTH WITH INTERACTIVE SELECTION)
    # -------------------------
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-header"><h2>🔗 Analisis Similaritas</h2></div>', unsafe_allow_html=True)
    
    # Pilihan visualisasi untuk similarity
    viz_option = st.radio(
        "📊 Pilih jenis visualisasi:",
        ["Dendrogram", "Heatmap"],
        index=0,
        horizontal=True,
        key="sim_viz_option"
    )
    
    st.markdown("### SIC 1 Digit")
    col_sim1, col_sim2 = st.columns(2)
    
    with col_sim1:
        st.markdown("#### Cosine Similarity")
        if viz_option == "Dendrogram":
            fig_cos1_dend = plot_cosine_dendogram(cosine_sim1, id_col='SIC_1digit')
            st.plotly_chart(fig_cos1_dend, use_container_width=True)
        else:
            fig_cos1_heat = plot_cosine_heatmap(cosine_sim1, id_col='SIC_1digit')
            st.plotly_chart(fig_cos1_heat, use_container_width=True)
    
    with col_sim2:
        st.markdown("#### Pearson Correlation")
        if viz_option == "Dendrogram":
            fig_pear1_dend = plot_cosine_dendogram(pearson1, id_col='SIC_1digit')
            st.plotly_chart(fig_pear1_dend, use_container_width=True)
        else:
            fig_pear1 = plot_cosine_heatmap(pearson1, id_col='SIC_1digit')
            st.plotly_chart(fig_pear1, use_container_width=True)
    
    st.markdown("---")
    
    st.markdown("### SIC 2 Digit")
    col_sim3, col_sim4 = st.columns(2)
    
    with col_sim3:
        st.markdown("#### Cosine Similarity")
        if viz_option == "Dendrogram":
            fig_cos2_dend = plot_cosine_dendogram(cosine_sim2, id_col='SIC_2digit')
            st.plotly_chart(fig_cos2_dend, use_container_width=True)
        else:
            fig_cos2_heat = plot_cosine_heatmap(cosine_sim2, id_col='SIC_2digit')
            st.plotly_chart(fig_cos2_heat, use_container_width=True)
    
    with col_sim4:
        st.markdown("#### Pearson Correlation")
        if viz_option == "Dendrogram":
            fig_pear2_dend = plot_cosine_dendogram(pearson2, id_col='SIC_2digit')
            st.plotly_chart(fig_pear2_dend, use_container_width=True)
        else:
            fig_pear2 = plot_cosine_heatmap(pearson2, id_col='SIC_2digit')
            st.plotly_chart(fig_pear2, use_container_width=True)
    
    # -------------------------
    # G-INDEX ANALYSIS (NOT FILTERED - FULL WIDTH)
    # -------------------------
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-header"><h2>📊 G-Index Analysis (Trait Distribution)</h2></div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <p><strong>G-Index (Gini Coefficient)</strong> mengukur ketidakmerataan distribusi nilai personality trait antar kelompok SIC.</p>
        <p>• G = 0: Distribusi sempurna merata (semua kelompok memiliki nilai yang sama)</p>
        <p>• G = 1: Distribusi sangat tidak merata (ketimpangan maksimal)</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("#### G-Index - SIC 1 Digit")
    st.caption(f"✓ Menganalisis distribusi trait pada {rata_rata1['SIC_1digit'].nunique()} kategori")
    fig_g1 = plot_G_bar(df_G1, title_suffix="(SIC 1 Digit)")
    st.plotly_chart(fig_g1, use_container_width=True)

    # -------------------------
    # G-SCORE PER GROUP (V2) - NOT FILTERED - FULL WIDTH
    # -------------------------
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-header"><h2>🧮 G_i(Trait) per SIC 1 Digit (Versi 2)</h2></div>', unsafe_allow_html=True)
    st.caption("Menampilkan besaran ketidakmerataan tiap kelompok terhadap semua kelompok lain untuk setiap trait.")

    # Pilih satu SIC_1digit untuk detail bar chart
    sic1_options = sorted(df_Gv2_sic1['SIC_1digit'].astype(str).unique())
    selected_sic1 = st.selectbox(
        "Pilih SIC 1 Digit untuk melihat bar chart:",
        options=sic1_options,
        key="g_v2_bar_select"
    )

    # Filter hanya baris yang dipilih
    df_sic1 = df_Gv2_sic1[df_Gv2_sic1['SIC_1digit'].astype(str) == selected_sic1]

    # Hanya simpan kolom yang diperlukan untuk bar chart: id + trait columns
    keep_cols = ['SIC_1digit'] + [c for c in TRAIT_COLS if c in df_sic1.columns]
    df_sic1_clean = df_sic1[keep_cols].copy()

    # Jika ada lebih dari satu baris (duplikasi), agregasi dengan mean
    if len(df_sic1_clean) > 1:
        df_sic1_clean = df_sic1_clean.groupby('SIC_1digit', as_index=False).mean(numeric_only=True)

    # Count how many SIC_2digit in this SIC_1digit group
    num_sic2 = rata_rata2[rata_rata2['SIC_1digit'].astype(str) == selected_sic1]['SIC_2digit'].nunique()

    st.markdown(f"##### G_i(Trait) untuk SIC {selected_sic1}")
    st.caption(f"✓ {num_sic2} kategori SIC 2 Digit diproses")

    # Tampilkan bar chart yang hanya berisi trait yang relevan
    if not df_sic1_clean.empty:
        fig_bar = plot_G_per_group_v2(df_sic1_clean, id_col='SIC_1digit')
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("Data tidak tersedia untuk pilihan ini.")
    
    # Footer
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <p>Dashboard Analisis Personality CEO | Data diproses secara real-time</p>
    </div>
    """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="info-box">
        <h3>👋 Selamat Datang!</h3>
        <p>Silakan upload file data CEO (format CSV atau Excel) untuk memulai analisis komprehensif kepribadian CEO berdasarkan klasifikasi SIC.</p>
    </div>
    """, unsafe_allow_html=True)