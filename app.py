# app.py
import base64
import streamlit as st
import pandas as pd

from sic_utils import get_mean_sic1, get_mean_sic2, prepare_data, compute_cosine_similarity, compute_pearson_similarity, compute_G_scores, TRAIT_COLS
from plots import mean_line_chart, errorbar_plot_from_means, plot_cosine_heatmap, plot_cosine_dendogram, plot_G_bar, pie_chart, pie_chart_styled, TRAIT_LABELS

st.set_page_config(page_title="Personality CEO Dashboard", layout="wide", initial_sidebar_state="collapsed")

# Custom CSS untuk styling
st.markdown("""
<style>
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 2.5rem;
    }
    .main-header p {
        color: rgba(255,255,255,0.9);
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
    }
    
    /* Section headers */
    .section-header {
        background: linear-gradient(90deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem 1.5rem;
        border-radius: 8px;
        margin: 2rem 0 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .section-header h2 {
        color: white;
        margin: 0;
        font-size: 1.8rem;
    }
    
    /* Filter box */
    .filter-box {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Metrics */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border-top: 3px solid #667eea;
    }
    
    /* Info box */
    .info-box {
        background: #e3f2fd;
        padding: 1rem;
        border-radius: 6px;
        border-left: 3px solid #2196f3;
        margin: 1rem 0;
    }
    
    /* Divider */
    .divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #667eea, transparent);
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

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
    
    cosine_sim1 = compute_cosine_similarity(rata_rata1, 'SIC_1digit')
    cosine_sim2 = compute_cosine_similarity(rata_rata2, 'SIC_2digit')

    pearson1 = compute_pearson_similarity(rata_rata1, 'SIC_1digit')
    pearson2 = compute_pearson_similarity(rata_rata2, 'SIC_2digit')
    
    # Compute G scores
    df_G1 = compute_G_scores(rata_rata1, 'SIC_1digit')
    df_G2 = compute_G_scores(rata_rata2, 'SIC_2digit')

    return dfCEO, dfSIC, rata_rata1, rata_rata2, cosine_sim1, cosine_sim2, pearson1, pearson2, df_G1, df_G2

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
    dfCEO, dfSIC, rata_rata1, rata_rata2, cosine_sim1, cosine_sim2, pearson1, pearson2, df_G1, df_G2 = prepare_and_compute(upload)

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
    st.markdown('<div class="section-header"><h2>📋 Preview Data</h2></div>', unsafe_allow_html=True)
    
    col_prev1, col_prev2 = st.columns([2, 1])
    with col_prev1:
        st.dataframe(dfCEO.head(10), use_container_width=True)
    with col_prev2:
        figpie = pie_chart_styled(dfCEO, column='SIC_1digit')
        st.plotly_chart(figpie, use_container_width=True)

    # -------------------------
    # FILTER SIDEBAR (in main content)
    # -------------------------
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    col_filter, col_main = st.columns([1, 4])
    
    with col_filter:
        st.markdown('<div class="filter-box">', unsafe_allow_html=True)
        st.markdown("### 🎯 Filter")
        
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
        # RATA-RATA SECTION
        # -------------------------
        st.markdown('<div class="section-header"><h2>📈 Analisis Rata-Rata Kepribadian</h2></div>', unsafe_allow_html=True)
        
        st.markdown("#### SIC 1 Digit")
        st.caption(f"✓ {len(rata_rata1[rata_rata1['SIC_1digit'].isin(selected_sic)])} kategori dianalisis")
        fig1 = mean_line_chart(rata_rata1, id_col='SIC_1digit', desc_col='Description_1', selected_ids=selected_sic)
        st.plotly_chart(fig1, use_container_width=True)
        
        st.markdown("#### SIC 2 Digit")
        rata_rata2_filtered = rata_rata2[rata_rata2['SIC_1digit'].isin(selected_sic)]
        st.caption(f"✓ {len(rata_rata2_filtered)} kategori dianalisis")
        fig2 = mean_line_chart(rata_rata2_filtered, id_col='SIC_2digit', desc_col='Description_2', selected_ids=None)
        st.plotly_chart(fig2, use_container_width=True)
        
        # -------------------------
        # STANDAR DEVIASI SECTION
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
        st.caption(f"✓ {rata_rata1['SIC_1digit'].nunique()} kategori diproses")
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
        # SIMILARITY ANALYSIS
        # -------------------------
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-header"><h2>🔗 Analisis Similaritas</h2></div>', unsafe_allow_html=True)
        
        col_sim1, col_sim2 = st.columns(2)
        
        with col_sim1:
            st.markdown("#### Cosine Similarity - SIC 1 Digit")
            fig_cos1_heat = plot_cosine_heatmap(cosine_sim1, id_col='SIC_1digit')
            st.plotly_chart(fig_cos1_heat, use_container_width=True)
            
            fig_cos1_dend = plot_cosine_dendogram(cosine_sim1, id_col='SIC_1digit')
            st.plotly_chart(fig_cos1_dend, use_container_width=True)
        
        with col_sim2:
            st.markdown("#### Pearson Similarity - SIC 1 Digit")
            fig_pear1 = plot_cosine_heatmap(pearson1, id_col='SIC_1digit')
            st.plotly_chart(fig_pear1, use_container_width=True)
        
        st.markdown("---")
        
        col_sim3, col_sim4 = st.columns(2)
        
        with col_sim3:
            st.markdown("#### Cosine Similarity - SIC 2 Digit")
            fig_cos2_heat = plot_cosine_heatmap(cosine_sim2, id_col='SIC_2digit')
            st.plotly_chart(fig_cos2_heat, use_container_width=True)
            
            fig_cos2_dend = plot_cosine_dendogram(cosine_sim2, id_col='SIC_2digit')
            st.plotly_chart(fig_cos2_dend, use_container_width=True)
        
        with col_sim4:
            st.markdown("#### Pearson Similarity - SIC 2 Digit")
            fig_pear2 = plot_cosine_heatmap(pearson2, id_col='SIC_2digit')
            st.plotly_chart(fig_pear2, use_container_width=True)
        
        # -------------------------
        # G-INDEX ANALYSIS
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
        
        col_g1, col_g2 = st.columns(2)
        
        with col_g1:
            st.markdown("#### G-Index - SIC 1 Digit")
            st.caption(f"✓ Menganalisis distribusi trait pada {rata_rata1['SIC_1digit'].nunique()} kategori")
            fig_g1 = plot_G_bar(df_G1, title_suffix="(SIC 1 Digit)")
            st.plotly_chart(fig_g1, use_container_width=True)
        
        with col_g2:
            st.markdown("#### G-Index - SIC 2 Digit")
            st.caption(f"✓ Menganalisis distribusi trait pada {rata_rata2['SIC_2digit'].nunique()} kategori")
            fig_g2 = plot_G_bar(df_G2, title_suffix="(SIC 2 Digit)")
            st.plotly_chart(fig_g2, use_container_width=True)

        # -------------------------
        # G-INDEX PER SIC 1 DIGIT (menurunkan ke level 2-digit)
        # -------------------------
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-header"><h2>📊 G-Index per SIC 1 Digit (berdasarkan kelompok 2 Digit)</h2></div>', unsafe_allow_html=True)

        unique_sic1 = sorted(rata_rata2['SIC_1digit'].astype(str).unique())
        choice = st.selectbox(
            "Pilih SIC 1 Digit untuk melihat G-Index (Trait) dihitung dari kelompok SIC 2 Digit di dalamnya:",
            options=['All'] + unique_sic1,
            index=0,
            key="gindex_sic1_picker"
        )

        def render_g_for_sic1(sic1_value: str):
            sub = rata_rata2[rata_rata2['SIC_1digit'].astype(str) == sic1_value]
            if sub.empty:
                st.info(f"Tidak ada data untuk SIC 1 Digit {sic1_value}")
                return
            df_g = compute_G_scores(sub, 'SIC_2digit')
            fig = plot_G_bar(df_g, title_suffix=f"(SIC 1 Digit {sic1_value})")
            st.plotly_chart(fig, use_container_width=True)

        if choice == 'All':
            # Tampilkan semua (sekitar 10 chart) dalam grid
            cols = st.columns(2)
            for idx, sic1 in enumerate(unique_sic1):
                with cols[idx % 2]:
                    st.markdown(f"##### SIC 1 Digit {sic1}")
                    render_g_for_sic1(sic1)
        else:
            render_g_for_sic1(choice)

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