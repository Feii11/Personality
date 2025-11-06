# app.py
import base64
import streamlit as st
import pandas as pd

from sic_utils import get_mean_sic1_v2, get_mean_sic2_v2, prepare_data, prepare_data_v2, build_hover_maps, compute_cosine_similarity, TRAIT_COLS
from plots import mean_line_chart, boxplot_from_means, errorbar_plot_from_means, plot_G_bar, plot_cosine_heatmap, TRAIT_LABELS
st.set_page_config(page_title="Personality CEO (Refactor)", layout="wide")

# -------------------------
# Cache data loading + computations
# -------------------------
@st.cache_data
def load_raw():
    dfCEO = pd.read_csv('data/Data.csv')
    dfSIC = pd.read_excel('data/2 digit.xlsx')
    return dfCEO, dfSIC

@st.cache_data
def prepare_and_compute():
    dfCEO, dfSIC = load_raw()

    prepare_data_v2(dfCEO, dfSIC)

    dfCEO_clean = pd.read_excel('dfCEO_clean.xlsx')
    
    get_mean_sic1_v2(dfCEO_clean, dfSIC)
    get_mean_sic2_v2(dfCEO_clean, dfSIC)
    compute_cosine_similarity(rata_rata1, 'SIC_1digit')

rata_rata1 = pd.read_excel('mean_sic1.xlsx')
rata_rata2 = pd.read_excel('mean_sic2.xlsx')
cosine_sim1 = pd.read_excel('cosine_similarity_SIC_1digit.xlsx')
# cosine_sim2 = pd.read_excel('cosine_sic2.xlsx', index_col=0)

# -------------------------
# UI Header
# -------------------------
st.markdown("## :bar_chart: Personality CEO")
st.markdown("---")
st.button("Load Data", on_click=prepare_and_compute)

# Load & encode filter icon (optional)
try:
    with open("images/filter_icon.jpg", "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    filter_html = f"""
    <div style="display:flex; align-items:center; gap:8px">
        <img src="data:image/jpeg;base64,{encoded}" width=36 height=36>
        <h4 style="margin:0">FILTER</h4>
    </div>
    """
except FileNotFoundError:
    filter_html = "<h4>FILTER</h4>"

# -------------------------
# Tabs
# -------------------------
tab1, tab2 , tab3, tab4 = st.tabs(["Rata-rata", "Standar Deviasi", "G(personality)", "Cosine Similarity"])

# Shared filter: SIC 1 digit checkbox list (sticky in left column of tab1)
with tab1:
    col_filter, col_chart = st.columns([1, 4], gap="large")

    # Filter
    with col_filter:
        st.markdown(filter_html, unsafe_allow_html=True)

        list_sic1 = sorted(rata_rata1['SIC_1digit'].astype(str).unique())
        
        if 'selected_sic' not in st.session_state:
            st.session_state.selected_sic = list_sic1.copy()

        selected = []
        for item in list_sic1:
            checked = st.checkbox(f"SIC {item}", value=(item in st.session_state.selected_sic))
            if checked:
                selected.append(item)
        st.session_state.selected_sic = selected

    # Chart
    with col_chart:
        selected_sic = st.session_state.selected_sic or list_sic1

        st.subheader("Rata-Rata Kepribadian CEO berdasarkan SIC digit 1")
        st.caption(f"{len(rata_rata1[rata_rata1['SIC_1digit'].isin(selected_sic)])} data diproses")
        
        fig1 = mean_line_chart(
            rata_rata1,
            id_col='SIC_1digit',
            desc_col='Description_1',
            selected_ids=selected_sic
        )
        st.plotly_chart(fig1, use_container_width=True)

        st.subheader("Rata-Rata Kepribadian CEO berdasarkan SIC digit 2")
        rata_rata2_filtered = rata_rata2[rata_rata2['SIC_1digit'].isin(selected_sic)]
        st.caption(f"{len(rata_rata2_filtered)} data diproses")

        fig2 = mean_line_chart(
            rata_rata2_filtered,
            id_col='SIC_2digit',
            desc_col='Description_2',
            selected_ids=None
        )
        st.plotly_chart(fig2, use_container_width=True)

with tab2:
    rr1_long = rata_rata1.melt(
        id_vars=['SIC_1digit', 'Description_1'],
        value_vars=TRAIT_COLS,
        var_name='Dimensi_Kepribadian',
        value_name='Rata_Rata'
    )
    rr1_long['Dimensi_Kepribadian'] = rr1_long['Dimensi_Kepribadian'].map(TRAIT_LABELS)

    # fig_box1 = boxplot_from_means(rr1_long, height=600)
    # st.plotly_chart(fig_box1, use_container_width=True)

    st.subheader("Rata-rata dan Standar Deviasi per Dimensi (SIC 1 Digit)")
    st.caption(f"{rata_rata1['SIC_1digit'].nunique()} data diproses")

    fig_err1 = errorbar_plot_from_means(rr1_long, title_suffix="(SIC 1 Digit)")
    st.plotly_chart(fig_err1, use_container_width=True)

    # ----------------------------------------------------------------
    # BAGIAN FILTER UNTUK SIC 2 DIGIT
    # ----------------------------------------------------------------
    st.subheader("Rata-Rata dan Standar Deviasi Kepribadian per Dimensi (SIC 2 Digit)")
    
    sic1_options = ['All'] + sorted(rata_rata2['SIC_1digit'].astype(str).unique())
    selected_sic1_filter = st.selectbox(
        "Pilih Kategori SIC 1 Digit",
        sic1_options,
        index=0,
        key="sic2_error_filter"
    )

    # Filter data berdasarkan pilihan user
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

    st.caption(f"{grouped['SIC_2digit'].nunique()} data diproses")

    # fig_box2 = boxplot_from_means(grouped_long, height=700)
    # st.plotly_chart(fig_box2, use_container_width=True)

    # Tambahkan grafik error bar yang bisa difilter
    fig_err2 = errorbar_plot_from_means(grouped_long, title_suffix=f"{title_suffix} (SIC 2 Digit)")
    st.plotly_chart(fig_err2, use_container_width=True)

with tab3:
    st.subheader("G(Personality) berdasarkan SIC 1 Digit")
    # df_G1 = compute_G_per_trait(dfCEO, 'SIC_1digit')
    # figG1 = plot_G_bar(df_G1, 'SIC_1digit')
    # st.plotly_chart(figG1, use_container_width=True)

    # st.subheader("G(Personality) berdasarkan SIC 2 Digit")
    # df_G2 = compute_G_per_trait(dfCEO, 'SIC_2digit')
    # figG2 = plot_G_bar(df_G2, 'SIC_2digit')
    # st.plotly_chart(figG2, use_container_width=True)

with tab4:  # atau tab baru
    st.subheader("Heatmap Cosine Similarity — SIC 1 Digit")
    fig_sim1 = plot_cosine_heatmap(cosine_sim1, id_col='SIC_1digit', title="Cosine Similarity antar SIC 1 Digit")
    st.plotly_chart(fig_sim1, use_container_width=True)

    st.subheader("Heatmap Cosine Similarity — SIC 2 Digit")
    sim2 = compute_cosine_similarity(rata_rata2, 'SIC_2digit', TRAIT_COLS)
    fig_sim2 = plot_cosine_heatmap(sim2, id_col='SIC_2digit', title="Cosine Similarity antar SIC 2 Digit")
    st.plotly_chart(fig_sim2, use_container_width=True)
