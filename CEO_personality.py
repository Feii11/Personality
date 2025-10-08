import base64
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import warnings
import SIC1
import Mean
warnings.filterwarnings('ignore')

# Tab
st.set_page_config(page_title="Personality CEO", page_icon=":bar_chart:", layout="wide")

# Ubah jadi dataframe
dfCEO = pd.DataFrame(pd.read_csv('data/Data.csv'))
dfSIC = pd.DataFrame(pd.read_excel('data/2 digit.xlsx'))

# Tittle
st.markdown("## :bar_chart: Personality CEO")
st.markdown("---")

# Load data
dataCEO = pd.read_csv('data/Data.csv')
dataSIC = pd.read_excel('data/2 digit.xlsx')

dfCEO = pd.DataFrame(dataCEO)
dfSIC = pd.DataFrame(dataSIC)

# Data Cleaning
dfCEO['SIC'] = dfCEO['SIC'].astype(str).str.zfill(4)
dfCEO['SIC_2digit'] = dfCEO['SIC'].str[:2]
dfSIC['SIC_2digit'] = dfSIC['SIC_2digit'].astype(str).str.zfill(2)

# Merge data
gabung = pd.merge(dfCEO, dfSIC, on='SIC_2digit', how='left')

# Cari rata-rata
rata_rata1 = gabung.groupby('SIC_1digit')[['agree','consc','extra','neuro','openn']].mean().reset_index()
rata_rata1 = pd.merge(rata_rata1, dfSIC[['SIC_1digit','Description_1']].drop_duplicates(), on='SIC_1digit', how='left')

rata_rata2 = gabung.groupby('SIC_2digit')[['agree','consc','extra','neuro','openn']].mean().reset_index()
rata_rata2 = pd.merge(rata_rata2, dfSIC[['SIC_2digit','SIC_1digit','Description_2']].drop_duplicates(), on='SIC_2digit', how='left')

# Layout
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "Data"

def switch_tab(tab_name):
    st.session_state.active_tab = tab_name

# Pilihan tab
tab1, tab2 = st.tabs(["📈 Rata-rata", "📊 Standar Deviasi"])

# Session untuk tab
active_tab = st.session_state.active_tab

st.divider()

# Tab 1 - Rata-rata SIC
with tab1:
    # Bagi kolom untuk filter
    col_filter, col_chart = st.columns([1, 4], gap="large")

    # Icon filter
    file_path = "images/filter_icon.jpg"

    with open(file_path, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()

    # Filter
    with col_filter:
        st.markdown(
        f"""
        <div style="display: flex; align-items: center; gap: 10px;">
            <img src="data:image/png;base64,{encoded}"
            width="40" height="40">
            <h4 style="margin: 0;">FILTER</h4>
        </div>
        """,
        unsafe_allow_html=True
        )

        # Ambil list SIC 1 digit
        list_sic1 = sorted(rata_rata1['SIC_1digit'].unique())

        # State awal
        if "selected_sic" not in st.session_state:
            st.session_state.selected_sic = list_sic1.copy()
        if "check_all" not in st.session_state:
            st.session_state.check_all = True

        selected = []
        
        for item in list_sic1:
            checked = st.checkbox(f"SIC {item}", value=item in st.session_state.selected_sic)
            if checked:
                selected.append(item)
        st.session_state.selected_sic = selected

    # Chart
    with col_chart:
        selected_sic = st.session_state.selected_sic

        # Ambil data rata-rata
        rata_rata1 = Mean.SIC_1digit(dfCEO, dfSIC)

        # Connect dengan filter
        rata_rata1_filtered = rata_rata1[rata_rata1['SIC_1digit'].isin(selected_sic)]
        rata_transpose1 = rata_rata1_filtered.set_index('SIC_1digit')[['agree','consc','extra','neuro','openn']].T

        # Rata-rata SIC 1 digit
        fig1 = go.Figure()
        for kolom in rata_transpose1.columns:
            desc = rata_rata1_filtered.loc[rata_rata1_filtered['SIC_1digit'] == kolom, 'Description_1'].values[0]
            fig1.add_trace(go.Scatter(
                x=rata_transpose1.index,
                y=rata_transpose1[kolom],
                mode='lines+markers',
                name=f"SIC {kolom}",
                hovertemplate=f"{kolom}. {desc}<br>Dimensi: %{{x}}<br>Skor: %{{y:.2f}}<extra></extra>"
            ))

        fig1.update_layout(
            title="Rata-Rata Skor Kepribadian per SIC 1 digit",
            xaxis_title="Dimensi Kepribadian",
            yaxis_title="Rata-rata Skor",
            yaxis=dict(range=[1,7]),
            template="simple_white"
        )
        st.plotly_chart(fig1, use_container_width=True)

        # Ambil data rata-rata
        rata_rata2 = Mean.SIC_2digit(dfCEO, dfSIC)
        
        # Connect dengan filter
        rata_rata2_filtered = rata_rata2[rata_rata2['SIC_1digit'].isin(selected_sic)]
        rata_transpose2 = rata_rata2_filtered.set_index('SIC_2digit')[['agree','consc','extra','neuro','openn']].T

        # Rata-rata SIC 2 digit
        fig2 = go.Figure()
        for kolom in rata_transpose2.columns:
            desc = rata_rata2_filtered.loc[rata_rata2_filtered['SIC_2digit'] == kolom, 'Description_2'].values[0]
            fig2.add_trace(go.Scatter(
                x=rata_transpose2.index,
                y=rata_transpose2[kolom],
                mode='lines+markers',
                name=f"SIC {kolom}",
                hovertemplate=f"{kolom}. {desc}<br>Dimensi: %{{x}}<br>Skor: %{{y:.2f}}<extra></extra>"
            ))

        fig2.update_layout(
            title="Rata-Rata Skor Kepribadian per SIC 2 digit",
            xaxis_title="Dimensi Kepribadian",
            yaxis_title="Rata-rata Skor",
            yaxis=dict(range=[1,7]),
            template="simple_white"
        )
        st.plotly_chart(fig2, use_container_width=True)

# Tab 2 - Standar Deviasi
with tab2:
    # Bagi kolom untuk filter dan chart
    col_filter, col_table = st.columns([1, 4], gap="large")

    # ICON FILTER
    file_path = "images/filter_icon.jpg"
    with open(file_path, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()

    with col_filter:
        st.markdown(
        f"""
        <div style="display: flex; align-items: center; gap: 10px;">
            <img src="data:image/png;base64,{encoded}" width="40" height="40">
            <h4 style="margin: 0;">FILTER</h4>
        </div>
        """,
        unsafe_allow_html=True
        )

        # Filter personality traits
        list_traits = ['agree', 'consc', 'extra', 'neuro', 'openn']
        label_traits = {
            'agree': 'Agreeableness',
            'consc': 'Conscientiousness',
            'extra': 'Extraversion',
            'neuro': 'Neuroticism',
            'openn': 'Openness'
        }

        st.markdown("#### Dimensi Kepribadian")
        if "selected_traits" not in st.session_state:
            st.session_state.selected_traits = list_traits.copy()

        selected_traits = []
        for t in list_traits:
            checked = st.checkbox(label_traits[t], value=t in st.session_state.selected_traits, key=f"traits_{t}")
            if checked:
                selected_traits.append(t)
        st.session_state.selected_traits = selected_traits

        st.markdown("---")
        
        # Filter SIC 1 digit
        list_sic1 = sorted(gabung['SIC_1digit'].unique())
        st.markdown("#### Kategori SIC 1 Digit")
        if "selected_sic_tab2" not in st.session_state:
            st.session_state.selected_sic_tab2 = list_sic1.copy()

        selected_sic_tab2 = []
        for item in list_sic1:
            checked = st.checkbox(f"SIC {item}", value=item in st.session_state.selected_sic_tab2, key=f"sic_tab2_{item}")
            if checked:
                selected_sic_tab2.append(item)
        st.session_state.selected_sic_tab2 = selected_sic_tab2


    # Kolom utama
    with col_table:
        selected_traits = st.session_state.selected_traits
        selected_sic_tab2 = st.session_state.selected_sic_tab2

        if len(selected_traits) == 0:
            st.warning("Pilih minimal satu dimensi kepribadian untuk ditampilkan.")
        else:
            # ===========================
            # Hitung rata-rata personality per CEO
            # ===========================
            gabung['mean_personality'] = gabung[selected_traits].mean(axis=1)

            # Filter berdasarkan SIC 1 digit
            gabung_filtered = gabung[gabung['SIC_1digit'].isin(selected_sic_tab2)]

            # ===========================
            # BOX PLOT
            # ===========================
            st.subheader("📦 Sebaran Skor Kepribadian Rata-rata per SIC 1 Digit (Box Plot)")

            import plotly.express as px
            fig_box = px.box(
                gabung_filtered,
                x='SIC_1digit',
                y='mean_personality',
                color='SIC_1digit',
                title="Sebaran Skor Rata-rata Kepribadian Berdasarkan SIC 1 Digit",
                points='all'
            )

            fig_box.update_xaxes(categoryorder='category ascending')

            fig_box.update_layout(
                xaxis_title="SIC 1 Digit",
                yaxis_title="Rata-rata Dimensi Kepribadian",
                showlegend=False,
                template="simple_white",
                height=500
            )

            st.plotly_chart(fig_box, use_container_width=True)

            # ===========================
            # STANDAR DEVIASI (Lollipop Chart)
            # ===========================
            st.subheader("🍭 Standar Deviasi Gabungan per SIC 2 Digit (Lollipop Chart)")

            # Hitung std dev berdasarkan personality yang dipilih
            std_2 = (
                gabung_filtered
                .groupby(['SIC_2digit', 'SIC_1digit'])[selected_traits]
                .std()
                .mean(axis=1)
                .reset_index(name='Std_Dev_Gabungan')
            )

            # Tambahkan deskripsi
            std_2 = pd.merge(
                std_2,
                dfSIC[['SIC_2digit', 'SIC_1digit', 'Description_2']].drop_duplicates(),
                on=['SIC_2digit', 'SIC_1digit'],
                how='left'
            )

            # Terapkan kembali filter SIC 1 digit
            std_2 = std_2[std_2['SIC_1digit'].isin(selected_sic_tab2)]

            # Urutkan dari standar deviasi terbesar ke terkecil
            std_2_sorted = std_2.sort_values(by='Std_Dev_Gabungan', ascending=False)

            # Buat lollipop chart
            fig_lollipop = go.Figure()

            # Garis vertikal
            fig_lollipop.add_trace(go.Scatter(
                x=std_2_sorted['Std_Dev_Gabungan'],
                y=std_2_sorted['SIC_2digit'],
                mode='lines',
                line=dict(color='lightgray', width=2),
                showlegend=False
            ))

            # Titik
            fig_lollipop.add_trace(go.Scatter(
                x=std_2_sorted['Std_Dev_Gabungan'],
                y=std_2_sorted['SIC_2digit'],
                mode='markers',
                marker=dict(color='#FF8C00', size=10),
                hovertemplate=(
                    "<b>SIC %{y}</b><br>" +
                    "SIC 1 Digit: %{customdata[0]}<br>" +
                    "Deskripsi: %{customdata[1]}<br>" +
                    "Std Dev: %{x:.3f}<extra></extra>"
                ),
                customdata=std_2_sorted[['SIC_1digit', 'Description_2']].values,
                showlegend=False
            ))

            # Layout
            fig_lollipop.update_layout(
                xaxis_title="Standar Deviasi Gabungan",
                yaxis_title="SIC 2 Digit",
                template="simple_white",
                height=900,
                margin=dict(l=100, r=50, t=50, b=50),
            )

            st.plotly_chart(fig_lollipop, use_container_width=True)