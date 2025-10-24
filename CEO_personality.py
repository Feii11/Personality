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

@st.cache_data
def load_data():
    dfCEO = pd.read_csv('data/Data.csv')
    dfSIC = pd.read_excel('data/2 digit.xlsx')
    dfCEO['SIC'] = dfCEO['SIC'].astype(str).str.zfill(4)
    dfCEO['SIC_2digit'] = dfCEO['SIC'].str[:2]
    dfSIC['SIC_2digit'] = dfSIC['SIC_2digit'].astype(str).str.zfill(2)
    return dfCEO, dfSIC

@st.cache_data
def compute_means(dfCEO, dfSIC):
    rata1 = SIC1.SIC_1digit(dfCEO, dfSIC)
    rata2 = SIC1.SIC_2digit(dfCEO, dfSIC)
    return rata1, rata2

dfCEO, dfSIC = load_data()
rata_rata1, rata_rata2 = compute_means(dfCEO, dfSIC)

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
tab1, tab2 = st.tabs(["Rata-rata", "Standar Deviasi"])

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
    # ===========================
            # BOX PLOT dari Rata-rata SIC 1 Digit (pakai function Mean.SIC_1digit)
            # ===========================
            st.subheader("Sebaran Rata-Rata Kepribadian per Dimensi (SIC 1 Digit)")

            # Ambil hasil rata-rata dari function yang sudah ada
            rata_rata1 = Mean.SIC_1digit(dfCEO, dfSIC)

            # Pastikan kolom personality sesuai
            traits = ['agree', 'consc', 'extra', 'neuro', 'openn']

            # Ubah ke long format agar bisa dipakai untuk boxplot
            rata_rata1_long = rata_rata1.melt(
                id_vars=['SIC_1digit', 'Description_1'],
                value_vars=traits,
                var_name='Dimensi_Kepribadian',
                value_name='Rata_Rata'
            )

            # Mapping label trait ke nama lengkap
            label_traits = {
                'agree': 'Agreeableness',
                'consc': 'Conscientiousness',
                'extra': 'Extraversion',
                'neuro': 'Neuroticism',
                'openn': 'Openness'
            }
            rata_rata1_long['Dimensi_Kepribadian'] = rata_rata1_long['Dimensi_Kepribadian'].map(label_traits)

            import plotly.express as px

            # Buat box plot dari rata-rata tiap SIC 1 digit
            fig_box_avg = px.box(
                rata_rata1_long,
                x='Dimensi_Kepribadian',
                y='Rata_Rata',
                points='all',  # tampilkan semua titik SIC 1 digit
                hover_data={
                    'SIC_1digit': True,
                    'Description_1': True
                },
                color='Dimensi_Kepribadian',
                title="Sebaran Rata-Rata Kepribadian per Dimensi (SIC 1 Digit)"
            )

            fig_box_avg.update_layout(
                xaxis_title="Dimensi Kepribadian",
                yaxis_title="Rata-rata Skor",
                template="simple_white",
                height=600,
                showlegend=False
            )

            st.plotly_chart(fig_box_avg, use_container_width=True)

            # Jumlah kategori SIC 1 digit yang diproses
            jumlah_sic1 = rata_rata1['SIC_1digit'].nunique()
            st.markdown(f"**Jumlah SIC 1 digit yang diproses:** {jumlah_sic1}")

            # ===========================
            # BOX PLOT RATA-RATA SIC 2 DIGIT (berdasarkan filter SIC 1 digit)
            # ===========================
            st.subheader("Sebaran Rata-Rata Kepribadian per Dimensi (SIC 2 Digit)")

            # Ambil data rata-rata per SIC 2 digit dari function
            rata_rata2 = Mean.SIC_2digit(dfCEO, dfSIC)

            # Gabungkan deskripsi 1 digit dari dfSIC biar bisa tampilkan "A - Mining"
            sic1_desc_map = dfSIC[['SIC_1digit', 'Description_1']].drop_duplicates()

            # Merge deskripsi 1 digit ke hasil rata-rata
            rata_rata2 = rata_rata2.merge(sic1_desc_map, on='SIC_1digit', how='left')

            # Ambil list SIC 1 digit untuk dropdown
            sic1_options = ['All'] + sorted(rata_rata2['SIC_1digit'].unique())
            selected_sic1_filter = st.selectbox("Pilih Kategori SIC 1 Digit", sic1_options, index=0, key="sic2_box_filter")

            # Cari deskripsi SIC 1 digit (misal A - Mining)
            if selected_sic1_filter != 'All':
                desc_sic1 = rata_rata2.loc[rata_rata2['SIC_1digit'] == selected_sic1_filter, 'Description_1'].iloc[0]
                title_suffix = f"{selected_sic1_filter} - {desc_sic1}"
            else:
                title_suffix = "Semua Kategori"

            # Jika user memilih satu SIC 1 digit, agregasikan data SIC 2 digit di bawahnya
            if selected_sic1_filter != 'All':
                rata_rata2_filtered = rata_rata2[rata_rata2['SIC_1digit'] == selected_sic1_filter]

                # Ambil rata-rata kepribadian per SIC 2 digit
                grouped = rata_rata2_filtered.groupby(
                    ['SIC_2digit', 'Description_2', 'SIC_1digit', 'Description_1'], as_index=False
                )[["agree", "consc", "extra", "neuro", "openn"]].mean()
            else:
                # Kalau 'All', pakai semua data rata-rata SIC 2 digit
                grouped = rata_rata2.copy()

            # Hitung jumlah SIC 2 digit yang diproses
            jumlah_sic2 = grouped['SIC_2digit'].nunique()
            st.markdown(f"**Jumlah SIC 2 digit yang diproses:** {jumlah_sic2}")

            # Ubah ke format long untuk boxplot
            traits = ['agree', 'consc', 'extra', 'neuro', 'openn']
            grouped_long = grouped.melt(
                id_vars=['SIC_2digit', 'SIC_1digit', 'Description_2'],
                value_vars=traits,
                var_name='Dimensi_Kepribadian',
                value_name='Rata_Rata'
            )

            # Mapping label ke nama lengkap
            label_traits = {
                'agree': 'Agreeableness',
                'consc': 'Conscientiousness',
                'extra': 'Extraversion',
                'neuro': 'Neuroticism',
                'openn': 'Openness'
            }
            grouped_long['Dimensi_Kepribadian'] = grouped_long['Dimensi_Kepribadian'].map(label_traits)

            import plotly.express as px

            # Buat box plot
            fig_box_avg2 = px.box(
                grouped_long,
                x='Dimensi_Kepribadian',
                y='Rata_Rata',
                points='all',
                hover_data={
                    'SIC_2digit': True,
                    'Description_2': True,
                    'SIC_1digit': True
                },
                color='Dimensi_Kepribadian',
                title=f"Sebaran Rata-Rata Kepribadian per Dimensi (SIC 2 Digit – {title_suffix})"
            )

            fig_box_avg2.update_layout(
                xaxis_title="Dimensi Kepribadian",
                yaxis_title="Rata-rata Skor",
                template="simple_white",
                height=700,
                showlegend=False
            )

            st.plotly_chart(fig_box_avg2, use_container_width=True)


