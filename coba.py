import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Superstore!!!", page_icon=":bar_chart:", layout="wide")

st.title(" :bar_chart: CEO Personality")
st.divider()
st.markdown('<style>div.block-container{padding-top:1rem;}</style>', unsafe_allow_html=True)

#Ambil Data
dataCEO = pd.read_csv('Data.csv')
dataSIC = pd.read_excel('2 digit.xlsx')

dfCEO = pd.DataFrame(dataCEO)
dfSIC = pd.DataFrame(dataSIC)

#Pastikan SICnya 4 digit dan ambil 2 digit paling depan
dfCEO['SIC'] = dfCEO['SIC'].astype(str).str.zfill(4)
dfCEO['SIC_2digit'] = dfCEO['SIC'].str[:2]

#Pastikan 2 digit dan ubah jadi string
dfSIC['SIC_2digit'] = dfSIC['SIC_2digit'].astype(str).str.zfill(2)

#Merge 2 tabel
gabung = pd.merge(dfCEO, dfSIC, on='SIC_2digit', how='left')

#Cari rata-rata tiap SIC 1 digit
rata_rata1 = gabung.groupby('SIC_1digit')[['agree', 'consc', 'extra', 'neuro', 'openn']].mean()
rata_rata1 = pd.merge(
    rata_rata1,
    #Munculin digit SIC dan descriptionnya
    dfSIC[['SIC_1digit', 'Description_1']].drop_duplicates(),
    on='SIC_1digit',
    how='left'
)

#Cari rata-rata tiap SIC 2 digit
rata_rata2 = gabung.groupby('SIC_2digit')[['agree', 'consc', 'extra', 'neuro', 'openn']].mean()
rata_rata2 = pd.merge(
    rata_rata2,
    #Munculin digit SIC dan descriptionnya
    dfSIC[['SIC_2digit', 'Description_2']].drop_duplicates(),
    on='SIC_2digit',
    how='left'
)

rata_rata_indexed1 = rata_rata1.set_index('SIC_1digit')
rata_rata_indexed2 = rata_rata2.set_index('SIC_2digit')

# Transpose matrixnya
rata_transpose1 = rata_rata_indexed1.T
rata_transpose2 = rata_rata_indexed2.T

# Title
st.title("Personality CEO 10 Major Division")


fig1 = go.Figure()
for kolom in rata_transpose1.columns:
    desc = rata_rata1.loc[rata_rata1['SIC_1digit'] == kolom, 'Description_1'].values[0]
    
    fig1.add_trace(go.Scatter(
        x=rata_transpose1.index,
        y=rata_transpose1.iloc[:-1][kolom],
        mode='lines+markers',
        name=f"SIC {kolom}",
        hovertemplate=(
            str(kolom) + '. ' + desc + '<br>' +
            'Dimensi: %{x}<br>' +
            'Skor: %{y:.2f}<extra></extra>'
        )
    ))

fig1.update_layout(
    title="Rata-rata Skor Kepribadian per SIC 1 Digit",
    xaxis_title="Dimensi Kepribadian",
    yaxis_title="Rata-rata Skor",
    yaxis=dict(range=[1, 7]),
    template="simple_white"
)

rata_transpose1 = rata_transpose1.iloc[:-1]   # buang baris pertama


fig2 = go.Figure()
for kolom in rata_transpose2.columns:
    desc = rata_rata2.loc[rata_rata2['SIC_2digit'] == kolom, 'Description_2'].values[0]

    fig2.add_trace(go.Scatter(
        x=rata_transpose2.index,
        y=rata_transpose2.iloc[:-1][kolom],
        mode='lines+markers',
        name=f"SIC {kolom}",
        hovertemplate=(
            str(kolom) + '. ' + desc + '<br>' +
            'Dimensi: %{x}<br>' +
            'Skor: %{y:.2f}<extra></extra>'
        )
    ))

fig2.update_layout(
    title="Rata-rata Skor Kepribadian per SIC 2 Digit",
    xaxis_title="Dimensi Kepribadian",
    yaxis_title="Rata-rata Skor",
    yaxis=dict(range=[1, 7]),
    template="simple_white"
)

col1, col2 = st.columns(2)

with col1:
    st.plotly_chart(fig1)
with col2:
    st.plotly_chart(fig2)