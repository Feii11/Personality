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
    dfSIC[['SIC_1digit', 'Description_1']].drop_duplicates(),
    on='SIC_1digit',
    how='left'
)

#Cari rata-rata tiap SIC 2 digit
rata_rata2 = gabung.groupby('SIC_2digit')[['agree', 'consc', 'extra', 'neuro', 'openn']].mean()

rata_transpose1 = rata_rata1.T
rata_transpose2 = rata_rata2.T

st.title("Personality CEO 10 Major Division")

fig1 = go.Figure()
for kolom in rata_transpose1.columns:
    fig1.add_trace(go.Scatter(
        x=rata_transpose1.index,
        y=rata_transpose1[kolom],
        mode='lines+markers',
        name=f'SIC {kolom}',
        hovertemplate='SIC ' + str(kolom) + '<br>Dimensi: %{x}<br>Skor: %{y:.2f}<extra></extra>'
    ))

fig1.update_layout(
    title="Rata-rata Skor Kepribadian per SIC 1 Digit",
    xaxis_title="Dimensi Kepribadian",
    yaxis_title="Rata-rata Skor",
    yaxis=dict(range=[1, 7]),
    template="simple_white"
)

st.plotly_chart(fig1, use_container_width=True)

fig1, ax1 = plt.subplots(figsize=(6, 5))
fig2, ax2 = plt.subplots(figsize=(6, 5))

for kolom in rata_transpose1.columns:
    ax1.plot(rata_transpose1.index, rata_transpose1[kolom], marker='o', label=f'SIC {kolom}')
ax1.set_ylim(1, 7)
ax1.set_xlabel("Big Five Personality Traits")
ax1.set_ylabel("Rata-rata Skor")
ax1.legend(title='SIC_1digit', bbox_to_anchor=(1.05, 1), loc='upper left')
ax1.grid(True)

for kolom in rata_transpose2.columns:
    ax2.plot(rata_transpose2.index, rata_transpose2[kolom], marker='o', label=f'SIC {kolom}')
ax2.set_ylim(1, 7)
ax2.set_xlabel("Big Five Personality Traits")
ax2.set_ylabel("Rata-rata Skor")
ax2.legend(title='SIC_1digit', bbox_to_anchor=(1.05, 1), loc='upper left')
ax2.grid(True)

col1, col2 = st.columns(2)

with col1:
    st.subheader("SIC 1 Digit")
    st.pyplot(fig1)
with col2:
    st.subheader("SIC 2 Digit")
    st.pyplot(fig2)