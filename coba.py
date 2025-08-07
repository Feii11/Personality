import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
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

gabung = pd.merge(dfCEO, dfSIC, on='SIC_2digit', how='left')

rata_rata = gabung.groupby('SIC_1digit')[['agree', 'consc', 'extra', 'neuro', 'openn']].mean()

st.title("Personality CEO 10 Major Division")
rata_transpose = rata_rata.T

fig, ax = plt.subplots(figsize=(10, 6))

for kolom in rata_transpose.columns:
    ax.plot(rata_transpose.index, rata_transpose[kolom], marker='o', label=f'SIC {kolom}')

ax.set_ylim(1, 7)
ax.set_xlabel("Big Five Personality Traits")
ax.set_ylabel("Rata-rata Skor")
ax.legend(title='SIC_1digit', bbox_to_anchor=(1.05, 1), loc='upper left')
ax.grid(True)

# Tampilkan di Streamlit
st.pyplot(fig)

