import pandas as pd

def SIC_1digit(dfCEO, dfSIC):
    # Merge data
    gabung = pd.merge(dfCEO, dfSIC, on='SIC_2digit', how='left')

    # Cari rata-rata
    rata_rata1 = gabung.groupby('SIC_1digit')[['agree','consc','extra','neuro','openn']].mean().reset_index()

    # Tambah description buat hover
    rata_rata1 = pd.merge(rata_rata1, dfSIC[['SIC_1digit','Description_1']].drop_duplicates(), on='SIC_1digit', how='left')
    
    # Return
    return rata_rata1


def SIC_2digit(dfCEO, dfSIC):
    # Merge data
    gabung = pd.merge(dfCEO, dfSIC, on='SIC_2digit', how='left')

    # Cari rata-rata
    rata_rata2 = gabung.groupby('SIC_2digit')[['agree','consc','extra','neuro','openn']].mean().reset_index()

    # Tambah desccription buat hover
    rata_rata2 = pd.merge(rata_rata2, dfSIC[['SIC_2digit','SIC_1digit','Description_2']].drop_duplicates(), on='SIC_2digit', how='left')

    # Return
    return rata_rata2