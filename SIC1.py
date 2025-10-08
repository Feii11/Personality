import pandas as pd

def listSIC1(dfCEO, dfSIC):
    # Data Cleaning
    dfCEO['SIC'] = dfCEO['SIC'].astype(str).str.zfill(4)
    dfCEO['SIC_2digit'] = dfCEO['SIC'].str[:2]
    dfSIC['SIC_2digit'] = dfSIC['SIC_2digit'].astype(str).str.zfill(2)

    # Merge data
    gabung = pd.merge(dfCEO, dfSIC, on='SIC_2digit', how='left')

    return sorted(gabung['SIC_1digit'].unique())