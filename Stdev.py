import pandas as pd

def SIC_1digit(dfCEO, dfSIC):
    # Merge data
    gabung = pd.merge(dfCEO, dfSIC, on='SIC_2digit', how='left')

    std_1 = gabung.groupby('SIC_1digit')[['agree','consc','extra','neuro','openn']].std().mean(axis=1).reset_index()
    std_1.columns = ['SIC_1digit', 'Std_Dev_Gabungan']
    std_1 = pd.merge(std_1, dfSIC[['SIC_1digit','Description_1']].drop_duplicates(), on='SIC_1digit', how='left')

    return std_1