# sic_utils.py (replace existing)
import pandas as pd
import numpy as np
from itertools import combinations
from typing import List
import os
from pathlib import Path

TRAIT_COLS = ['agree', 'consc', 'extra', 'neuro', 'openn']

def prepare_data(dfCEO: pd.DataFrame, dfSIC: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Standardisasi kolom SIC, tambahkan kolom SIC_2digit di dfCEO dan dfSIC.
    Tidak mengubah kolom SIC_1digit karena sudah tersedia di dfSIC.
    Mengembalikan (dfCEO_clean, dfSIC_clean).
    """
    dfCEO = dfCEO.copy()
    dfSIC = dfSIC.copy()
    
    # ---------- dfCEO ----------
    if 'SIC' in dfCEO.columns:
        dfCEO['SIC'] = dfCEO['SIC'].astype(str).str.zfill(4)
        dfCEO['SIC_2digit'] = dfCEO['SIC'].str[:2]

    if 'SIC_2digit' in dfCEO.columns:
        dfCEO['SIC_2digit'] = dfCEO['SIC_2digit'].astype(str).str.zfill(2)
    else:
        raise ValueError("dfCEO tidak memiliki kolom 'SIC' atau 'SIC_2digit'")

    # ---------- Samakan tipe data dulu ----------
    dfSIC['SIC_2digit'] = dfSIC['SIC_2digit'].astype(str).str.zfill(2)

    # ---------- Mapping SIC_1digit dari dfSIC ----------
    if 'SIC_2digit' in dfSIC.columns and 'SIC_1digit' in dfSIC.columns:
        mapping = dfSIC.drop_duplicates(subset=['SIC_2digit'])[['SIC_2digit', 'SIC_1digit']]
        dfCEO = dfCEO.merge(mapping, on='SIC_2digit', how='left')
    else:
        raise ValueError("dfSIC harus memiliki kolom 'SIC_2digit' dan 'SIC_1digit' untuk pemetaan.")

    # ---------- dfSIC ----------
    if 'SIC' in dfSIC.columns:
        dfSIC['SIC'] = dfSIC['SIC'].astype(str).str.zfill(4)
        dfSIC['SIC_2digit'] = dfSIC['SIC'].str[:2]

    if 'SIC_2digit' in dfSIC.columns:
        dfSIC['SIC_2digit'] = dfSIC['SIC_2digit'].astype(str).str.zfill(2)
    else:
        raise ValueError("Kolom 'SIC_2digit' tidak ditemukan di dfSIC")
    
    return dfCEO, dfSIC

def prepare_data_v2(dfCEO: pd.DataFrame, dfSIC: pd.DataFrame) -> None:
    dfCEO = pd.read_csv('data/Data.csv')
    dfSIC = pd.read_excel('data/2 digit.xlsx')

    dfCEO = dfCEO.copy()
    dfSIC = dfSIC.copy()

    # ---------- dfCEO ----------
    if 'SIC' in dfCEO.columns:
        dfCEO['SIC'] = dfCEO['SIC'].astype(str).str.zfill(4)
        dfCEO['SIC_2digit'] = dfCEO['SIC'].str[:2]

    if 'SIC_2digit' in dfCEO.columns:
        dfCEO['SIC_2digit'] = dfCEO['SIC_2digit'].astype(str).str.zfill(2)
    else:
        raise ValueError("dfCEO tidak memiliki kolom 'SIC' atau 'SIC_2digit'")

    # ---------- Samakan tipe data dulu ----------
    dfSIC['SIC_2digit'] = dfSIC['SIC_2digit'].astype(str).str.zfill(2)

    # ---------- Mapping SIC_1digit dari dfSIC ----------
    if 'SIC_2digit' in dfSIC.columns and 'SIC_1digit' in dfSIC.columns:
        mapping = dfSIC.drop_duplicates(subset=['SIC_2digit'])[['SIC_2digit', 'SIC_1digit']]
        dfCEO = dfCEO.merge(mapping, on='SIC_2digit', how='left')
    else:
        raise ValueError("dfSIC harus memiliki kolom 'SIC_2digit' dan 'SIC_1digit' untuk pemetaan.")
    # ---------- dfSIC ----------
    if 'SIC' in dfSIC.columns:
        dfSIC['SIC'] = dfSIC['SIC'].astype(str).str.zfill(4)
        dfSIC['SIC_2digit'] = dfSIC['SIC'].str[:2]
    if 'SIC_2digit' in dfSIC.columns:
        dfSIC['SIC_2digit'] = dfSIC['SIC_2digit'].astype(str).str.zfill(2)
    else:
        raise ValueError("Kolom 'SIC_2digit' tidak ditemukan di dfSIC")

    # Tentukan folder output
    try:
        out_dir = Path(__file__).resolve().parent
    except NameError:
        out_dir = Path(os.path.abspath(""))  # fallback for Jupyter

    path_ceo = out_dir / 'dfCEO_clean.xlsx'

    try:
        dfCEO.to_excel(path_ceo, index=False, engine='openpyxl')
    except Exception as e:
        raise IOError(f"Gagal menyimpan Excel ke '{out_dir}': {e}")

    # tidak mengembalikan apa-apa (files tersimpan)

def get_mean_sic1(dfCEO: pd.DataFrame, dfSIC: pd.DataFrame) -> pd.DataFrame:
    """
    Hitung rata-rata skor personality per SIC_1digit.
    Tambahkan Description_1 dari dfSIC jika ada.
    Kembalikan DataFrame hasil, tidak menyimpan ke file.
    """
    dfCEO = dfCEO.copy()
    dfSIC = dfSIC.copy()

    # Pastikan format kolom kunci seragam
    dfCEO['SIC_1digit'] = dfCEO['SIC_1digit'].astype(str).str.strip()
    dfSIC['SIC_1digit'] = dfSIC['SIC_1digit'].astype(str).str.strip()

    # Hitung rata-rata berdasarkan SIC_1digit
    rata1 = dfCEO.groupby('SIC_1digit')[TRAIT_COLS].mean(numeric_only=True).reset_index()

    # Gabungkan dengan deskripsi (jika ada di dfSIC)
    if 'Description_1' in dfSIC.columns:
        desc = dfSIC[['SIC_1digit', 'Description_1']].drop_duplicates()
        rata1 = rata1.merge(desc, on='SIC_1digit', how='left')
    else:
        # fallback kalau description belum ada
        rata1['Description_1'] = rata1['SIC_1digit']

    return rata1

def get_mean_sic1_v2(dfCEO: pd.DataFrame, dfSIC: pd.DataFrame) -> None:
    """
    Hitung rata-rata skor personality per SIC_1digit.
    Tambahkan Description_1 dari dfSIC jika ada.
    Alih-alih mengembalikan DataFrame, fungsi ini menyimpan hasil ke file Excel.
    Jika output_path None -> simpan 'mean_sic1.xlsx' di folder modul.
    """

    dfCEO = dfCEO.copy()
    dfSIC = dfSIC.copy()

    # Pastikan format kolom kunci seragam
    dfCEO['SIC_1digit'] = dfCEO['SIC_1digit'].astype(str).str.strip()
    dfSIC['SIC_1digit'] = dfSIC['SIC_1digit'].astype(str).str.strip()

    # Hitung rata-rata berdasarkan SIC_1digit
    rata1 = dfCEO.groupby('SIC_1digit')[TRAIT_COLS].mean(numeric_only=True).reset_index()

    # Gabungkan dengan deskripsi (jika ada di dfSIC)
    if 'Description_1' in dfSIC.columns:
        desc = dfSIC[['SIC_1digit', 'Description_1']].drop_duplicates()
        rata1 = rata1.merge(desc, on='SIC_1digit', how='left')
    else:
        # fallback kalau description belum ada
        rata1['Description_1'] = rata1['SIC_1digit']

    try:
        out_dir = Path(__file__).resolve().parent
    except NameError:
        out_dir = Path(os.path.abspath(""))  # fallback for Jupyter

    # Pastikan kita menulis ke file, bukan direktori
    out_path = out_dir / 'mean_sic1.xlsx'

    try:
        rata1.to_excel(out_path, index=False, engine='openpyxl')
    except Exception as e:
        raise IOError(f"Gagal menyimpan hasil ke Excel di '{out_path}': {e}")

    # tidak mengembalikan apa-apa (file tersimpan)

def get_mean_sic2(dfCEO: pd.DataFrame, dfSIC: pd.DataFrame) -> pd.DataFrame:
    """
    Hitung rata-rata skor personality per SIC_2digit.
    Kembalikan dataframe: SIC_2digit + TRAIT_COLS + SIC_1digit + Description_2 (jika ada) + Description_1 (dari dfSIC).
    """
    dfCEO = dfCEO.copy()
    dfSIC = dfSIC.copy()

    # Pastikan format kunci sama
    dfCEO['SIC_2digit'] = dfCEO['SIC_2digit'].astype(str).str.strip().str.zfill(2)
    dfSIC['SIC_2digit'] = dfSIC['SIC_2digit'].astype(str).str.strip().str.zfill(2)

    # Hitung rata-rata berdasarkan SIC_2digit
    rata2 = dfCEO.groupby('SIC_2digit')[TRAIT_COLS].mean(numeric_only=True).reset_index()

    # Tambahkan kolom SIC_1digit dari dfCEO (lebih konsisten dibanding hasil merge)
    extra = dfCEO[['SIC_2digit', 'SIC_1digit']].drop_duplicates()

    # Jika dfSIC punya Description_2, tambahkan juga
    if 'Description_2' in dfSIC.columns:
        desc = dfSIC[['SIC_2digit', 'Description_2']].drop_duplicates()
        extra = extra.merge(desc, on='SIC_2digit', how='left')

    # Tambahkan Description_1 dari dfSIC berdasarkan SIC_1digit
    if 'Description_1' in dfSIC.columns:
        desc1 = dfSIC[['SIC_1digit', 'Description_1']].drop_duplicates()
        extra = extra.merge(desc1, on='SIC_1digit', how='left')

    # Gabungkan tambahan kolom dengan hasil rata-rata
    rata2 = rata2.merge(extra, on='SIC_2digit', how='left')

    return rata2

def get_mean_sic2_v2(dfCEO: pd.DataFrame, dfSIC: pd.DataFrame) -> None:
    """
    Hitung rata-rata skor personality per SIC_2digit.
    Kembalikan dataframe: SIC_2digit + TRAIT_COLS + SIC_1digit + Description_2 (jika ada).
    """
    dfCEO = dfCEO.copy()
    dfSIC = dfSIC.copy()

    # Pastikan format kunci sama
    dfCEO['SIC_2digit'] = dfCEO['SIC_2digit'].astype(str).str.strip().str.zfill(2)
    dfSIC['SIC_2digit'] = dfSIC['SIC_2digit'].astype(str).str.strip().str.zfill(2)

    # Hitung rata-rata berdasarkan SIC_2digit
    rata2 = dfCEO.groupby('SIC_2digit')[TRAIT_COLS].mean(numeric_only=True).reset_index()

    # Tambahkan kolom SIC_1digit dari dfCEO (lebih konsisten dibanding hasil merge)
    extra = dfCEO[['SIC_2digit', 'SIC_1digit']].drop_duplicates()

    # Jika dfSIC punya Description_2, tambahkan juga
    if 'Description_2' in dfSIC.columns:
        desc = dfSIC[['SIC_2digit', 'Description_2']].drop_duplicates()
        extra = extra.merge(desc, on='SIC_2digit', how='left')

    # Gabungkan tambahan kolom dengan hasil rata-rata
    rata2 = rata2.merge(extra, on='SIC_2digit', how='left')

    try:
        out_dir = Path(__file__).resolve().parent
    except NameError:
        out_dir = Path(os.path.abspath(""))  # fallback for Jupyter

    # Pastikan kita menulis ke file, bukan direktori
    out_path = out_dir / 'mean_sic2.xlsx'

    try:
        rata2.to_excel(out_path, index=False, engine='openpyxl')
    except Exception as e:
        raise IOError(f"Gagal menyimpan hasil ke Excel di '{out_path}': {e}")    

def build_hover_maps(dfSIC: pd.DataFrame) -> tuple[dict, dict]:
    """
    Kembalikan dua dict: hover_map_1 (SIC_1digit -> Description_1) dan hover_map_2 (SIC_2digit -> Description_2)
    """
    hover_map_1 = {}
    hover_map_2 = {}

    if 'SIC_1digit' in dfSIC.columns and 'Description_1' in dfSIC.columns:
        hover_map_1 = dict(dfSIC[['SIC_1digit', 'Description_1']].drop_duplicates().values)

    if 'SIC_2digit' in dfSIC.columns and 'Description_2' in dfSIC.columns:
        hover_map_2 = dict(dfSIC[['SIC_2digit', 'Description_2']].drop_duplicates().values)

    return hover_map_1, hover_map_2

import pandas as pd

def prepare_long_format(df, id_col, desc_col):
    """
    Mengubah dataframe rata-rata ke format long:
    dari kolom agree, consc, extra, neuro, openn → baris Dimensi_Kepribadian dan Rata_Rata
    """
    long_df = df.melt(
        id_vars=[id_col, desc_col],
        value_vars=['agree', 'consc', 'extra', 'neuro', 'openn'],
        var_name='Dimensi_Kepribadian',
        value_name='Rata_Rata'
    )

    # Ganti label jadi versi panjang (sesuai plots.py)
    label_map = {
        'agree': 'Agreeableness',
        'consc': 'Conscientiousness',
        'extra': 'Extraversion',
        'neuro': 'Neuroticism',
        'openn': 'Openness'
    }
    long_df['Dimensi_Kepribadian'] = long_df['Dimensi_Kepribadian'].map(label_map)
    return long_df

TRAIT_COLS = ['agree', 'consc', 'extra', 'neuro', 'openn']

def compute_G_scores(df_mean: pd.DataFrame, id_col: str) -> pd.DataFrame:
    """
    Hitung G-index untuk setiap trait personality sebagai total selisih absolut
    dari rata-rata antar kelompok SIC.
    
    Parameters:
    -----------
    df_mean : pd.DataFrame
        DataFrame berisi rata-rata personality traits per kelompok
    id_col : str
        Nama kolom untuk pengelompokan ('SIC_1digit' atau 'SIC_2digit')
    
    Returns:
    --------
    pd.DataFrame
        DataFrame dengan kolom: 'Personality Trait' dan 'G(Trait)'
    """
    G_values = {}
    
    for trait in TRAIT_COLS:
        if trait not in df_mean.columns:
            continue
        
        # Ambil nilai rata-rata trait untuk semua kelompok
        values = df_mean[trait].dropna().astype(float).values
        
        # Hitung total selisih absolut antar semua pasangan kelompok
        total_diff = 0.0
        for i in range(len(values)):
            for j in range(i + 1, len(values)):
                total_diff += abs(values[i] - values[j])
        
        G_values[trait] = total_diff
    
    # Convert to DataFrame format untuk plotting
    df_G = pd.DataFrame({
        'Personality Trait': list(G_values.keys()),
        'G(Trait)': list(G_values.values())
    })

    return df_G
import numpy as np
import pandas as pd

TRAIT_COLS = ["agree", "consc", "extra", "neuro", "openn"]

import numpy as np
import pandas as pd

import numpy as np
import pandas as pd

def compute_G_scores_v2(
    df_mean: pd.DataFrame,
    sic2_col: str = "SIC_2digit",
    sic1_col: str = "SIC_1digit",
    desc_col: str = "Description_2",
    trait_cols: list = None,
    na_action: str = "raise"  # "raise", "drop", atau "fill0"
) -> pd.DataFrame:
    """
    Hitung G(trait) per SIC_1digit dari rata-rata trait per SIC_2digit.

    Matematika:
      Untuk tiap group (SIC_1digit) dengan k unique SIC_2digit dan nilai trait x_1..x_k:
        G_sum = sum_{1 <= i < j <= k} |x_i - x_j|
      Jika k <= 1, G = 0.

    Output hanya mencakup:
      SIC_1digit, Description_1, dan <trait_cols>.
    """
    if trait_cols is None:
        trait_cols = ["agree", "consc", "extra", "neuro", "openn"]

    # cek kolom
    required = [sic2_col, sic1_col] + trait_cols
    missing = [c for c in required if c not in df_mean.columns]
    if missing:
        raise ValueError(f"Missing columns in df_mean: {missing}")

    df = df_mean.copy()

    # normalize keys
    df[sic1_col] = df[sic1_col].astype(str).str.strip()
    df[sic2_col] = df[sic2_col].astype(str).str.strip()
    if desc_col in df.columns:
        df[desc_col] = df[desc_col].astype(str).str.strip()

    # convert trait columns to numeric
    for t in trait_cols:
        df[t] = pd.to_numeric(df[t].astype(str).str.replace(",", "."), errors="coerce")

    results = []
    for sic1, grp in df.groupby(sic1_col, sort=True):
        # representative description
        desc_val = grp[desc_col].iloc[0] if desc_col in grp.columns else ""

        # drop duplicate SIC_2digit
        sub = grp.drop_duplicates(subset=[sic2_col]).reset_index(drop=True)
        k = sub.shape[0]

        # NaN handling
        if sub[trait_cols].isna().any().any():
            if na_action == "raise":
                raise ValueError(f"NaN found in traits for SIC_1digit={sic1}")
            elif na_action == "drop":
                sub = sub.dropna(subset=trait_cols).reset_index(drop=True)
                k = sub.shape[0]
            elif na_action == "fill0":
                sub[trait_cols] = sub[trait_cols].fillna(0)
            else:
                raise ValueError("na_action must be 'raise','drop','fill0'")

        row = {
            "SIC_1digit": sic1,
            "Description_1": desc_val
        }

        for t in trait_cols:
            vals = sub[t].astype(float).to_numpy()

            if len(vals) <= 1:
                row[t] = 0.0
            else:
                diff = np.abs(vals[:, None] - vals[None, :])
                G_sum = float(np.triu(diff, k=1).sum())
                row[t] = G_sum

        results.append(row)

    res_df = pd.DataFrame(results).sort_values("SIC_1digit").reset_index(drop=True)
    return res_df


def compute_cosine_similarity(df_mean: pd.DataFrame, id_col: str) -> pd.DataFrame:
    """
    Hitung cosine similarity antar kelompok (SIC 1 digit atau 2 digit)
    berdasarkan nilai rata-rata personality traits.
    Kembalikan DataFrame similarity (index & columns = id_col).
    Tidak membuat file.
    """
    df = df_mean.copy()
    if id_col not in df.columns:
        raise ValueError(f"Kolom '{id_col}' tidak ditemukan di DataFrame")
    missing_traits = [t for t in TRAIT_COLS if t not in df.columns]
    if missing_traits:
        raise ValueError(f"Trait columns missing: {missing_traits}")

    df = df[[id_col] + TRAIT_COLS].dropna(subset=TRAIT_COLS)
    df[id_col] = df[id_col].astype(str)

    ids = df[id_col].values
    X = df[TRAIT_COLS].fillna(0).to_numpy(dtype=float)

    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    X_norm = X / norms

    sim_matrix = np.dot(X_norm, X_norm.T)
    sim_matrix = np.clip(sim_matrix, -1.0, 1.0)

    sim_df = pd.DataFrame(sim_matrix, index=ids, columns=ids)
    return sim_df

def compute_cosine_similarity_v2(df_mean: pd.DataFrame, id_col: str) -> None:
    """
    Hitung cosine similarity antar kelompok (SIC 1 digit atau 2 digit) 
    berdasarkan nilai rata-rata personality traits.
    Fungsi tidak mengembalikan apa-apa, melainkan menyimpan hasil ke Excel
    (file: cosine_similarity_<id_col>.xlsx di folder modul).
    """
    df = df_mean.copy()
    if id_col not in df.columns:
        raise ValueError(f"Kolom '{id_col}' tidak ditemukan di DataFrame")
    missing_traits = [t for t in TRAIT_COLS if t not in df.columns]
    if missing_traits:
        raise ValueError(f"Trait columns missing: {missing_traits}")

    df = df[[id_col] + TRAIT_COLS].dropna(subset=TRAIT_COLS)
    df[id_col] = df[id_col].astype(str)

    ids = df[id_col].values
    X = df[TRAIT_COLS].fillna(0).to_numpy(dtype=float)

    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    X_norm = X / norms

    sim_matrix = np.dot(X_norm, X_norm.T)
    sim_matrix = np.clip(sim_matrix, -1.0, 1.0)

    sim_df = pd.DataFrame(sim_matrix, index=ids, columns=ids)

    try:
        out_dir = Path(__file__).resolve().parent
    except NameError:
        out_dir = Path(os.path.abspath(""))

    safe_id = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in id_col)
    out_path = out_dir / f"cosine_similarity_{safe_id}.xlsx"

    try:
        # Export without writing the DataFrame index as a separate column
        sim_df.to_excel(out_path, index=False, engine='openpyxl')
    except Exception as e:
        raise IOError(f"Gagal menyimpan cosine similarity ke '{out_path}': {e}")
    
def compute_pearson_similarity(df_mean: pd.DataFrame, id_col: str) -> pd.DataFrame:
    """
    Hitung Pearson correlation similarity untuk traits antar kelompok SIC 
    berdasarkan DataFrame rata-rata traits.

    Parameters:
    -----------
    df_mean : pd.DataFrame 
        DataFrame berisi rata-rata personality traits per SIC
    id_col : str
        Nama kolom untuk pengelompokan ('SIC_1digit' atau 'SIC_2digit')
    
    Returns:
    --------
    pd.DataFrame
        Matrix korelasi Pearson antar kelompok SIC
    """
    df = df_mean.copy()
    
    # Validasi input
    if id_col not in df.columns:
        raise ValueError(f"Kolom '{id_col}' tidak ditemukan di DataFrame")
    missing_traits = [t for t in TRAIT_COLS if t not in df.columns]
    if missing_traits:
        raise ValueError(f"Trait columns missing: {missing_traits}")

    # Set index ke id_col untuk memudahkan pivoting    
    df = df.set_index(id_col)[TRAIT_COLS]
    
    # Hitung korelasi Pearson antar kelompok
    pearson_matrix = df.T.corr(method='pearson')
    
    return pearson_matrix