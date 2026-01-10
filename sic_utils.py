# sic_utils.py (replace existing)
import pandas as pd
import numpy as np
from itertools import combinations
from typing import List, Optional
import os
from pathlib import Path
from scipy.stats import f_oneway

TRAIT_COLS = ['agree', 'consc', 'extra', 'neuro', 'openn']

def prepare_data(dfCEO: pd.DataFrame, dfSIC: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
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

    # Pastikan tipe data SIC_2digit adalah string
    dfSIC['SIC_2digit'] = dfSIC['SIC_2digit'].astype(str).str.zfill(2)

    # Mapping SIC_1digit dari dfSIC
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
    
    # Filter dfCEO untuk hanya baris dengan SIC_1digit valid
    if 'SIC_1digit' in dfCEO.columns:
        non_na = dfCEO['SIC_1digit'].notna()
        dfCEO.loc[non_na, 'SIC_1digit'] = dfCEO.loc[non_na, 'SIC_1digit'].astype(str).str.strip()
        mask_ceo = non_na & (dfCEO['SIC_1digit'] != "")
        dfCEO = dfCEO[mask_ceo].reset_index(drop=True)
    else:
        raise ValueError("dfCEO tidak memiliki kolom 'SIC_1digit' setelah pemetaan.")
    
    # Pastikan personality traits 1-7 dan numeric
    missing_traits = [t for t in TRAIT_COLS if t not in dfCEO.columns]
    if missing_traits:
        raise ValueError(f"Kolom trait tidak ditemukan di dfCEO: {missing_traits}")

    for t in TRAIT_COLS:
        dfCEO[t] = pd.to_numeric(dfCEO[t].astype(str).str.replace(",", "."), errors="coerce")

    valid_mask = dfCEO[TRAIT_COLS].apply(lambda col: col.between(1, 7)).all(axis=1)
    dfCEO = dfCEO[valid_mask].reset_index(drop=True)

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
        out_dir = Path(os.path.abspath(""))

    path_ceo = out_dir / 'dfCEO_clean.xlsx'

    try:
        dfCEO.to_excel(path_ceo, index=False, engine='openpyxl')
    except Exception as e:
        raise IOError(f"Gagal menyimpan Excel ke '{out_dir}': {e}")

def get_mean_sic1(dfCEO: pd.DataFrame, dfSIC: pd.DataFrame) -> pd.DataFrame:
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
    hover_map_1 = {}
    hover_map_2 = {}

    if 'SIC_1digit' in dfSIC.columns and 'Description_1' in dfSIC.columns:
        hover_map_1 = dict(dfSIC[['SIC_1digit', 'Description_1']].drop_duplicates().values)

    if 'SIC_2digit' in dfSIC.columns and 'Description_2' in dfSIC.columns:
        hover_map_2 = dict(dfSIC[['SIC_2digit', 'Description_2']].drop_duplicates().values)

    return hover_map_1, hover_map_2

import pandas as pd

def prepare_long_format(df, id_col, desc_col):
    long_df = df.melt(
        id_vars=[id_col, desc_col],
        value_vars=['agree', 'consc', 'extra', 'neuro', 'openn'],
        var_name='Dimensi_Kepribadian',
        value_name='Rata_Rata'
    )

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
    G_values = {}
    
    for trait in TRAIT_COLS:
        if trait not in df_mean.columns:
            continue
        
        values = df_mean[trait].dropna().astype(float).values
        
        total_diff = 0.0
        for i in range(len(values)):
            for j in range(i + 1, len(values)):
                total_diff += abs(values[i] - values[j])
        
        G_values[trait] = total_diff

    df_G = pd.DataFrame({
        'Personality Trait': list(G_values.keys()),
        'G(Trait)': list(G_values.values())
    })

    return df_G

def compute_G_scores_pairs(df_mean: pd.DataFrame, id_col: str) -> pd.DataFrame:
    if id_col not in df_mean.columns:
        raise ValueError(f"Kolom '{id_col}' tidak ditemukan di DataFrame")

    missing_traits = [t for t in TRAIT_COLS if t not in df_mean.columns]
    if missing_traits:
        raise ValueError(f"Trait columns missing: {missing_traits}")

    label_map = {
        'agree': 'Agreeableness',
        'consc': 'Conscientiousness',
        'extra': 'Extraversion',
        'neuro': 'Neuroticism',
        'openn': 'Openness'
    }

    results = []
    for t1, t2 in combinations(TRAIT_COLS, 2):
        sub = df_mean[[id_col, t1, t2]].copy()
        # normalisasi angka (toleransi koma) dan konversi ke numeric
        sub[t1] = pd.to_numeric(sub[t1].astype(str).str.replace(",", "."), errors="coerce")
        sub[t2] = pd.to_numeric(sub[t2].astype(str).str.replace(",", "."), errors="coerce")
        sub = sub.dropna(subset=[t1, t2]).reset_index(drop=True)

        vals = sub[[t1, t2]].to_numpy(dtype=float)
        if vals.shape[0] <= 1:
            G_val = 0.0
        else:
            diff = vals[:, None, :] - vals[None, :, :]  # shape (n, n, 2)
            dist = np.linalg.norm(diff, axis=2)        # pairwise euclidean distances
            G_val = float(np.triu(dist, k=1).sum())

        pair_label = f"{label_map.get(t1, t1)} & {label_map.get(t2, t2)}"
        results.append({"Personality Pair": pair_label, "G(Pair)": G_val})

    res_df = pd.DataFrame(results).sort_values("Personality Pair").reset_index(drop=True)
    return res_df

def compute_G_scores_v2(
    df_mean: pd.DataFrame,
    sic2_col: str = "SIC_2digit",
    sic1_col: str = "SIC_1digit",
    desc_col: str = "Description_2",
    trait_cols: list = None,
    na_action: str = "raise",  # "raise", "drop", atau "fill0"
    save: bool = True,
    output_path: Optional[str] = None
) -> pd.DataFrame:
    if trait_cols is None:
        trait_cols = ["agree", "consc", "extra", "neuro", "openn"]

    required = [sic2_col, sic1_col] + trait_cols
    missing = [c for c in required if c not in df_mean.columns]
    if missing:
        raise ValueError(f"Missing columns in df_mean: {missing}")

    df = df_mean.copy()

    df[sic1_col] = df[sic1_col].astype(str).str.strip()
    df[sic2_col] = df[sic2_col].astype(str).str.strip()
    if desc_col in df.columns:
        df[desc_col] = df[desc_col].astype(str).str.strip()

    for t in trait_cols:
        df[t] = pd.to_numeric(df[t].astype(str).str.replace(",", "."), errors="coerce")

    results = []
    for sic1, grp in df.groupby(sic1_col, sort=True):
        desc_val = grp[desc_col].iloc[0] if desc_col in grp.columns else ""

        sub = grp.drop_duplicates(subset=[sic2_col]).reset_index(drop=True)

        if sub[trait_cols].isna().any().any():
            if na_action == "raise":
                raise ValueError(f"NaN found in traits for SIC_1digit={sic1}")
            elif na_action == "drop":
                sub = sub.dropna(subset=trait_cols).reset_index(drop=True)
            elif na_action == "fill0":
                sub[trait_cols] = sub[trait_cols].fillna(0)
            else:
                raise ValueError("na_action must be 'raise','drop','fill0'")

        n_sic2 = int(sub.shape[0])

        row = {
            sic1_col: sic1,
            "Description_1": desc_val,
            "N_SIC2": n_sic2
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

    res_df = pd.DataFrame(results).sort_values(sic1_col).reset_index(drop=True)

    return res_df

def compute_cosine_similarity(df_mean: pd.DataFrame, id_col: str) -> pd.DataFrame:
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
    df = df_mean.copy()
    
    # Validasi input
    if id_col not in df.columns:
        raise ValueError(f"Kolom '{id_col}' tidak ditemukan di DataFrame")
    missing_traits = [t for t in TRAIT_COLS if t not in df.columns]
    if missing_traits:
        raise ValueError(f"Trait columns missing: {missing_traits}")

    df = df.set_index(id_col)[TRAIT_COLS]
    
    pearson_matrix = df.T.corr(method='pearson')
    
    return pearson_matrix