# app.py
import streamlit as st
import pandas as pd
import altair as alt
from io import BytesIO

st.set_page_config(page_title="SIC CEO Mean Viewer", layout="wide")

st.title("SIC CEO — Upload & Show Mean")
st.write("Upload file CSV atau Excel (xls/xlsx). Aplikasi akan menghitung mean dari kolom numerik dan menampilkannya dalam line chart.")

uploaded = st.file_uploader("Pilih file CSV / Excel", type=["csv", "xls", "xlsx"])

def load_file(uploaded_file):
    fname = uploaded_file.name.lower()
    try:
        if fname.endswith(".csv"):
            # try common encodings; allow user to change if needed later
            return pd.read_csv(uploaded_file)
        else:
            # excel
            return pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Gagal membaca file: {e}")
        return None

if uploaded is not None:
    df = load_file(uploaded)
    if df is None:
        st.stop()

    st.subheader("Preview data (top 10 rows)")
    st.dataframe(df.head(10))

    # Pilih kolom numerik saja
    numeric_df = df.select_dtypes(include=["number"])
    if numeric_df.shape[1] == 0:
        st.error("Tidak ada kolom numerik di file. Pastikan data mengandung kolom angka untuk dihitung mean-nya.")
        st.stop()

    st.sidebar.header("Pengaturan")
    axis_choice = st.sidebar.radio("Hitung mean per:", ("Per kolom (rata-rata tiap kolom)", "Per baris (rata-rata tiap baris)"))
    show_labels = st.sidebar.checkbox("Tampilkan nilai di tiap titik", value=False)
    sort_choice = st.sidebar.checkbox("Urutkan berdasarkan nilai mean (descending)", value=False)

    if axis_choice.startswith("Per kolom"):
        means = numeric_df.mean(axis=0, skipna=True)
        title = "Rata-rata tiap kolom"
        x_name = "column"
        mean_df = means.reset_index()
        mean_df.columns = [x_name, "mean"]
        # urutkan jika diminta
        if sort_choice:
            mean_df = mean_df.sort_values("mean", ascending=False).reset_index(drop=True)

        # jika jumlah kolom banyak, atur agar sumbu x terbaca
        chart = alt.Chart(mean_df).mark_line(point=True).encode(
            x=alt.X(f"{x_name}:N", title="Kolom"),
            y=alt.Y("mean:Q", title="Mean"),
            tooltip=[f"{x_name}:N", alt.Tooltip("mean:Q", format=".4f")]
        ).properties(width=800, height=400, title=title)

        if show_labels:
            text = alt.Chart(mean_df).mark_text(dy=-10, size=11).encode(
                x=alt.X(f"{x_name}:N"),
                y=alt.Y("mean:Q"),
                text=alt.Text("mean:Q", format=".4f")
            )
            chart = chart + text

    else:
        # per baris
        means = numeric_df.mean(axis=1, skipna=True)
        title = "Rata-rata tiap baris"
        # gunakan index asli jika ada, jika numerik gunakan range
        idx = df.index.astype(str)
        mean_df = pd.DataFrame({"index": idx, "mean": means})
        if sort_choice:
            mean_df = mean_df.sort_values("mean", ascending=False).reset_index(drop=True)
        chart = alt.Chart(mean_df).mark_line(point=True).encode(
            x=alt.X("index:N", title="Baris (index)"),
            y=alt.Y("mean:Q", title="Mean"),
            tooltip=["index", alt.Tooltip("mean:Q", format=".4f")]
        ).properties(width=800, height=400, title=title)

        if show_labels:
            text = alt.Chart(mean_df).mark_text(dy=-10, size=11).encode(
                x=alt.X("index:N"),
                y=alt.Y("mean:Q"),
                text=alt.Text("mean:Q", format=".4f")
            )
            chart = chart + text

    st.subheader("Line chart mean")
    st.altair_chart(chart, use_container_width=True)

    st.subheader("Table mean (top 50)")
    st.dataframe(mean_df.head(50))

    # offer download of mean table
    to_download = mean_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV hasil mean", data=to_download, file_name="means.csv", mime="text/csv")

else:
    st.info("Silakan upload file CSV atau Excel untuk melihat mean.")
