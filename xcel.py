import streamlit as st
import pandas as pd
import numpy as np

# --- PAGE CONFIG ---
st.set_page_config(page_title="📊 Smart Dataset Preprocessor", layout="wide")

# --- TITLE ---
st.title("📊 Smart Dataset Preprocessor")

# --- SIDEBAR ---
st.sidebar.header("⚙️ Preprocessing Options")

# --- FILE UPLOAD ---
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

@st.cache_data
def load_data(file):
    return pd.read_csv(file)

if uploaded_file is not None:
    df = load_data(uploaded_file)

    st.subheader("🔍 Data Preview")
    st.dataframe(df.head(), use_container_width=True)

    st.subheader("📏 Basic Info")
    st.write(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")
    st.write(df.describe())

    st.subheader("🧮 Null Summary")
    st.write(df.isnull().sum())

    with st.sidebar.expander("🧴 Fill Missing Values"):
        fill_method = st.radio("Fill method:", ["None", "Mean", "Median", "Zero", "Custom"])
        custom_value = None
        if fill_method == "Custom":
            custom_value = st.text_input("Enter custom value:")

    with st.sidebar.expander("🗑️ Drop Nulls"):
        drop_choice = st.radio("Drop nulls by:", ["None", "Drop Rows", "Drop Columns"])

    with st.sidebar.expander("🚫 Remove Outliers"):
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        outlier_col = st.selectbox("Numeric column to clean:", ["None"] + numeric_cols)

    with st.sidebar.expander("✏️ Rename Columns"):
        col_to_rename = st.selectbox("Column to rename:", ["None"] + df.columns.tolist())
        new_col_name = ""
        if col_to_rename != "None":
            new_col_name = st.text_input(f"New name for '{col_to_rename}':")

    if st.sidebar.button("⚡ Apply Preprocessing"):
        with st.spinner("Processing..."):
            if fill_method != "None":
                if fill_method == "Mean":
                    df = df.fillna(df.mean(numeric_only=True))
                elif fill_method == "Median":
                    df = df.fillna(df.median(numeric_only=True))
                elif fill_method == "Zero":
                    df = df.fillna(0)
                elif fill_method == "Custom" and custom_value != "":
                    df = df.fillna(custom_value)

            if drop_choice == "Drop Rows":
                df = df.dropna(axis=0)
            elif drop_choice == "Drop Columns":
                df = df.dropna(axis=1)

            if outlier_col != "None":
                Q1 = df[outlier_col].quantile(0.25)
                Q3 = df[outlier_col].quantile(0.75)
                IQR = Q3 - Q1
                df = df[(df[outlier_col] >= Q1 - 1.5 * IQR) & (df[outlier_col] <= Q3 + 1.5 * IQR)]

            if col_to_rename != "None" and new_col_name.strip() != "":
                df = df.rename(columns={col_to_rename: new_col_name.strip()})

            st.success("✅ Preprocessing Applied!")
            st.dataframe(df.head(), use_container_width=True)

    # Download
    if uploaded_file:
        @st.cache_data
        def convert_df(df):
            return df.to_csv(index=False).encode('utf-8')

        csv = convert_df(df)
        st.download_button(
            "⬇️ Download Cleaned CSV",
            csv,
            "cleaned_data.csv",
            "text/csv"
        )
else:
    st.info("👈 Upload a CSV file to get started.")
