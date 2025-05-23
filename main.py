import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="CSV Visualizer", layout="wide")
st.title("📊 CSV Visualizer")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.subheader("🔍 Data Preview")
        st.dataframe(df.head())

        st.subheader("📈 Column Statistics")
        st.write(df.describe(include='all'))

        st.subheader("📊 Quick Visualization")
        col = st.selectbox("Select a column to visualize", df.columns)

        if pd.api.types.is_numeric_dtype(df[col]):
            st.plotly_chart(px.histogram(df, x=col), use_container_width=True)
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            st.plotly_chart(px.line(df, x=col, y=df.select_dtypes('number').columns[0]), use_container_width=True)
        else:
            st.plotly_chart(px.bar(df[col].value_counts().reset_index(), x='index', y=col), use_container_width=True)
    except Exception as e:
        st.error(f"Error reading file: {e}")
