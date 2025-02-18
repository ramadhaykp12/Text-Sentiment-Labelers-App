import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_config import label_text

st.write("# Sentiment Labeling Tools")

# Upload file
uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            try:
                df = pd.read_csv(uploaded_file, delimiter=',', encoding="utf-8")
            except UnicodeDecodeError:
                df = pd.read_csv(uploaded_file, delimiter=',', encoding="ISO-8859-1")
        elif uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)

        st.write("### Preview Data:")
        st.dataframe(df.head())
        st.write("Insert the choosen column name")
        df = df.head(20)
        column_name = st.text_input('Column name')
        choosen_column = df[column_name]
        start_label = st.button('Start')

        if start_label:
            df['sentiment_label'] = df[column_name].apply(lambda x: label_text({"text":x}))

            # Export the labeled data as csv file
            csv_file = df.to_csv(index=False)

            # Create download button for the new labeled data
            st.download_button(
                label='Download the labeled data',
                data=csv_file,
                file_name='labeled_data.csv',
                mime='text/csv'
            )

            # Visualization of sentiment label distribution in new labeled data
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.countplot(data=df, x=df['sentiment_label'], palette="viridis", ax=ax)
            plt.xticks(rotation=45)  # Rotasi label agar lebih rapi
            
            st.pyplot(fig)

    except Exception as e:
        st.error(f"Error: {e}")







