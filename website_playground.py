import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error

st.title('CHRONOBERT Pairs Trading')

# Load and prepare data
spread_df = pd.read_csv("outputs/spreads.csv", parse_dates=["Date"])
spread_df.set_index("Date", inplace=True)
portfolios_df = pd.read_csv("outputs/portfolios.csv", parse_dates=["Date"])
portfolios_df.set_index("Date", inplace=True)

chrono_df = pd.read_csv("outputs/chrono_dummy.csv", parse_dates=["Date"])
chrono_df.set_index("Date", inplace=True)

bert_df = pd.read_csv("outputs/bert_dummy.csv", parse_dates=["Date"])
bert_df.set_index("Date", inplace=True)

traditional_df = pd.read_csv("outputs/traditional_dummy.csv", parse_dates=["Date"])
traditional_df.set_index("Date", inplace=True)

# Merge all
spread_df = spread_df.merge(chrono_df, how="inner", on=["Date", "Ticker Pair"], validate='one_to_one')
spread_df = spread_df.merge(bert_df, how="inner", on=["Date", "Ticker Pair"], validate='one_to_one')
spread_df = spread_df.merge(traditional_df, how="inner", on=["Date", "Ticker Pair"], validate='one_to_one')

# Select ticker pair
ticker_options = spread_df["Ticker Pair"].unique()
selected_pair = st.selectbox("Select Ticker Pair:", ticker_options)

# Select which prediction columns to show
col_options = {
    "CHRONOBERT": st.checkbox("CHRONOBERT", value=True),
    "BERT": st.checkbox("BERT", value=True),
    "Traditional": st.checkbox("Traditional", value=True)
}
spread_columns = ["Spread"] + [col + " Spread" for col, checked in col_options.items() if checked]
return_columns = [col + " Cumulative Return" for col, checked in col_options.items() if checked]

# Filter by selected pair
filtered_df = spread_df[spread_df["Ticker Pair"] == selected_pair]
portfolios_df = portfolios_df[portfolios_df["Ticker Pair"] == selected_pair]

# Compute and display R² and MSE
metrics = {}
for col in spread_columns:
    if col != "Spread":
        r2 = r2_score(filtered_df["Spread"], filtered_df[col])
        mse = mean_squared_error(filtered_df["Spread"], filtered_df[col])
        metrics[col] = {"R²": r2, "MSE": mse}

# Show data
st.dataframe(filtered_df[spread_columns])

# Show metrics
if metrics:
    st.subheader("Model Performance vs True Spread")
    st.table(pd.DataFrame(metrics).T)

# Plot chart
st.subheader('Model Spread vs True Spread')
st.line_chart(filtered_df[spread_columns])
st.subheader('Model Returns')
if not return_columns:
    st.info("Please check at least one box to see returns.") 
else:
    st.line_chart(portfolios_df[return_columns])
