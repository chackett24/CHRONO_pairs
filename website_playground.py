import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error

st.set_page_config(layout="wide")
st.title('CHRONOBERT Pairs Trading Dashboard')

# Load and prepare data
spread_df = pd.read_csv("outputs/spreads_testing.csv", parse_dates=["Date"])
spread_df.set_index("Date", inplace=True)
portfolios_df = pd.read_csv("outputs/portfolios.csv", parse_dates=["Date"])
portfolios_df.set_index("Date", inplace=True)

chrono_df = pd.read_csv("outputs/chrono_dummy.csv", parse_dates=["Date"])
chrono_df.set_index("Date", inplace=True)
bert_df = pd.read_csv("outputs/bert_dummy.csv", parse_dates=["Date"])
bert_df.set_index("Date", inplace=True)
traditional_df = pd.read_csv("outputs/traditional_dummy.csv", parse_dates=["Date"])
traditional_df.set_index("Date", inplace=True)

# Merge all predictions
spread_df = spread_df.merge(chrono_df, how="inner", on=["Date", "Ticker Pair"], validate='one_to_one')
spread_df = spread_df.merge(bert_df, how="inner", on=["Date", "Ticker Pair"], validate='one_to_one')
spread_df = spread_df.merge(traditional_df, how="inner", on=["Date", "Ticker Pair"], validate='one_to_one')

# Create tabs
tabs = st.tabs(["🧭 Project Aims", "🔬 Hypotheses", "📊 Data Analysis", "🧠 Hypothesis Evaluation"])

# ---- Tab 1: Project Aims ----
with tabs[0]:
    st.header("Project Aims")
    st.markdown("""
    This project explores whether the **CHRONOBERT model**, a Transformer-based time series model, can better predict 
    asset spread movements for use in **pairs trading**. We aim to compare CHRONOBERT with:
    - Traditional econometric models (e.g., ARIMA/rolling mean)
    - BERT (fine-tuned on time-series format)

    We evaluate performance based on:
    - **Spread prediction accuracy**
    - **Profitability of trading signals derived from predictions**
    """)

# ---- Tab 2: Hypotheses ----
with tabs[1]:
    st.header("Hypotheses")
    st.markdown("""
    - **H1:** CHRONOBERT predicts spreads more accurately (lower MSE, higher R²) than traditional models and BERT.
    - **H2:** Trading strategies based on CHRONOBERT predictions yield higher cumulative returns.
    """)

# ---- Tab 3: Data ----
with tabs[2]:
    st.header("Data")
    st.markdown("Choose a ticker pair to explore:")
    ticker_options = spread_df["Ticker Pair"].unique()
    selected_pair = st.selectbox("Select Ticker Pair:", ticker_options, key="ticker")

    st.markdown("Select which model predictions to show:")
    col_options = {
        "CHRONOBERT": st.checkbox("CHRONOBERT", value=True, key="chrono"),
        "BERT": st.checkbox("BERT", value=True, key="bert"),
        "Traditional": st.checkbox("Traditional", value=True, key="trad")
    }
    st.header("Model Outputs and Metrics")
    
    spread_columns = ["Spread"] + [col + " Spread" for col, checked in col_options.items() if checked]
    return_columns = [col + " Cumulative Return" for col, checked in col_options.items() if checked]

    # Filter by pair
    filtered_df = spread_df[spread_df["Ticker Pair"] == selected_pair]
    port_df = portfolios_df[portfolios_df["Ticker Pair"] == selected_pair]

    # Compute metrics
    metrics = {}
    for col in spread_columns:
        if col != "Spread":
            r2 = r2_score(filtered_df["Spread"], filtered_df[col])
            mse = mean_squared_error(filtered_df["Spread"], filtered_df[col])
            metrics[col] = {"R²": r2, "MSE": mse}

    # Display
    st.subheader("Model Spread vs True Spread")
    st.line_chart(filtered_df[spread_columns])

    st.subheader("Model Performance Metrics")
    if metrics:
        st.table(pd.DataFrame(metrics).T)

    st.subheader("Cumulative Returns")
    if not return_columns:
        st.info("Please check at least one model to view return plots.")
    else:
        st.line_chart(port_df[return_columns])
        
        st.subheader("Hypothesis Evaluation Based on Your Selections")

    if metrics and not port_df.empty:
        # H1: Check which model had lowest MSE
        best_model_spread = min(metrics.items(), key=lambda x: x[1]["MSE"])[0]
        best_r2_model = max(metrics.items(), key=lambda x: x[1]["R²"])[0]

        spread_support = "✅ Supported" if "CHRONOBERT Spread" == best_model_spread else "❌ Not Supported"
        spread_support_bool = True if "CHRONOBERT Spread" == best_model_spread else False
        r2_support = "✅ Supported" if "CHRONOBERT Spread" == best_r2_model else "❌ Not Supported"
        r2_support_bool = True if "CHRONOBERT Spread" == best_r2_model else False
        
        if spread_support_bool and r2_support_bool:
            h1_support = "✅ Supported"
        elif (not spread_support_bool) and (not r2_support_bool):
            h1_support = "❌ Not Supported"
        else:
            h1_support = "🟡 Kind of Supported"

        # H2: Check which model had highest cumulative return
        final_returns = port_df[return_columns].iloc[-1]
        best_return_model = final_returns.idxmax()
        h2_support = "✅ Supported" if "CHRONOBERT Cumulative Return" == best_return_model else "❌ Not Supported"

        # Display
        st.markdown(f"""
        ### Hypothesis 1: CHRONOBERT predicts spreads most accurately  
        - Best MSE model: **{best_model_spread}**
        
        {spread_support}
        
        - Best R² model: **{best_r2_model}**  
        
        {r2_support}
        
        **Conclusion:** {h1_support}

        ### Hypothesis 2: CHRONOBERT returns are highest  
        - Best final return model: **{best_return_model}**  
        **Conclusion:** {h2_support}
        """)
    else:
        st.info("Select at least one model to evaluate hypotheses.")


# ---- Tab 4: Hypothesis Evaluation ----
with tabs[3]:
    st.header("Do the Results Support Our Hypotheses?")
    st.markdown("""
    Based on the spread prediction accuracy and return curves:
    
    - **H1**: *[Insert conclusion here — e.g., CHRONOBERT outperforms others in MSE/R² across most pairs]*
    - **H2**: *[Insert conclusion here — e.g., CHRONOBERT returns are higher and more stable]*

    These results suggest that CHRONOBERT has potential to enhance spread-based trading strategies, though further backtesting across more assets and market conditions is recommended.
    """)

