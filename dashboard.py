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

chrono_df = pd.read_csv("outputs/CHRONOBERT_spreads_weekly.csv", parse_dates=["Date"])
chrono_df.set_index("Date", inplace=True)
bert_df = pd.read_csv("outputs/bert_spread.csv", parse_dates=["Date"])
bert_df.set_index("Date", inplace=True)
bert_df.rename(columns={"Bert Spread": "BERT Spread", "Bert Position": "Bert Position", "Ticker_Pair":"Ticker Pair"}, inplace=True)
traditional_df = pd.read_csv("outputs/Traditional Spreads weekly Return.csv", parse_dates=["Date"])
traditional_df.set_index("Date", inplace=True)
traditional_df.rename(columns={"Ticker Pair": "Ticker Pair", "Traditional_Spread": "Traditional Spread"}, inplace=True)

# Merge all predictions
spread_df = spread_df.merge(chrono_df, how="inner", on=["Date", "Ticker Pair"], validate='one_to_one')
spread_df = spread_df.merge(bert_df, how="inner", on=["Date", "Ticker Pair"], validate='one_to_one')
spread_df = spread_df.merge(traditional_df, how="inner", on=["Date", "Ticker Pair"], validate='one_to_one')

# Create tabs
tabs = st.tabs(["Overview", "🔬 Hypotheses", "📊 Data Playground", "🧠 Hypothesis Evaluation"])

# ---- Tab 1: Project Aims ----
with tabs[0]:
    st.header("Overview")
    st.markdown("""
    This project explores whether **CHRONOBERT**, a Transformer-based model trained on chronologically ordered data, 
    can improve **financial time series forecasting**, particularly in the context of **pairs trading**.

    ---

    ### What is Pairs Trading?

    **Pairs trading** is a market-neutral strategy that involves taking long and short positions simultaneously in two historically correlated stocks. 
    When the price spread between the two assets diverges from its historical average, the strategy profits from the expectation that the spread will revert to the mean. 
    Success depends on accurately forecasting these spread movements.

    ---

    ### What is CHRONOBERT?

    **CHRONOBERT** is a variation of BERT that is trained exclusively on **chronologically ordered data** to prevent lookahead bias—an issue especially problematic in time-sensitive domains like finance. 
    By respecting temporal order during training, CHRONOBERT avoids data leakage and provides more realistic, generalizable forecasts. 
    It outperforms standard BERT on multiple benchmarks involving time-sensitive text, making it a promising candidate for financial forecasting.

    ---

    ### Project Goals

    We evaluate whether CHRONOBERT can improve:
    - **Spread prediction accuracy** (measured by MSE and R²)
    - **Profitability of trading signals** (measured by Sharpe Ratio and cumulative returns)
    - **Out-of-sample robustness** (across different market conditions)

    We benchmark its performance against:
    - **BERT**, which is not time-aware
    - **Traditional methods**, such as moving averages and econometric models

    All models are trained on 2016–2018 data and tested out-of-sample on 2019 equity pairs from the S&P 500 universe.
    """)



# ---- Tab 2: Hypotheses ----
with tabs[1]:
    st.header("Hypotheses")
    st.markdown("""
    - **H1:** CHRONOBERT predicts spreads more accurately (lower MSE, higher R²) than traditional models and BERT.
    - **H2:** Trading strategies based on CHRONOBERT predictions yield higher cumulative returns.
    """)
    st.markdown("""
    ### Why These Hypotheses?

    We propose two main hypotheses based on CHRONOBERT's unique design and our objective of reducing overfitting and improving out-of-sample performance:

    - **H1:** *CHRONOBERT predicts spreads more accurately (lower MSE, higher R²) than traditional models and BERT.*  
    This hypothesis stems from CHRONOBERT’s training on strictly chronological data, which minimizes lookahead bias—a common problem in financial modeling. We expect that this temporal integrity will result in more trustworthy forecasts when evaluated out-of-sample.

    - **H2:** *Trading strategies based on CHRONOBERT predictions yield higher cumulative returns.*  
    If CHRONOBERT can produce better predictions of spread direction and magnitude, it should translate into more effective trading signals. We hypothesize that portfolios using CHRONOBERT forecasts will generate higher Sharpe Ratios and cumulative returns compared to those based on BERT or traditional techniques.

    Together, these hypotheses test both **predictive power** and **practical trading performance**, offering a complete view of CHRONOBERT’s value in financial applications.
    """)
    
    st.markdown("""
    ### Success Metrics

    To evaluate whether CHRONOBERT provides a meaningful improvement over existing methods, we focus on both **forecasting quality** and **trading performance**:

    - **Prediction Accuracy**  
    - Lower **Mean Squared Error (MSE)** between predicted and actual spreads  
    - Higher **R²** for out-of-sample predictions  
    These metrics assess how well each model captures the dynamics of spread movements.

    - **Trading Performance**  
    - Higher **Sharpe Ratio** (risk-adjusted returns)  
    - Greater **cumulative returns** across the testing period  
 
    This indicates reduced overfitting and improved generalizability—key benefits of CHRONOBERT’s time-respecting training.

    By evaluating each model across these dimensions, we can holistically measure whether CHRONOBERT enhances pairs trading outcomes.
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
    y_true = spread_df["Spread"]
    final_returns_per_pair = portfolios_df.groupby("Ticker Pair").tail(1)


    avg_returns = {
        "CHRONOBERT": final_returns_per_pair["CHRONOBERT Cumulative Return"].mean(),
        "BERT": final_returns_per_pair["BERT Cumulative Return"].mean(),
        "Traditional": final_returns_per_pair["Traditional Cumulative Return"].mean(),
    }

    results = {
        "Model": ["CHRONOBERT", "BERT", "Traditional"],
        "MSE": [
            mean_squared_error(y_true, spread_df["CHRONOBERT Spread"]),
            mean_squared_error(y_true, spread_df["BERT Spread"]),
            mean_squared_error(y_true, spread_df["Traditional Spread"]),
        ],
        "R2": [
            r2_score(y_true, spread_df["CHRONOBERT Spread"]),
            r2_score(y_true, spread_df["BERT Spread"]),
            r2_score(y_true, spread_df["Traditional Spread"]),
        ],
        "Return": [
            avg_returns["CHRONOBERT"],
            avg_returns["BERT"],
            avg_returns["Traditional"],
        ]
    }
    metrics_df = pd.DataFrame(results)
    # Identify best-performing model based on MSE and R²
    best_mse_model = metrics_df.loc[metrics_df["MSE"].idxmin(), "Model"]
    best_r2_model = metrics_df.loc[metrics_df["R2"].idxmax(), "Model"]
    best_return_model = metrics_df.loc[metrics_df["Return"].idxmax(), "Model"]

    # Analyze Hypotheses
    h1_result = f"CHRONOBERT {'outperforms' if best_mse_model == 'CHRONOBERT' and best_r2_model == 'CHRONOBERT' else 'does not clearly outperform'} others in both MSE and R²."
    h2_result = f"CHRONOBERT {'achieved the highest average cumulative return' if best_return_model == 'CHRONOBERT' else 'did not achieve the highest return'}."

    # Header and Summary
    st.header("Do the Results Support Our Hypotheses?")
    st.markdown(f"""
    Based on the spread prediction accuracy and trading performance:

    - **H1**: *{h1_result}*
    - **H2**: *{h2_result}*

    These results suggest that while we believe CHRONOBERT has potential to enhance spread-based trading strategies, especially in predictive modeling. 
    However, additional work—such as improved tokens and parameters is needed in order to create better predictions.
    """)

    # Optional Detailed Table and Charts
    st.subheader("Metric Summary by Model")
    st.dataframe(metrics_df.style.format({
        "MSE": "{:.4f}",
        "R2": "{:.4f}",
        "Return": "{:.4f}"
    }))

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**MSE (Lower is Better)**")
        st.bar_chart(metrics_df.set_index("Model")["MSE"])

    with col2:
        st.markdown("**R² (Higher is Better)**")
        st.bar_chart(metrics_df.set_index("Model")["R2"])

    with col3:
        st.markdown("**Avg. Cumulative Return (Higher is Better)**")
        st.bar_chart(metrics_df.set_index("Model")["Return"])

