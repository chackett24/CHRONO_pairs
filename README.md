# CHRONOBERT for Pairs Trading  

Time-Aware Forecasting and Strategy Design

Try the live dashboard here: [chronopairs.streamlit.app](https://chronopairs.streamlit.app)

## Project Overview

This project explores whether CHRONOBERT, a Transformer model that incorporates temporal structure, improves spread prediction and trading performance in pairs trading strategies compared to standard BERT and traditional econometric models.

The project evaluates each model's ability to:
- Accurately predict spreads between cointegrated stock pairs
- Generate stable, risk-adjusted returns using those forecasts in live-like strategy simulations

## Methodology

1. Data Collection:  
   Historical price data was collected and processed into weekly spreads for chosen stock pairs.

2. Modeling Approaches:  
   - CHRONOBERT: A BERT-based Transformer trained on chronologically ordered spread sequences  
   - BERT: A pretrained model applied to unordered sequences  
   - Traditional: Linear OLS models and moving averages

3. Evaluation:  
   - Prediction Metrics: Mean Squared Error (MSE) and R²  
   - Strategy Metrics: Cumulative returns from long-short spread reversion trades

## Repository Structure

```
├── dashboard.py                 # Streamlit web application
├── proposal.md                  # Project proposal
├── requirements.txt             # Dependencies
├── analysis.ipynb              # Combined evaluation and plotting
├── get_data.ipynb              # Data acquisition and preprocessing
├── bert.ipynb                  # BERT model pipeline
├── chronobert.ipynb            # CHRONOBERT model training and predictions
├── traditional_OLS.ipynb       # Traditional forecasting models
│
├── outputs/                    # Model outputs and performance files
│   ├── CHRONOBERT_spreads_weekly.csv
│   ├── Traditional spreads weekly Return.csv
│   ├── bert_spread.csv
│   ├── spreads_testing.csv
│   └── spreads_weekly_large.csv
│
├── images/                     # Dashboard mockups and visuals
│   └── dashboard sketch
```

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/chackett24/CHRONO_pairs.git
   ```

2. Install the required Python packages:
   ```bash
   pip install streamlit pandas numpy scikit-learn yfinance transformers torch tqdm python-dateutil
   ```

3. To modify the data and update the benchmarked models:

   - Open `get_data.ipynb` and edit the tickers and date ranges to control which asset pairs and timeframes you want to analyze.
   - Run `bert.ipynb`, `chronobert.ipynb`, and `traditional_OLS.ipynb` to generate model predictions for each forecasting method. These notebooks will output updated spread predictions and returns to the `outputs/` folder.

4. Once predictions are updated:

   - You can explore the results and generate summary statistics using `analysis.ipynb`.
   - Or, run the interactive dashboard to explore the project visually:
     ```bash
     streamlit run dashboard.py
     ```


## Key Results

- Traditional OLS methods currently outperform both BERT and CHRONOBERT in terms of R² and mean squared error when predicting weekly spreads between asset pairs.
- Our current CHRONOBERT and BERT models achieve poor R² scores, suggesting limited predictive power in their current form.
- While CHRONOBERT's cumulative returns show promise on select pairs, overall performance is inconsistent and not yet robust.
- Traditional models remain competitive due to their simplicity, interpretability, and stability under this dataset.

## Notes and Future Work

- The underperformance of BERT and CHRONOBERT highlights a critical issue in how textual data is tokenized and preprocessed before being fed into the models.
- Future iterations of this project will focus on:
  - Improving news preprocessing and filtering.
  - Experimenting with domain-specific LLMs or fine-tuned financial transformers.
  - Aligning tokenization windows more closely with trading periods.
  - Exploring ensemble and hybrid models combining traditional features with embeddings from LLMs.
