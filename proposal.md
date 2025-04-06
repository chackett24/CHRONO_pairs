# Research Proposal: Using CHRONOBERT Time Series Forecasting to Utilize Pairs Trading Strategies

**By Caleb Johnson, Casey Hackett, Edgard Cuadra, Shanshan Gong**

---

## Bigger Question  
Can methods originally developed for modeling temporally sensitive textual data be effectively adapted for financial time series forecasting?  
More specifically, can a time-insensitive LLM like CHRONOBERT capture relationships in financial asset price movements over time to enhance the performance of systematic trading strategies?

---

## Specific Questions  
- Can CHRONOBERT be used to implement LLM forecasting and pairs trading techniques?  
- How does CHRONOBERT compare when benchmarked against other, time-sensitive, LLMs?  
- How does CHRONOBERT compare to traditional pairs trading forecasting methods?

---

## Hypotheses  
1. CHRONOBERT will outperform models that do not encode time sensitivity in forecasting spread direction or magnitude.  
2. CHRONOBERT generates more consistent trading profits across multiple market regimes than static models.

---

## Success Metrics  
- Higher R² and lower MSE for predicting spread direction and magnitude  
- Higher Sharpe Ratio  
- Better returns across multiple market scenarios  
- All benchmarked against other LLM models and traditional forecasting methods

---

## Data Needed  

### Historic Spread Data  
| Date       | Ticker Pair | Spread |
|------------|-------------|--------|
||||
||
||

### Market Return Data  
| Date       | Ticker       | Adjusted Close |
|------------|--------------|----------------|
||
||
||

### Predictions (Future Spreads)  
| Date       | Ticker Pair | Spread | CHRONOBERT Spread | CHRONOBERT Position | LLM Spread | LLM Position | Traditional Method Spread | Traditional Position |
|------------|-------------|--------|-------------------|----------------------|-------------|---------------|----------------------------|----------------------|
||
||
||

---

## Observations  
- **Unit of Observation:** Ticker Pair by Date

---

## Sample Period  
(up for grabs)
- **Training Period:** 2012–2016  
- **Testing Period:** 2017

---

## Sample Conditions  
- Include only ticker pairs with high market cap and liquidity for robust data  
- Limit sample to S&P 500 stocks  
---

## Variables

### Absolutely Necessary  
- Historic spread data for stock pairs  
- Historic stock returns  
- CHRONOBERT predictions  
- Other LLM predictions  
- Traditional method predictions  
- CHRONOBERT holdings  
- Other LLM holdings  
- Traditional method holdings  

### Nice to Have  
- Market condition indicators  
- Macro economic indicators (e.g., VIX, interest rates, inflation)

---

## Data Inventory

### Already Have*
*(We have learned how to acquire it previously)
- Historic spread data  
- Historic return data  

### Need to Generate  
- CHRONOBERT spread and position predictions  
- Other LLM spread and position predictions  
- Traditional method spread and position predictions  

---

## Data Collection  
- Use `yfinance` or CRSP for market return and spread data  
- Calculate spread manually using historical close prices

---

## Raw Inputs & Storage  

**Folder Structure**  (Up for grabs) 
```
/project/
├── inputs/ (in .gitignore)
│   ├── text files
│   ├── html files
│   ├── zip
├── outputs/
│   ├── returns.csv
│   ├── spreads.csv
│   ├── chronobert.csv
│   ├── llm.csv
│   ├── traditional.csv
├── get_data.ipynb
├── chronobert.ipynb/
├── llm.ipynb
├── traditional.ipynb
├── analysis.ipynb
```

---

## Data Transformation Pipeline (High-Level)  
1. Pull S&P 500 price and return data  
2. Calculate spreads manually from adjusted close 
3. Train CHRONOBERT, competing LLM, and traditional models on training data  
4. Generate 1-year-ahead spread forecasts  
5. Classify signals (long/short) based on forecast direction/magnitude  
6. Construct portfolios for each model and simulate weekly returns  
7. Evaluate forecasting and trading performance  
