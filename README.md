# Stock Market Prediction App

A Streamlit-based application for predicting future stock market (index) prices using **deep learning** models. This project covers multiple indices (NASDAQ 100, S&P 500, FTSE 100, Nikkei 225, BIST 100, CAC 40, and Dow Jones) and provides an easy-to-use web interface for viewing predictions, charts, and historical data.

---

## Overview
This application **predicts 5-day future prices** of various stock indices by leveraging pre-trained LSTM (or similar) neural network models stored in `.h5` files. Historical stock data is fetched via [yfinance](https://pypi.org/project/yfinance/), scaled with `MinMaxScaler`, and then passed to the model to generate predictions.  

Users interact with the app through a **Streamlit** user interface to:
1. Select the desired stock index (e.g., NASDAQ 100, S&P 500, BIST 100, etc.).
2. Pick a reference date.
3. Generate predictions and visualize them alongside real market data (graphs, tables, volatility, RSI, moving averages, etc.).

---

## Demo Video

https://github.com/user-attachments/assets/79d57a5d-eaf4-4986-8896-9b9d8ec415a3

---

## Features
- **Multiple Indices**: Predict prices for NASDAQ 100 (^IXIC), S&P 500 (^GSPC), FTSE 100 (^FTSE), Nikkei 225 (^N225), BIST 100 (XU100.IS), CAC 40 (^FCHI), and Dow Jones (^DJI).
- **Deep Learning Models**: Each index has its own pre-trained Keras/TensorFlow model file (`.h5`).
- **Streamlit Interface**: Simple sidebar menus for user input; intuitive charts and tables for result display.
- **Advanced Metrics**:
  - Next 5-day forecast
  - Recent 10-day actual vs. predicted comparison
  - Daily changes & 10-day rolling volatility
  - Moving averages (SMA10, SMA20) & RSI
  - Tabular summary of predictions (mean, max, min, standard deviation)

---

## Technologies Used
- **Python** (3.8+ recommended)
- **Streamlit** (for building the web app)
- **TensorFlow/Keras** (for loading LSTM models)
- **scikit-learn** (for MinMaxScaler and other utilities)
- **NumPy / Pandas** (for data manipulation)
- **Matplotlib** (for plots)
- **yfinance** (for fetching market data)

---

## Installation
1. **Clone this repository** (or download the ZIP):
   ```bash
   git clone https://github.com/alperenec/stockMarketPrediction-App
