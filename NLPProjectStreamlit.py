import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import yfinance as yf

# Import the functions and variables from your reference code
from NLPProject import ticker, finance_lexicon, manage_lexicon, custom_sentiment_analyzer, news_tables, parsed_data, df, mean_df, stock_data

# Create a Streamlit sidebar for user input
st.sidebar.title("User Input")
ticker = st.sidebar.text_input("Enter the stock ticker (e.g., AMD):")

# Create a Streamlit sidebar for sentiment lexicon management
st.sidebar.title("Sentiment Lexicon Management")
manage_lexicon()

# Create a Streamlit section for displaying sentiment analysis and stock price data
st.title("Sentiment Analysis Dashboard")

# Display the existing finance-related sentiment words
st.sidebar.title("Existing finance-related sentiment words:")
for word, score in finance_lexicon.items():
    st.sidebar.write(f"{word}: {score}")

# Create a Streamlit section for displaying sentiment analysis and stock price data
st.write("### Sentiment Analysis and Stock Price Over Time")

# Plot the sentiment data as a line chart
fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.set_xlabel("Date")
ax1.set_ylabel("Mean Compound Score", color='tab:blue')
ax1.plot(mean_df.index, mean_df[ticker], color='tab:blue', marker='o', linestyle='-', markersize=5, label="Sentiment")
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax1.legend(loc='upper left')

# Create a second y-axis for stock price data
ax2 = ax1.twinx()
ax2.set_ylabel("Stock Price", color='tab:red')
ax2.plot(stock_data.index, stock_data['Close'], color='tab:red', marker='o', linestyle='-', markersize=5, label="Stock Price")
ax2.tick_params(axis='y', labelcolor='tab:red')
ax2.legend(loc='upper right')

st.pyplot(fig)

# Display the DataFrame
st.write(df)