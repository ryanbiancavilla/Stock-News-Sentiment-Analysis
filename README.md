# Sentiment Analysis and Stock Price Visualization

This is a Jupyter Notebook containing Python code that performs sentiment analysis on financial news headlines related to a given stock ticker and visualizes the sentiment trend along with the stock's historical price data. The code utilizes various libraries and functions to achieve this.

## Prerequisites

Before running the code in Jupyter Notebook, make sure you have the following Python libraries installed. You can install them using `!pip` commands in a Jupyter cell:

```python
!pip install urllib3 beautifulsoup4 nltk pandas matplotlib yfinance
```

Additionally, you need to download NLTK's VADER sentiment analyzer data. You can do this by running the following code in a Jupyter cell:

```python
import nltk
nltk.download('vader_lexicon')
nltk.download('stopwords')
```

## Usage

1. Execute each code cell in the Jupyter Notebook sequentially.
2. When prompted, enter the stock ticker in the provided input field.

## Features

### Custom Sentiment Analysis

The code performs sentiment analysis on financial news headlines using two methods:

- A custom sentiment lexicon containing finance-specific words and their sentiment scores.
- NLTK's VADER sentiment analyzer.

The results include a compound score and lists of positive and negative words for each headline.

### Sentiment Lexicon Management

The code allows you to manage the sentiment lexicon, including adding, editing, or deleting sentiment words and their scores. This enables you to tailor sentiment analysis to your specific needs.

### Data Collection

The code collects news data related to the entered stock ticker from FinViz, parses it, and stores it in a Pandas DataFrame. The data includes the ticker, time, date, title, compound score, and lists of positive and negative words.

### Data Preprocessing

The title text is preprocessed, including converting it to lowercase, removing non-alphabetic characters, and eliminating common English stopwords.

### Visualization

The code visualizes the mean compound sentiment scores and the historical stock price of the specified stock ticker over time in a line chart. It helps you identify potential correlations between sentiment and stock price trends.

## Note

- This code is just stock sentiment analysis, in the future the data will be regressed to stock prices 
- This code may not work if the source website structure (FinViz) changes.
- The accuracy of sentiment analysis largely depends on the quality of the sentiment lexicon and the nature of the text being analyzed.
- Always ensure you have the necessary permissions and rights to access and use financial data and news from the specified sources.

Feel free to adapt and extend this code within your Jupyter Notebook to suit your specific requirements. Enjoy your sentiment analysis and stock price visualization!
