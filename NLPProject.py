from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
import ssl
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import yfinance as yf
import re
import nltk
from nltk.corpus import stopwords

# Allow the user to input the stock ticker
ticker = input("Enter the stock ticker (e.g., AMD): ")

# Define the URL for FinViz
finviz_url = "https://finviz.com/quote.ashx?t="
tickers = [ticker]
news_tables = {}

# Define a custom sentiment lexicon with finance-specific words
finance_lexicon = {
    'bullish': 1.0,
    'bearish': -1.0,
    'profit': 0.5,
    'loss': -0.5,
    'dividend': 0.3,
    'earnings': 0.3,
    'revenue': 0.2,
    'volatility': -0.2,
    'market': 0.1,
    'stock': 0.1,
    'shares': 0.1,
}

# Initialize the NLTK VADER sentiment analyzer
vader = SentimentIntensityAnalyzer()

# Display the existing finance-related sentiment words
print("\nExisting finance-related sentiment words:")
for word, score in finance_lexicon.items():
    print(f"{word}: {score}")

# Function to manage the sentiment lexicon
def manage_lexicon():
    while True:
        print("\nSentiment Lexicon Management:")
        print("1. Add a sentiment word")
        print("2. Edit an existing sentiment word")
        print("3. Delete an existing sentiment word")
        print("4. Leave existing words as they are")
        print("5. Exit sentiment lexicon management")
        choice = input("Enter your choice: ")

        if choice == '1':
            word = input("Enter the sentiment word: ")
            score = float(input("Enter the sentiment score: "))
            finance_lexicon[word] = score
        elif choice == '2':
            word = input("Enter the sentiment word to edit: ")
            if word in finance_lexicon:
                score = float(input(f"Edit the sentiment score for {word}: "))
                finance_lexicon[word] = score
            else:
                print(f"{word} is not in the lexicon.")
        elif choice == '3':
            word = input("Enter the sentiment word to delete: ")
            if word in finance_lexicon:
                del finance_lexicon[word]
            else:
                print(f"{word} is not in the lexicon.")
        elif choice == '4':
            pass
        elif choice == '5':
            break
        else:
            print("Invalid choice. Please try again.")

# Function to perform custom sentiment analysis
def custom_sentiment_analyzer(text):
    """
    Custom sentiment analysis function that combines lexicon-based and VADER sentiment analysis.

    Args:
        text (str): The text to be analyzed.

    Returns:
        float: Compound sentiment score.
        list: Positive words.
        list: Negative words.
    """
    compound_score = 0
    positive_words = []  # List to store positive words
    negative_words = []  # List to store negative words
    words = text.split()

    for word in words:
        if word in finance_lexicon:
            compound_score += finance_lexicon[word]
        else:
            sentiment_scores = vader.polarity_scores(word)
            compound_score += sentiment_scores['compound']
            if sentiment_scores['pos'] > sentiment_scores['neg']:
                positive_words.append(word)
            elif sentiment_scores['neg'] > sentiment_scores['pos']:
                negative_words.append(word)

    return compound_score, positive_words, negative_words

# Allow the user to manage the sentiment lexicon
manage_lexicon()

# Fetch news data from FinViz
for ticker in tickers:
    url = finviz_url + ticker
    req = Request(url=url, headers={"user-agent": "SentimentAnalysis"})
    gcontext = ssl.SSLContext()
    res = urlopen(req, context=gcontext)

    html = BeautifulSoup(res, "lxml")
    table = html.find(id="news-table")
    news_tables[ticker] = table

# Process and store news data
parsed_data = []
for ticker, news_table in news_tables.items():
    for row in news_table.findAll("tr"):
        title = row.a.text if row.a is not None else "No title available"
        date_info = row.td.get_text()
        date_data = date_info.split()
        if len(date_data) == 1:
            if date_data[0] == "Today":
                current_date = datetime.now().strftime("%b-%d-%y")
                date = current_date
            else:
                time = datetime.strptime(date_data[0], '%I:%M%p').strftime('%H:%M')
        else:
            time = datetime.strptime(date_data[1], '%I:%M%p').strftime('%H:%M')
            date = date_data[0] if date_data[0] != "Today" else datetime.now().strftime("%b-%d-%y")
        compound_score, positive_words, negative_words = custom_sentiment_analyzer(title)
        parsed_data.append([ticker, time, date, title, compound_score, positive_words, negative_words])

# Create a pandas DataFrame
df = pd.DataFrame(parsed_data, columns=(["ticker", "time", "date", "title", "compound_score", "positive_words", "negative_words"]))

# Data Preprocessing
def preprocess_text(text):
    """
    Preprocess text data by converting to lowercase and removing non-alphabetic characters and stopwords.

    Args:
        text (str): The text to be preprocessed.

    Returns:
        str: Preprocessed text.
    """
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

# Apply preprocessing to the title column
df["title"] = df["title"].apply(preprocess_text)

# Convert the date column to datetime format
df["date"] = pd.to_datetime(df["date"], format="%b-%d-%y").dt.date

# Group by date and ticker, and calculate the mean compound score
mean_df = df.groupby(['date', 'ticker'])['compound_score'].mean()
mean_df = mean_df.unstack()

# Define the date range based on sentiment data
start_date = mean_df.index.min()
end_date = mean_df.index.max()

# Fetch historical stock price data for the specified date range
stock_data = yf.download(ticker, start=start_date, end=end_date)

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

plt.title(f"Sentiment Analysis and Stock Price for {ticker} Over Time")
plt.show()

# Print the DataFrame
print(df)
