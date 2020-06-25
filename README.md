# Talking-points-global-hackathon

## Data Source & Description

## Source

We have gathered the data for training our model from Kaggle's dataset [Daily News for Stock Market Prediction](https://www.kaggle.com/aaron7sun/stocknews)

### Description:

#### Data
There are two channels of data provided in this dataset:

**1. News data:** The data-set owner has crawled historical news headlines from Reddit WorldNews Channel (/r/worldnews). They are ranked by reddit users' votes, and only the top 25 headlines are considered for a single date.

**2. Stock data:** Dow Jones Industrial Average (DJIA) is used to "prove the concept".

#### Tables 

*RedditNews.csv:* two columns
The first column is the "date", and second column is the "news headlines".
All news are ranked from top to bottom based on how hot they are.
Hence, there are 25 lines for each date.

*DJIA_table.csv:* 
Downloaded directly from Yahoo Finance: check out the web page for more info.

*CombinedNewsDJIA.csv:*
To make things easier for my students, I provide this combined dataset with 27 columns.
The first column is "Date", the second is "Label", and the following ones are news headlines ranging from "Top1" to "Top25".
