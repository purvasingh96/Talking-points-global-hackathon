# Talking-points-global-hackathon

# Problem Description

# Data Source & Description

# Source

We have gathered the data for training our model from Kaggle's dataset [Daily News for Stock Market Prediction](https://www.kaggle.com/aaron7sun/stocknews)

## Description:

### Data
There are two channels of data provided in this dataset:

**1. News data:** The data-set owner has crawled historical news headlines from Reddit WorldNews Channel (/r/worldnews). They are ranked by reddit users' votes, and only the top 25 headlines are considered for a single date.

**2. Stock data:** Dow Jones Industrial Average (DJIA) is used to "prove the concept".

### Tables 

*RedditNews.csv:* two columns
The first column is the "date", and second column is the "news headlines".
All news are ranked from top to bottom based on how hot they are.
Hence, there are 25 lines for each date.

*DJIA_table.csv:* 
Downloaded directly from Yahoo Finance: check out the web page for more info.

*CombinedNewsDJIA.csv:*
To make things easier for my students, I provide this combined dataset with 27 columns.
The first column is "Date", the second is "Label", and the following ones are news headlines ranging from "Top1" to "Top25".

# Deep Learning Algorithm for Talking Points

The deep-learning algorithm is implemented using [Pytorch](https://pytorch.org).

## Step 1: Get the data

We have unzipped ---> # TODO

## Step 2: Implement pre-processing functions

Here, we implement thw following 2 pre-processing functions - 

1. Look-up Table
2. Tokenize Punctuation

### Look-up Table
To create a word embedding, we first need to transform the words to ids. In this function, we have create two dictionaries:<br>

1. Dictionary to go from the words to an id, we'll call `vocab_to_int`
2. Dictionary to go from the id to word, we'll call `int_to_vocab`

Return these dictionaries in the following tuple `(vocab_to_int, int_to_vocab)`


### Tokenize Punctuation

We'll be splitting the stock news into a word array using spaces as delimiters. However, punctuations like periods and exclamation marks can create multiple ids for the same word. For example, "rise" and "rise!" would generate two different word ids.<br><br>
The function `token_lookup` returns a dict that will be used to tokenize symbols like "!" into "||Exclamation_Mark||". Some more examples are described below-<br><br>
Period ( . )<br>
Comma ( , )<br>
Quotation Mark ( " )<br>
Semicolon ( ; )<br>
Exclamation mark ( ! )<br>
Question mark ( ? )<br>
Left Parentheses ( ( )<br>
Right Parentheses ( ) )<br>
Dash ( - )<br>
Return ( \n )<br><br>
This dictionary will be used to tokenize the symbols and add the delimiter (space) around it. This separates each symbols as its own word, making it easier for the neural network to predict the next word. 

# Build the Neural Network

## Input
Let's start with the preprocessed input data. We'll use [TensorDataset](http://pytorch.org/docs/master/data.html#torch.utils.data.TensorDataset) to provide a known format to our dataset; in combination with [DataLoader](http://pytorch.org/docs/master/data.html#torch.utils.data.DataLoader), it will handle batching, shuffling, and other dataset iteration functions.

We have created data with TensorDataset by passing in feature and target tensors. Then created a DataLoader as usual.
```
data = TensorDataset(feature_tensors, target_tensors)
data_loader = torch.utils.data.DataLoader(data, 
                                          batch_size=batch_size)
```

## Batching
We have batched `words` data into chunks of size `batch_size` using the `TensorDataset` and `DataLoader` classes.<br>

For example, say we have these as input:<br>
```
words = [1, 2, 3, 4, 5, 6, 7]
sequence_length = 4
```

Our first `feature_tensor` contains the values:<br>
```
[1, 2, 3, 4]
```
And the corresponding `target_tensor` is just the next "word"/tokenized word value:<br>
```
5
```
This continues with the second `feature_tensor`, `target_tensor` being:<br>
```
[2, 3, 4, 5]  # features
6             # target
```
---
## The Neural Network
We have implemented an RNN using PyTorch's [Module class](http://pytorch.org/docs/master/nn.html#torch.nn.Module) along with LSTM. The following functions completed our RNN model -<br>
 - `__init__` - The initialize function. 
 - `init_hidden` - The initialization function for an LSTM hidden state
 - `forward` - Forward propagation function.
 

**The output of this model is the *last* batch of word scores** after a complete sequence has been processed. That is, for each input sequence of words, we only want to output the word scores for a single, most likely, next word.

















