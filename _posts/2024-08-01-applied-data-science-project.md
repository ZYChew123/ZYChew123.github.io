---
layout: post
author: Chew Zheng Yang
title: "Improve Public Health Response to Future Pandemic"
categories: ITD214
---
## Project Background
In the early 2020, Covid-19 pandemic took the world by storm, and reshaped the world in every way possible. The widespread diseases have highlighted both vulnerablilities and strengths within the global healthcare systems, society and goverment structures. It has become crucial to assess our preparedness and response mechanisms to fortify ourselves against future pandemics.

Fast forward earlier this year, we also multiple flu related reports surging across different nations, most notable one would be the passing of artist Barbie Hsu in Japan. Such news made me wonder, after the recent Covid-19 pandemic, how well prepared are we in facing the next pandemic?

Business goal: Improve Public Health Response to Future Pandemic

Objective: Analyze Twitter data to gauge public sentiment and awareness regarding pandemic 


## Work Accomplished

### Data Understanding
| Label          | Description                                                                                      | Include or exclude?                              |
|----------------|--------------------------------------------------------------------------------------------------|--------------------------------------------------|
| UserName       | Username of the twitter account (masked)                                                         | Exclude (will not be used in sentiment analysis) |
| ScreenName     | Name of the account (masked)                                                                     | Exclude (will not be used in sentiment analysis) |
| Location       | User’s location                                                                                  | Exclude (will not be used in sentiment analysis) |
| TweetAt        | Tweet creation date2/3/2020 - 16/3/2020                                                          | Exclude (will not be used in sentiment analysis) |
| Original Tweet | Content of the tweet                                                                             | Include                                          |
| Sentiment      | Sentiment classification of the tweet, generalised the sentiment by removing “extreme” sentiment | Include (for validation only)                    |

About the data: <br>
The data above was pulled from twitter with manual tagging on sentiment regarding Covid-19 over the period of 2/3/2020 to 16/3/2020, which is around the lockdown period during the early stages of Covid-19. twitter account user name and name was masked for privacy purpose. Sentiment was generalised ("Extremely Negative" to "Negative" and vice versa) to create a more general picture of trends or insights <br>
<br>
dataset shape: 3798 x 6

### Data Preparation
import library
```python
# import library
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report, confusion_matrix
```

To do sentiment analysis, After we loaded the data, then we only keep "original tweet" and "Sentiment" column
``` python
#load training data
df = pd.read_csv('dataset/Corona_NLP_train.csv',encoding = 'latin-1')
#extract relevant data only
df = df[['OriginalTweet','Sentiment']]
```
Generalise the sentiments to create a more general picture of trends or insights
``` python
#replace "Extremely Negative" with "Negative" and vise versa
df['Sentiment'] = df['Sentiment'].replace({'Extremely Negative': 'Negative', 'Extremely Positive': 'Positive'})
```
check for null values for Original tweet column and sentiment column
```python
df.isnull.any()
```

Split the data into training data(80%) and validation data (20%)
```python
# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(df['OriginalTweet'], df['Sentiment'], test_size=0.2, random_state=42)
```
tokenize the text
```python
# Tokenize the text
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_val_seq = tokenizer.texts_to_sequences(X_val)
```
Pad the sequence to ensure all tokenized text are equal length
```python
# Pad sequences to have the same length
max_len = 100  
X_train_pad = pad_sequences(X_train_seq, maxlen=max_len)
X_val_pad = pad_sequences(X_val_seq, maxlen=max_len)
```
Encode the sentiments into numerical values, where 1 = negative, 2 = neutral, 3 = positive
```python
# Convert sentiment labels to numerical values
sentiment_mapping = {sentiment: i for i, sentiment in enumerate(df['Sentiment'].unique())}
y_train_num = np.array([sentiment_mapping[sentiment] for sentiment in y_train])
y_val_num = np.array([sentiment_mapping[sentiment] for sentiment in y_val])
```
### Modelling
Two model was used for modelling, 
- Logisitic Regression
- Recurrent Neural Network

#### Logistic Regression
import Logistic regression related library

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
```
define logistic regression model
```python
#define logistic regression model
vectorizer = TfidfVectorizer(max_features=5000)  # Limit vocabulary size
X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)

model = LogisticRegression(max_iter=1000) # Increased max_iter to ensure convergence
model.fit(X_train_vec, y_train)
```
Make predictions and evaluate the model
```python
# Make predictions
y_pred = model.predict(X_val_vec)

# Evaluate the model
print(classification_report(y_val, y_pred, target_names=df['Sentiment'].unique()))
print(confusion_matrix(y_val, y_pred))
```
#### Recurrent Neural Network

import RNN related model
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
```
Define Model, softmax was used as there's 3 categorical output
```python
# Define the model
model = Sequential()
model.add(Embedding(5000, 128, input_length=max_len))
model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(len(sentiment_mapping), activation='softmax'))
```
summarise the model
```python
model.summary()
```
model: "sequential"
| layer                 | output shape | param #      |
|-----------------------|--------------|--------------|
| embedding (embedding) | ?            | 0 (unbuilt)  |
| lstm                  | ?            | 0  (unbuilt) |
| dense                 | ?            | 0 (unbuilt)  |

Compile and train the Model, using epoch = 5
```python
# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# Train the model
model.fit(X_train_pad, y_train_num, epochs=5, batch_size=32, validation_data=(X_val_pad, y_val_num))
```
Evaluate the model
```python
# Evaluate the model
print(classification_report(y_val_num, y_pred, target_names=df['Sentiment'].unique()))
print(confusion_matrix(y_val_num, y_pred))
```

### Evaluation

The performance of both model was evaluated by it's accuracy statistics and the confusion matrix

| Logistic Regression|             |        |          |         | Recurrent Neural Network |           |        |          |         |
|---------------------|-----------|--------|----------|---------|--------------------------|-----------|--------|----------|---------|
|                     | precision | recall | F1 score | support |                          | precision | recall | F1 score | support |
| neutral             | 0.79      | 0.814  | 0.80     | 3062    | neutral                  | 0.84      | 0.79   | 0.82     | 1553    |
| positive            | 0.76      | 0.63   | 0.69     | 1553    | positive                 | 0.89      | 0.87   | 0.88     | 3617    |
| negative            | 0.81      | 0.85   | 0.83     | 3617    | negative                 | 0.84      | 0.88   | 0.86     | 3062    |
| Overall accuracy    | 80%       |        |          |         | Overall accuracy         | 86%       |        |          |         |

***Table 1*** Accuracy Statistics
1) **Overall Accuracy**<br>
  RNN model has higher overall accuracy compared to Log Regression model.<br>

2) **Precision**<br>
  By comparing all three sentiments, RNN model consistently shows higher precision values accross all sentiments.<br>

3) **Recall** <br>
  RNN model has higher recall values for positive and negative sentiments, however Logistic regression model has a slightly better recall for neutral sentiment. <br>

4) **F1 Score**<br>
  RNN model has higher F1 score for all three sentiments, indicating better balance between precision and recall as compared to logistic regression model.

| Logistic Regression        |          |         |          | Recurrent Neural Network   |          |         |          |
|----------------------------|----------|---------|----------|----------------------------|----------|---------|----------|
| Document class\ prediction | Neutral | positive | Negative | Document class\ prediction | Neutral | positive | Negative |
| Neutral                    | 2487     | 160     | 415      | Neutral                    | 1230     | 127     | 196      |
| Positive                   | 266      | 984     | 303      | Positive                   | 118      | 3164    | 335      |
| Negative                   | 387      | 149     | 3081     | Negative                   | 117      | 251     | 2694     |

***Table 2*** Confusion Matrix

**Neutral Class**<br>
  Logistic Regression has more correct predictions on neutral class as compared to RNN model, but however, the RNN model has fewer incorrect predictions. <br>

**Positive Class**<br>
  RNN model outperforms logistic regression model significantly in prediction positive class with higher predictions and fewer incorrect predictions. <br>

**Negative Class**<br>
  Logistic Regression model has more correct predictions for negative calss as compared to RNN, however RNN makes fewer incorrect predictions. <br>

Overall, the Recurrent Neural Network (RNN) model demonstrates better performance in terms of overall accuracy, precision, recall, and F1 scores across most classes compared to the Logistic Regression (LR) model. <br>

## Recommendation and Analysis
Recommended to use RNN to accurately gauge public sentiment on Covid-19

To gauge the public sentiment on Pandemic, recommended to train the data with sentiments from different period of Covid-19 (Lockdown, vaccine launch and New normal)


Use both topic modelling and sentiment analysis to understand public’s opinion on a topic and gauge their response on the particular topic. 

## AI Ethics
Discuss the potential data science ethics issues (privacy, fairness, accuracy, accountability, transparency) in your project. 

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Fusce bibendum neque eget nunc mattis eu sollicitudin enim tincidunt. Vestibulum lacus tortor, ultricies id dignissim ac, bibendum in velit. Proin convallis mi ac felis pharetra aliquam. Curabitur dignissim accumsan rutrum. In arcu magna, aliquet vel pretium et, molestie et arcu. Mauris lobortis nulla et felis ullamcorper bibendum. Phasellus et hendrerit mauris. Proin eget nibh a massa vestibulum pretium. Suspendisse eu nisl a ante aliquet bibendum quis a nunc. Praesent varius interdum vehicula. Aenean risus libero, placerat at vestibulum eget, ultricies eu enim. Praesent nulla tortor, malesuada adipiscing adipiscing sollicitudin, adipiscing eget est.

## Source Codes and Datasets
Upload your model files and dataset into a GitHub repo and add the link here. 
