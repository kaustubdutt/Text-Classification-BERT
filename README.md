Text Classification
Using Bert

Group 7

Emani Jones,
Yash Rajeev Karkhanis,
Srivatsa Srinivas Rusum,
Kaustub Dutt
Thirumala,
Ritesh Sabhapati Vermas

Abstract

In this project, we will classify a group of BBC articles into 5 predefined categories. We will use
the BERT embedding with a simple dense classifier and LSTM.

Introduction

80% of all information present today is unstructured and there is a growing need to organize that
information. One method to do that is text classification. Text classification is a natural language
processing technique that categorizes text into predefined categories based on their content. Text
classification allows us to take large text data and organize them into categories.

Data Description and Exploration

2225 BBC news articles are included in the dataset, and each of them is categorized into one of
the following five subcategories: business, entertainment, politics, sport, or technology.


Related Work

Sanghvi, D., Shah, K., and Patel, H. (2020), used KNN, random Forest, and Logistic Regression
models for the classification of BBC articles.
In this approach, TF-IDF vectors were used to generate vectors of the text and then various
machine learning algorithms were used to classify the text.

The words are transformed into a numerical representation using TF-IDF. TF-IDF determines the
relevancy of a word, by computing how frequently a word appears in the document (term
frequency) vs how much it appears in the entire corpus (inverse term frequency). Therefore, if a
word frequently appears in a document and frequently throughout the corpus as a whole, it is
unlikely to have a small TF-IDF score; however, if it rarely appears throughout the entire corpus,
it will be given an increased TF-IDF score. Below are the results of the classification of the TF-
IDF vectors.

This method's primary flaw is that the TF-IDF vector, which is based on word frequency, does
not take the text's context into account.

Model Accuracy

Logistic Regression 97%

Random Forest 93%

KNN 92%

Our Approach

To consider context, we propose using the BERT embedding along with sequential models such
as LSTM for classification.

The transformer architecture serves as the foundation for BERT, which stands for bidirectional
encoder representation from transforms. The word embedding is created by capturing the word's
context using an attention technique.


RNNs are machine learning models that take into account previous inputs. These are useful in
analyzing sequential data. The have two main drawbacks.

• Vanishing gradient problem

• Exploding gradient problem

LSTM (Long short-term memory): 
LSTMs are an extension of RNNs that have a memory cell
that is capable of selectively retaining and forgetting information over time. LSTMs address the
vanishing gradient problem and the short-term memory problem that are common in traditional
RNNs.


Data Preparation

We divided the data into sets for training, testing, and validation. The model was trained with a
train set, evaluated during training using a validation set, and accuracy metrics are generated
using a test set. These were kept constant across all models to get an accurate evaluation of the
models.


The categories which is out target variable were encoded using the below map.

Category Label

sport -->  0


business --> 1

politics --> 2

tech --> 3

entertainment --> 4

Baseline Model

We created a baseline model using TD-IDF vectors and random forest as mentioned in the
referenced paper. We considered only alphanumeric text and removed the English stop words as
a preprocessing step. Below are the results. The overall accuracy is 95%.


BERT Embedding

We now move onto the BERT embedding. To generate the BERT vectors, we used the
pretrained BERT model from hugging face. Since BERT considers context, we did not remove
stop words. The only preprocessing that was done was tokenization (splitting text into words),
lowercasing the text and removing special characters. Once the BERT vectors were generated,
we use two classification models.

Simple Dense Classifier

In this approach we stacked all the BERT word embedding and passed them through a simple
dense neural network to classify the text. The below screenshot shows the model architecture
with the dropout. This model was trained for 10 epochs with batch size of 16 and learning rate of
1e-4.

The overall accuracy is 96%.


LSTM Classifier

We used the BERT vectors along with LSTM to classify the text. The text is passed one word
vector at time through the LSTM for each of the articles. The below screenshot shows the model
architecture with the dropout. This model was trained for 25 epochs with batch size of 16 and
learning rate of 1e-4.

The overall accuracy is 97%.

Comparison

Model Accuracy

TF-IDF with random forest --> 95%

BERT with dense classifier --> 96%

BERT with LSTM --> 97%

As we can see the BERT embedding only slightly outperforms the TFIDF vectors by 1% or 2%
which is in the margin of error due to the randomness of the train test split. This may be due to
the fact we are classifying news articles which are quite large and written very clearly. These are
written in an easy-to-understand text while using the correct vocabulary. Hence, even the TFIDF
vector is able to accurately classify the text.


Challenges and Future Scope

BERT embeddings and LSTMs are deep learning methods. These deep learning algorithms are
computationally expensive and require a lot of computational resources to run. Considering the
high accuracy of the previous methods, deep learning algorithms may not be necessary to
classify the articles into categories. These methods are more useful in cases where the context
may not be easily captured using other simpler methods such as TF-IDF like sentiment analysis,
text summarization and spam detection.

Text classification is an extremely important task as the amount of textual data present is
increasing exponentially. Text classification allow us to classify and organize this data to enable
better storage and retrieval of information. It also has important applications in:

1. Online abuse detection
2. Fraud detection
3. Search engine optimization
4. CRM management
5. Fake news detection

References

Shah, K., Patel, H., Sanghvi, D. et al. A Comparative Analysis of Logistic Regression, Random
Forest and KNN Models for the Text Classification. Augment Hum Res 5, 12 (2020). 
https://doi-org.libezproxy2.syr.edu/10.1007/s41133-020-00032-0


