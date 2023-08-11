import numpy as np 
import random
import pandas as pd 
from sklearn.metrics import accuracy_score
import string
import openai
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import streamlit as st


st.title("Data Virtual Agent Implementation")
st.write("First, we train and define our Sentiment Analysis Model. This may take a few minutes to load, as our Model is using a large number of samples to train")

# Setup dataset for training
review = pd.read_csv(r'ChatGPT related Twitter Dataset.csv')

positive = review[review['labels'] == 'good'][:10000]
negative = review[review['labels'] == 'bad'][:10000]
neutral = review[review['labels'] == 'neutral'][:10000]

review_bal = pd.concat([positive, negative, neutral])

# Set up the Sentiment Dataset
sentiment_dataset = review_bal

# Remove punctuation from datasets
sentiment_dataset['tweets'] = sentiment_dataset['tweets'].str.replace('[{}]'.format(string.punctuation), '')

#Setup the Train/Test Split for the SA Model
from sklearn.model_selection import train_test_split

train,test = train_test_split(sentiment_dataset,test_size =0.5,random_state=42)

train_x, train_y = train['tweets'], train['labels']
test_x, test_y = test['tweets'], test['labels']

#Implement Stop Words, vectorize data
global tfidf
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(stop_words='english')
train_x_vector = tfidf.fit_transform(train_x)

# also fit the test_x_vector
test_x_vector = tfidf.transform(test_x)

#Transform Train_x data for later testing. 

pd.DataFrame.sparse.from_spmatrix(train_x_vector,
                                  index=train_x.index,
                                  columns=tfidf.get_feature_names_out())

# Store the training and testing accuracy during each epoch
train_accuracy = []
test_accuracy = []

#Fit the Data to an SVC Model
global svc
from sklearn.svm import SVC
svc = SVC(kernel='linear')
svc.fit(train_x_vector, train_y)

# Evaluate the model on training and testing data
train_predictions = svc.predict(train_x_vector)
test_predictions = svc.predict(test_x_vector)

# Calculate accuracy scores
train_acc = accuracy_score(train_y, train_predictions)
test_acc = accuracy_score(test_y, test_predictions)

train_accuracy.append(train_acc)
test_accuracy.append(test_acc)

st.write("The results of the completed Train/Test split are as follows:")
st.bar_chart(data = [train_accuracy, test_accuracy], x = None, y = None, width = 0, height = 0, use_container_width = True)
st.caption ("training score on the left, testing score on the right.")

st.write("The accuracy score for the testing secton was:")
st.caption(svc.score(test_x_vector, test_y))

#Display F1-score
from sklearn.metrics import f1_score
st.write("The F1 score:")
st.caption(f1_score(test_y,svc.predict(test_x_vector),
          labels = ['good', 'neutral', 'bad'],average=None))

#Display Classification Report
from sklearn.metrics import classification_report
st.write("The Classification Report:")
st.caption(classification_report(test_y,
                            svc.predict(test_x_vector),
                            labels = ['good', 'neutral', 'bad']))

#Declared Globals for later use in functions
global evaluation_acc
global sentiment_counts
global evaluation_f1
global evaluation_classification

#Function for sentiment analysis
def sa_model_evaluation(tfidf, svc):

    evaluation_sample = review.sample(100)
    evaluation_sample = evaluation_sample.dropna(subset=['tweets'])
    # Remove punctuation from datasets
    evaluation_sample['tweets'] = evaluation_sample['tweets'].str.replace('[{}]'.format(string.punctuation), '')
        
    evaluation_x,evaluation_y = evaluation_sample['tweets'],evaluation_sample['labels']

    #vectorize data for predictions
    evaluation_vectorized = tfidf.transform(evaluation_x.values.astype('U'))

    #Predict Sentiments for data
    evaluation_predictions = svc.predict(evaluation_vectorized)

    evaluation_acc = accuracy_score(evaluation_y, evaluation_predictions)

    # Count sentiment frequencies
    sentiment_counts = pd.Series(evaluation_predictions).value_counts(normalize=True)

    #Calculate F1 Score
    evaluation_f1 = f1_score(evaluation_y,evaluation_predictions,
                                       labels = ['good', 'neutral', 'bad'],average=None)
    #Create Classification Report
    evaluation_classification = classification_report(evaluation_y,
                                                        evaluation_predictions,
                                                        labels = ['good', 'neutral', 'bad'])

            
    st.caption(f"Accuracy score: '{evaluation_acc}'")
    st.caption("Sentiment Counts")
    st.caption(sentiment_counts)
    return evaluation_acc, sentiment_counts, evaluation_f1, evaluation_classification

def explain_results_with_gpt3(evaluation_acc, sentiment_counts, evaluation_f1, evaluation_classification):
    # Construct a prompt to explain the results.
    f1_scores_str = ', '.join([f"{score:.2f}" for score in evaluation_f1])
    
    prompt = f"A summary of the results:\n"
    
    prompt += f"The sentiment analysis model achieved an accuracy of {evaluation_acc*100:.2f}%. " \
             f"The F1 scores for each class are(in order of good, bad, neutral): {f1_scores_str}. " \
             f"The distribution of sentiments in the dataset is as follows:\n"

    for sentiment, percentage in sentiment_counts.items():
        prompt += f"- {sentiment}: {percentage*100:.2f}%\n"

    prompt += "Explain these results and suggest ways to improve them."
    # Use the GPT-3 model to generate a response.
    response = openai.Completion.create(
      engine="davinci",
      prompt=prompt,
      temperature=0.5,
      max_tokens=150
    )

    # Print the response
    st.caption(response.choices[0].text.strip())

st.write("After training our model, we now attempt to apply it to the program: ")
st.caption(f"Welcome to the Sentiment Analysis Model Evaluator")
st.caption(f"Model Status: Loaded")
st.caption(f"-------------------------------------------------------")
st.caption(f"Select an option to continue:")
st.caption(f"Type '1' to Run Model Evaluation")
st.caption(f"Type '2' to Discuss the results with the Virtual Assistant")

st.write("The user would first select to run the model evaluation:")
st.caption(f"-------------------------------------------------------")
st.caption(f"You have selected: Run Model Evaluation")
st.caption(f"-------------------------------------------------------")
evaluation_acc, sentiment_counts, evaluation_f1, evaluation_classification = sa_model_evaluation(tfidf, svc)

st.write("After running an evaluation, the user could then send the results to the Virtual Assistant to receive an explanation.")
st.caption(f"-------------------------------------------------------")

keyinput = 0
while (keyinput == 0):
    #Get API Key from user
    st.write("In order to continue the demonstration, please input a valid OpenAI API Key. An Error Message will appear later in the code until a valid key is applied:")
    key = st.text_input("API Key", "[Insert API Key Here]")
    keyinput = 1
st.write("The remaining code may take a few minutes to load.")
openai.api_key = key

explain_results_with_gpt3(evaluation_acc, sentiment_counts, evaluation_f1, evaluation_classification)

st.write("Future iterations could, from there, implement the ability to discuss the results with the virtual agent by allowing it to respond to user input questions, but would require a more extensively trained agent than this demo was able to implement.")

st.write("""Observations:

1) Implementation of currently available Virtual Agents and Generative AI programs directly into the process of analyzing the model is not currently possible without designing one with that capability from scratch. All of the current VA and GenAI programs are designed specifically as language models Power Virtual Agent, Microsoft's VA program, came closest to achieving the desired goal out of the box, but lacks functionality that would allow it to function as requested in this case study.""")

st.write("2) GPT's responses are often erratic. An untrained GPT Model lacks the 'knowledge' required to understand the data it's looking at. A Trained model would be able to produce better results.")
