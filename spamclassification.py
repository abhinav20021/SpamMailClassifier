#IMPORTING DEPENDENCIES AND LIBRARIES
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer #To convert text data into numeric data.
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#DATA COLLECTION AND PRE PROCESSING
# loading the data from csv file to a pandas Dataframe.
raw_mail_data = pd.read_csv('/content/mail_data.csv')
# replace the null values in dataset with a null string.
mail_data = raw_mail_data.where((pd.notnull(raw_mail_data)),'')
# checking the number of rows and columns in the dataframe.
mail_data.shape

#LABEL ENCODING
# Encoding spam mail as 0 and ham mail as 1.
mail_data.loc[mail_data['Category'] == 'spam', 'Category',] = 0
mail_data.loc[mail_data['Category'] == 'ham', 'Category',] = 1
# separating the data as texts and label
X = mail_data['Message']
Y = mail_data['Category']
#Splitting the data into training data & test data.
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

#FEATURE EXTRACTION
#transform the text data to feature vectors that can be used as input to the Logistic regression
feature_extraction = TfidfVectorizer(min_df = 1, stop_words='english', lowercase = 1)
X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)
# convert Y_train and Y_test values as integers as they are of object data type.
Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')

#TRAINING THE MODEL USING LOGISTIC REGRESSION.
model = LogisticRegression()
# training the Logistic Regression model with the training data
model.fit(X_train_features, Y_train)

#EVALUATION OF TRAINED MODEL
# prediction on training data
prediction_on_training_data = model.predict(X_train_features)
accuracy_on_training_data = accuracy_score(Y_train, prediction_on_training_data)
print('Accuracy on training data : ', accuracy_on_training_data)
# prediction on test data
prediction_on_test_data = model.predict(X_test_features)
accuracy_on_test_data = accuracy_score(Y_test, prediction_on_test_data)
print('Accuracy on testing data : ', accuracy_on_test_data)
#Building a Predictive System.
input_mail = ["Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's"]
# convert input text to feature vectors
input_data_features = feature_extraction.transform(input_mail)

#MAKING PREDICTION
prediction = model.predict(input_data_features)
print(prediction)
if (prediction[0]==1):
  print('This is a Ham mail')
else:
  print('This is a Spam mail')
