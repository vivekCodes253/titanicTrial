import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('train.csv')
dataset = dataset[dataset.Embarked.notnull()]
#mean as nan strategy for age
dataset.fillna(dataset.mean(), inplace=True)

X = dataset.iloc[:, [2,4,5,11]]
X['Family'] = dataset['Parch']+dataset['SibSp'] # Combining Parch and SibSP to one column called family
# Removing rows with missing embarkment values
# onehotencoder not working well so do a  personal encoding, 2 columns for C Q , S is the extra
X['EmbC'] = (X['Embarked']=='C')*1
X['EmbQ'] = (X['Embarked'] == 'Q')*1



X = X.iloc[:,X.columns!='Embarked']
Xt = X
X = X.values
y = dataset.iloc[:, 1].values



from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


#----------------------------------RANDOM FOREST   ------------------ 75%
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'gini', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cmSVM = confusion_matrix(y_test, y_pred)



#-------------------------------------ANN-----------------------------


import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 12, init = 'uniform', activation = 'relu', input_dim = 6))

# Adding the second hidden layer
#classifier.add(Dense(output_dim = 150, init = 'uniform', activation = 'sigmoid'))

# Adding the third hidden layer
classifier.add(Dense(output_dim = 80, init = 'uniform', activation = 'relu'))

# Adding the fourth hidden layer
classifier.add(Dense(output_dim = 12, init = 'uniform', activation = 'sigmoid'))


# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
cmANN = confusion_matrix(y_test, y_pred)





print('\n Confusion Matrix using ANN') 
print(cmANN)
print('\n Confusion Matrix using SVM') 
print(cmSVM)







