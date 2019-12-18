# e-commerce-fraud
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn import metrics
import json

dataset = pd.read_csv('customersdata.csv')


#taking care of missing values
dataset = dataset.fillna("Unknown") 
#creating dependent and independent variables  
X = dataset.drop(['fraudulent'],axis=1)
X_labels=['customer_customerEmail','customer_customerPhone','customerDevice','customerIPAddress','customerBillingAddress','orders_orderId',
            'orders_orderAmount','orders_orderState','orders_orderShippingAddress','paymentMethods_paymentMethodId','paymentMethods_paymentMethodRegistrationFailure','paymentMethods_paymentMethodType',
            'paymentMethods_paymentMethodProvider','paymentMethods_paymentMethodIssuer','transactions_transactionId','transactions_orderId','transactions_paymentMethodId','transactions_transactionAmount',
            'transactions_transactionFailed']
y = dataset.iloc[:,0].values
y = pd.get_dummies(y)
X = pd.get_dummies(X)

#creating train and test variables
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

#fitting to Randomforestclassifier model
clf = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)

# Training the classifier
clf.fit(X_train, y_train)




# Printing the name and importance of each feature
for feature in zip(X_labels, clf.feature_importances_):
    print(feature)
    

    
# features that have an importance of more than 0.0005
sfm = SelectFromModel(clf, threshold=0.0005)

# Training the selector
sfm.fit(X_train, y_train)



X_important_train = sfm.transform(X_train)
X_important_test = sfm.transform(X_test)

# Creating a new random forest classifier for the most important features
clf_important = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)

# Training the new classifier on the new dataset containing the most important features
clf_important.fit(X_important_train, y_train)

# Applying The Full Featured Classifier To The Test Data
y_pred = clf.predict(X_test)

# Applying The Full Featured Classifier To The Test Data
y_important_pred = clf_important.predict(X_important_test)


#accuracy of all features
print ("Overall Accuracy:", round(metrics.accuracy_score(y_test, y_pred), 3))

#accuracy of limited features
print ("Overall Accuracy:", round(metrics.accuracy_score(y_test, y_important_pred), 3))


