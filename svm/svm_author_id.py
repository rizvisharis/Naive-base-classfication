#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("/Users/rizvisharis/ud120-projects/tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


def SVMAccuracy(features_train, labels_train, features_test, labels_test):
    """ compute the accuracy of your Naive Bayes classifier """
    ### import the sklearn module for GaussianNB
    from sklearn.svm import SVC

    # features_train = features_train[:int(len(features_train) / 100)]
    # labels_train = labels_train[:int(len(labels_train) / 100)]
    ### create classifier
    clf = SVC(kernel='rbf', C=10000)


    ### fit the classifier on the training features and labels
    #TODO

    t0 = time()
    clf.fit(features_train, labels_train)
    print ("training time:", round(time() - t0, 3), "s")

    ### use the trained classifier to predict labels for the test features
    t0 = time()
    pred = clf.predict(features_test)
    print("pedicting time:", round(time() - t0, 3), "s")
    print("Prediction of element 10th, 26th and 50th are:",pred[10],pred[26],pred[50])
    print("number of events predicted in chris class is",sum(clf.predict(features_test) == 1))



    ### calculate and return the accuracy on the test data
    ### this is slightly different than the example,
    ### where we just print the accuracy
    ### you might need to import an sklearn module
    from sklearn.metrics import accuracy_score
    print (accuracy_score(pred, labels_test))
    accuracy = accuracy_score(pred, labels_test)
    return accuracy
#########################################################
if __name__ == '__main__' :
    SVMAccuracy(features_train, labels_train, features_test, labels_test)


