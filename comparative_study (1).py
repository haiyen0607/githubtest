# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 23:04:08 2021

@author: Admin
"""

import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn.svm import SVC     ### SVM for classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression

from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.ensemble import RandomForestClassifier 
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import StackingClassifier

import random
import numpy as np
import math
from collections import Counter
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn import tree

# instantiate labelencoder object
from sklearn.tree import export_text

from IPython.display import Image  
from sklearn.tree import export_graphviz
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.svm import SVC
from datetime import datetime







def sap_xep_lai_cot():
    df = pd.read_csv('c:\\baitap_datamining\\Leukemia_3c.csv', header = None ) 
    df.iloc[0,2000]
    cols = list(df.columns)
    c1=  cols[-1:]     ## lay phan tu cuoi cung cuar list
    c2 = cols[:-1]     ##  loai bo phan tu cuoi cung cuar list
    cols = cols[-1:] + cols[:-1]
    df = df[cols]
    df.columns.values[0] = "class"   # thay ten bang class
    
    ### Chuyen Dang Category thanh` 0-1;
    # Categorical boolean mask
    categorical_feature_mask = df.dtypes==object
   # filter categorical columns using mask and turn it into a list
    categorical_cols = df.columns[categorical_feature_mask].tolist()
    
    le = LabelEncoder()
    
    df[categorical_cols] = df[categorical_cols].apply(lambda col: le.fit_transform(col))
    df[categorical_cols].head(10)   
    df.to_csv('c:\\baitap_datamining\\Leukemia_3c1.csv', index= False)
    
    

def stratify_sampling(df):
    r,c = df.shape
    X = df.iloc[:,df.columns !='class']
    features = X.columns.tolist()
    y = df[['class']]

   # _random_state =  random.randint(0,100000)

    
    
  #  print (_random_state)
    acc_DT =list()
    acc_RF= list()
    
    for i in range (10):
        Train_x, Test_x, Train_y, Test_y = train_test_split(X, y, train_size=0.7)

        #Train_x, Test_x, Train_y, Test_y = train_test_split(X,  y, test_size=0.3, random_state = random.randint(0,100000))
        #Train_x, Test_x, Train_y, Test_y = train_test_split(X,  y, stratify=y, test_size=0.4, random_state = random.randint(0,100000))
        model = DecisionTreeClassifier()
        
        model.fit(Train_x, Train_y)
        
    
        predictions = model.predict(Test_x)
        print("Accuracy Cay Quyet Dinh:",metrics.accuracy_score(Test_y, predictions))
        acc_DT.append(metrics.accuracy_score(Test_y, predictions))

        rf_model = RandomForestClassifier(n_estimators=10, max_features= int(math.sqrt(c))+1)

        rf_model.fit(Train_x,Train_y.values.ravel())
        pred_y = rf_model.predict(Test_x)
        print("Accuracy RandomForest:",metrics.accuracy_score(Test_y, pred_y))
        acc_RF.append(metrics.accuracy_score(Test_y, pred_y))

        rf_model.bootstrap()[0]
       # print(metrics.confusion_matrix(Test_y,predictions))
       # print(metrics.classification_report(Test_y,predictions))
        #print(metrics.accuracy_score(Test_y, predictions))
    
    
    print ("Do Chinh Xac Trung Binh Cay quyet dinh: ", sum(acc_DT))
    print ("Do CHinh Xac Random forest: ", sum(acc_RF))
    return 


def comparative(df):
    Y = df[['class']]
    X = df.iloc[:,df.columns !='class']
    r,c = df.shape
    acc_DT =list()
    acc_SVM =list()
    acc_Navie = list()
    acc_Bagging = list()
    acc_RF = list()

    time_DT =list()
    time_SVM =list()
    time_Naive =list()
    time_Bagging =list()
    time_RF =list()
    i=0
    for i in range(10):
        print ("Chay Lan thu: ", i)

# split data
        X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y,  stratify=Y, train_size=0.7)
        
# decision tree        
        start = datetime.now()

        model = DecisionTreeClassifier()        
        model.fit(X_Train, Y_Train)            
        predictions = model.predict(X_Test)
        end = datetime.now() -start
        time_DT.append(end)

        print("Accuracy Cay Quyet Dinh:",metrics.accuracy_score(Y_Test, predictions))
        acc_DT.append(metrics.accuracy_score(Y_Test, predictions))
        

# SVM
        start = datetime.now()

        svclassifier = SVC(kernel='rbf')
        svclassifier.fit(X_Train, Y_Train.values.ravel())
        y_pred = svclassifier.predict(X_Test)
        end = datetime.now() -start
        time_SVM.append(end)

        print("Accuracy SVM:",metrics.accuracy_score(Y_Test, y_pred))
        acc_SVM.append(metrics.accuracy_score(Y_Test, y_pred))
        

# naive classifier
        start = datetime.now()
        model_navie = GaussianNB()
      #  model_navie = MultinomialNB()        
        model_navie.fit(X_Train, Y_Train.values.ravel()) 
        prediction = model_navie.predict(X_Test) 
        end = datetime.now() -start
        time_Naive.append(end)
        print("Naive Bayes: ", metrics.accuracy_score(Y_Test, prediction))
        acc_Navie.append(metrics.accuracy_score(Y_Test, prediction))
       
    
# bagging
       # base_cls = SVC()
       # base_cls = KNeighborsClassifier(n_neighbors=5)
       # base_clas = GaussianNB()  
        start = datetime.now()
        
        base_cls = DecisionTreeClassifier()
        
        #model = BaggingClassifier(base_estimator = base_cls, n_estimators = 100)
        model_bagging = BaggingClassifier(estimator = base_cls, n_estimators = 100)
       
        model_bagging.fit(X_Train, Y_Train.values.ravel())
        predictions = model_bagging.predict(X_Test)
        
        end = datetime.now() -start
        time_Bagging.append(end)
                
        acc = metrics.accuracy_score(Y_Test, predictions)    
        acc_Bagging.append(acc)        
        print ("Bagging: ",acc)     
        
        
#random forest
        start = datetime.now()

        rf_model = RandomForestClassifier(n_estimators=100, max_features= int(math.sqrt(c))+1)
        rf_model.fit(X_Train,Y_Train.values.ravel())
        pred_y = rf_model.predict(X_Test)
           
        end = datetime.now() -start
        time_RF.append(end)
                
        accRF = metrics.accuracy_score(Y_Test, pred_y)
        acc_RF.append(accRF)
        print ("Accuracy RF: ",accRF) 
    
        

    results =[]
    results.append(acc_DT)
    results.append(acc_SVM)
    results.append(acc_Navie)
    results.append(acc_Bagging)
    results.append(acc_RF)
    # results.append(acc_RandomForest)
    # results.append(acc_Boosting)
    # results.append(acc_Stacking)


    #names =('decision tree','randomforest','boosting','stacking')
    names =('Decision tree', 'SVM', 'Navie bayes','Bagging','Random forest')
    fig = plt.figure()
    fig.suptitle('Algorithm Comparison')
    
   # ax = fig.add_subplot(111)
#    plt.boxplot(results)
    #plt.boxplot(results, labels=names, showmeans = True)
    plt.boxplot(results, labels=names)
    plt.ylabel('Accuracy')    

    #ax.set_xticklabels(names)
    plt.show()            
    
    print ("Results")
    print ("decision tree")
    print (np.mean(acc_DT))    ## gia tri trung binh  
    print (np.std(acc_DT))     ## Do lech chuan   
    
    
    print ("SVM")
    print (np.mean(acc_SVM))    ## gia tri trung binh  
    print (np.std(acc_SVM))     ## Do lech chuan   
    
    print ("Naive Bayes")
    print (np.mean(acc_Navie))    ## gia tri trung binh  
    print (np.std(acc_Navie))     ## Do lech chuan   
    
    print ("Bagging")
    print (np.mean(acc_Bagging))    ## gia tri trung binh  
    print (np.std(acc_Bagging))     ## Do lech chuan   
    
    print ("Random forest")
    print (np.mean(acc_RF))    ## gia tri trung binh  
    print (np.std(acc_RF))     ## Do lech chuan   
    
    print ("Time")
    print (np.mean(time_DT))
    print (np.mean(time_SVM))
    print (np.mean(time_Naive))
    print (np.mean(time_Bagging))
    print (np.mean(time_RF))
    
    
'''
    CNS1  ::  data ok
    Colon  :  data ok 
    Brain   : data ok
    Prostate: data ok 
    LungCancer: data ok
    Adenocarcinoma
    Leukemia4c  
'''  
    
def load_data():
    
    x=2
    #Breast2classes
    df = pd.read_csv('c:\\CSLT\\DLBCL.csv', header = 0 ) 
    r,c = df.shape
    df.columns.values[0] = "class" 
    
    comparative(df)
    
    
    
    




def main():
    x=1
    load_data()
    
    
if __name__ =="__main__":
    main()