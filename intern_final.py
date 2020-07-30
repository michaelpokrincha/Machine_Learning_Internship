# -*- coding: utf-8 -*-
"""
Created on Sat May  9 11:21:41 2020

@author: pokim
"""
#List of Kaggle notebooks used for graphs and tuned parameters
#https://www.kaggle.com/cnic92/beat-the-stock-market-the-lazy-strategy
#https://www.kaggle.com/cnic92/200-financial-indicators-of-us-stocks-20142018/kernels
#https://www.kaggle.com/danofer/starter-financial-data

import seaborn as sns
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from sklearn.metrics import plot_precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import RFE
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import StratifiedKFold

df = pd.read_csv('2017_Financial_data.csv',index_col = 0)
print("Orignal shape of the dataframe: ", df.shape)
#Data classes without any cleaning
sns.barplot(np.arange(len(df['Class'].value_counts())), df['Class'].value_counts(),palette="RdBu")
plt.title('CLASS DIFFERENCE FULL DATASET', fontsize=25)
plt.show()
#distribution of different sectors of dataset before cleaning
df_sector = df['Sector'].value_counts()
sns.barplot(np.arange(len(df_sector)), df_sector,palette="Blues_d")
plt.xticks(np.arange(len(df_sector)), df_sector.index.values.tolist(), rotation=90)
plt.title('SECTOR DIFFERENCE FULL DATASET', fontsize=25)
plt.show()

#initial cleaning o f the data, just looking for rows and columns with more than 50% NAN values
nan_vals = df.isna().sum().sum()# 226269 data values NAN 
col_null = df.isnull().mean()# gets the mean column null of df
col_missing_features = col_null[col_null > 0.50].index# if there are columns with 60% or more they will be dropped
df_clean = df
df_clean.drop(col_missing_features, axis=1, inplace=True)#drop columns that do not meet threshhold
row_null = df.isnull().mean(axis = 1)# getting a list of row with nulls
row_missing_features = row_null[row_null > 0.50].index
df_clean.drop(row_missing_features, axis=0, inplace=True)# dropping rows that do no meet the threshhold
nan_vals_clean = df_clean.isna().sum().sum()# 226269 data values NAN 
#print(nan_vals_clean)#50801    ---- 226269-50801=175,468 values gone
#######################################
#using label encoder to deal with sector since it is a strong feature but not numerical
df_dropclass = df_clean.drop('Class',axis = 1)
labelencoder = LabelEncoder()
df_dropclass['NSECTOR']= labelencoder.fit_transform(df_clean['Sector'])
df_class = df['Class']
df_use = pd.concat([df_dropclass,df_class], axis=1)
df_new_clean1 = df_use.drop('Sector',axis = 1)#will use label encoder in future models
df_new_clean = df_new_clean1.drop('2018 PRICE VAR [%]',axis = 1)#dropping this to see how values/predictions change
df_new_clean = df_new_clean.transform(lambda x: x.fillna(x.mean()))#fast way of filling NAN values with column means
new_clean_nan_vals = df_new_clean.isna().sum().sum()# shows no nan values in clean dataset

#Locating features with a correlation greater  .8 and dropping from data set to eliminate features
correlated_features = set()
correlation_matrix = df_new_clean.drop('Class', axis=1).corr()

for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > 0.8:
            colname = correlation_matrix.columns[i]
            correlated_features.add(colname)
print(len(correlated_features))
df_new_clean_use = df_new_clean.drop(correlated_features,axis =1)
print(df_new_clean_use.shape)
##############################################################
#Class difference after cleaning
sns.barplot(np.arange(len(df_new_clean_use['Class'].value_counts())), df_new_clean_use['Class'].value_counts(),palette="RdBu")
plt.title('CLASS DIFFERENCE AFTER CLEANING ', fontsize=25)
plt.show()
#################################
df_sector_clean = df_new_clean_use['NSECTOR'].value_counts()
sns.barplot(np.arange(len(df_sector_clean)), df_sector_clean,palette="Blues_d")
plt.xticks(np.arange(len(df_sector_clean)), df_sector_clean.index.values.tolist(), rotation=90)
plt.title('SECTORS COUNT AFTER CLEANING', fontsize=25)
plt.show()
#################################
train_split, test_split = train_test_split(df_new_clean, test_size=0.2, random_state=1, stratify=df_new_clean_use['Class'])
X_train = train_split.iloc[:, :-1].values
y_train = train_split.iloc[:, -1].values
X_test = test_split.iloc[:, :-1].values
y_test = test_split.iloc[:, -1].values

print('Total number of sameples:    ', df_new_clean_use.shape[0])
print('Number of training samples:  ',X_train.shape[0])
print('Number of testing samples:   ',X_test.shape[0])
print('Number of features:          ',X_train.shape[1],'\n')


scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)


##############################**********#######################################
#tuned_parameters = [{'criterion': ['gini', 'entropy'],'splitter':['random','best'],
#                     'max_depth': [4, 6, 8],'max_features': ['auto', 'sqrt']}]
#decision_tree1 = GridSearchCV(DecisionTreeClassifier(random_state=1),
#                    tuned_parameters,
#                    n_jobs=4,
#                    scoring='precision_weighted',
#                    cv=5)
#decision_tree1.fit(X_train, y_train)
#print('Best parameters for DecisionTreeClassifier: ')
#print()
#print('%0.3f for %r' % (decision_tree1.best_score_, decision_tree1.best_params_))
#0.666 for {'criterion': 'entropy', 'max_depth': 4, 'max_features': 'auto', 'splitter': 'best'}
##############################**********#######################################
decision_tree1 = DecisionTreeClassifier(random_state=1,criterion = 'entropy', max_depth = 4,max_features = 'auto',splitter='best')
decision_tree1.fit(X_train,y_train)

df1 = pd.DataFrame(y_test, index=test_split.index.values, columns=['Original_class'])
df1['DTR1'] = decision_tree1.predict(X_test)

dftr1_precision = average_precision_score(y_test, df1['DTR1'])
print("DTR precision: ",dftr1_precision)
dtr1_pred_acc = decision_tree1.score(X_test, y_test)
print('DTR Pred Accuracy Score: ', dtr1_pred_acc,'\n')

dtr1_disp = plot_precision_recall_curve(decision_tree1, X_test, y_test)
dtr1_disp.ax_.set_title('Precision-Recall Curve')

############################################################
#tuned_parameters = [{'kernel': ['rbf', 'linear'], 'gamma': [1e-3, 1e-4], 'C': [0.01, 0.1, 1, 10, 100]}]
#svm_model = GridSearchCV(SVC(random_state=1),
#                    tuned_parameters,
#                    n_jobs=4,
#                    scoring='precision_weighted',
#                    cv=5)
#svm_model.fit(X_train, y_train)
#print('Best score parameters for SVM:')
#print()
#print('%0.3f for %r' % (svm_model.best_score_, svm_model.best_params_))#c:100 gamma 0.001 kernal:linear
#print()# settinngs changed 0.613 for {'C': 100, 'gamma': 0.001, 'kernel': 'rbf'}

svm_model = SVC(random_state=1, C= 100, gamma=1e-3, kernel = 'rbf')
svm_model.fit(X_train, y_train)

df1['SVM'] = svm_model.predict(X_test)
svm_precision = average_precision_score(y_test, df1['SVM'])
print("SVM Precision: ",svm_precision)
svm_pred_acc = svm_model.score(X_test, y_test)
print('SVM Pred Accuracy Score: ', svm_pred_acc,'\n')

ax = plt.gca()
svm_disp = plot_precision_recall_curve(svm_model, X_test, y_test,ax=ax)

#####################################################################

#tuned_parameters = {'n_estimators': [1024, 4096],
#                    'max_features': ['auto', 'sqrt'],
#                    'max_depth': [4, 6, 8],
#                    'criterion': ['gini', 'entropy']}
#
#rf_model = GridSearchCV(RandomForestClassifier(random_state=1),
#                    tuned_parameters,
#                    n_jobs=4,
#                    scoring='precision_weighted',
#                    cv=5)
#
#rf_model.fit(X_train, y_train)
#
#print('Best parameters for RandomForestClassifier:')
#print()
#print('%0.3f for %r' % (rf_model.best_score_, rf_model.best_params_))
#print()
## 1024..0.725 for {'criterion': 'entropy', 'max_depth': 4, 'max_features': 'auto', 'n_estimators': 1024}

rf_model=RandomForestClassifier(random_state=1,criterion = 'entropy', max_depth = 4,max_features = 'auto',n_estimators = 1024)
rf_model.fit(X_train, y_train)

df1['RF'] = rf_model.predict(X_test)

rf_precision = average_precision_score(y_test, df1['RF'])
print("RF Precision: ",rf_precision)
rf_pred_acc = rf_model.score(X_test, y_test)
print('RF Pred Accuracy Score: ', rf_pred_acc,'\n')

ax = plt.gca()
rf_disp = plot_precision_recall_curve(rf_model, X_test, y_test,ax=ax)

###############################################################################
#tuned_parameters = {'learning_rate': [0.01, 0.001],
#                    'max_depth': [4, 6, 8],
#                    'n_estimators': [512, 1024]}
#
#xgb_model = GridSearchCV(xgb.XGBClassifier(random_state=1),
#                   tuned_parameters,
#                   n_jobs=4,
#                   scoring='precision_weighted',
#                   cv=5)
#xgb_model.fit(X_train, y_train)
#print('Best parameters for XGB:')
#print()
#print('%0.3f for %r' % (xgb_model.best_score_, xgb_model.best_params_))
#print()
##1.000 for {'learning_rate': 0.01, 'max_depth': 4, 'n_estimators': 512}
#0.707 for {'learning_rate': 0.01, 'max_depth': 6, 'n_estimators': 512}

xgb_model=xgb.XGBClassifier(Learning_rate = .01, max_depth = 6,n_estimators = 512)
xgb_model.fit(X_train, y_train)

df1['XGB'] = xgb_model.predict(X_test)
xgb_precision = average_precision_score(y_test, df1['XGB'])
print("XGB Precision: ",xgb_precision)
xgb_pred_acc = xgb_model.score(X_test, y_test)
print('XGB prediction accuracy: ', xgb_pred_acc,'\n')

ax = plt.gca()
xgb_disp = plot_precision_recall_curve(xgb_model, X_test, y_test,ax=ax)
plt.show()


#########################################################################
dtr_disp_roc = plot_roc_curve(decision_tree1, X_test, y_test)
dtr_disp_roc.ax_.set_title('ROC Curve')
ax = plt.gca()
svm_disp_roc = plot_roc_curve(svm_model, X_test, y_test,ax=ax)
ax = plt.gca()
rf_disp_roc = plot_roc_curve(rf_model, X_test, y_test,ax=ax)
ax = plt.gca()
xgb_disp_roc = plot_roc_curve(xgb_model, X_test, y_test,ax=ax)
plt.show()
################### print predicted results to excel spreadsheet ##########
#df1['comparison'] = np.where(df1['DTR1'] == df1['RF'], 'no change', '1')
#later on i would like to do comparisons of these in the excel spreadsheet
df1.to_excel(r'C:\Users\pokim\predicted_values.xlsx', index = False)
###########################################################################
#comparison values for models using rfe
X = df_new_clean_use.drop("Class",1)   #Feature Matrix
y = df_new_clean_use["Class"]          #Target Variable
##################################################################
X_train1, X_test1, y_train1, y_test1 = train_test_split(X,y, test_size = 0.2, random_state = 0)
# Rerunning the decision tree accuracy score with the 75 features from rfe
dt_model1 = DecisionTreeClassifier(random_state=1,criterion = 'entropy', max_depth = 4,max_features = 'auto',splitter='best')
dt_rfe = RFE(dt_model1,75,step=1)
dt_X_train1_rfe = dt_rfe.fit_transform(X_train1,y_train1)
dt_X_test1_rfe = dt_rfe.transform(X_test1)
dt_model1.fit(dt_X_train1_rfe,y_train1)
dt_score1 = dt_model1.score(dt_X_test1_rfe,y_test1)
print("DecisionTree Accuracy Score",dt_score1)
##Rerunning the random forest accuracy score with the 75 features from rfe
#
Rf_model1 = RandomForestClassifier(random_state=1,criterion = 'entropy', max_depth = 4,max_features = 'auto',n_estimators = 1024)
rf_rfe = RFE(Rf_model1,75,step=1)
rf_X_train1_rfe = rf_rfe.fit_transform(X_train1,y_train1)
rf_X_test1_rfe = rf_rfe.transform(X_test1)
Rf_model1.fit(rf_X_train1_rfe,y_train1)
rf_score1 = Rf_model1.score(rf_X_test1_rfe,y_test1)
print("Random Forest Accuracy Score",rf_score1)
#
##Rerunning the svc accuracy score with the 75 features from rfe
##not able to complete because the test runs too long... kernel must = linear as well
##svm_model1 = SVC(random_state=1, C= 100, gamma=1e-3, kernel = 'rbf')
##svm_rfe = RFE(svm_model1,75,step=1)
##svm_X_train1_rfe = svm_rfe.fit_transform(X_train1,y_train1)
##svm_X_test1_rfe = svm_rfe.transform(X_test1)
##svm_model1.fit(svm_X_train1_rfe,y_train1)
##svm_score1 = svm_model1.score(svm_X_test1_rfe,y_test1)
##print("SVM Accuracy Score",svm_score1)
##Rerunning the XGB accuracy score with the 75 features from rfe
#
xgb_model1 = xgb.XGBClassifier(Learning_rate = .01, max_depth = 6,n_estimators = 512)
xgb_rfe = RFE(xgb_model1,75,step=1)
xgb_X_train1_rfe = xgb_rfe.fit_transform(X_train1,y_train1)
xgb_X_test1_rfe = xgb_rfe.transform(X_test1)
xgb_model1.fit(xgb_X_train1_rfe,y_train1)
xgb_score1 = dt_model1.score(xgb_X_test1_rfe,y_test1)
print("xgb Accuracy Score",xgb_score1)


#bar plot for the original scores vs the rfe scores
model_scores = {"First Score": [dtr1_pred_acc,rf_pred_acc,xgb_pred_acc], "Second Score": [dt_score1,rf_score1,xgb_score1]}
index = ["Decision Tree","Random Forest","XGB"]
df_scores = pd.DataFrame(data = model_scores)
df_scores.index = index
df_scores.plot.barh(rot=30,title="Prediction Score Comparison")
plt.show(block = True)
