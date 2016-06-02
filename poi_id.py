#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.metrics import *
from sklearn.decomposition import PCA
from sklearn import preprocessing
import numpy
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn import linear_model
from sklearn import grid_search
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.cross_validation import train_test_split, StratifiedShuffleSplit
from sklearn.feature_selection import SelectKBest
 


# ============================== #
# ============================== #
# Task 1: Select what features you'll use.
# ============================== #
# ============================== #


features_list = ['poi','salary', 'total_payments', 
				 'bonus', 'total_stock_value', 'expenses', 'other',
				'restricted_stock', 'shared_receipt_with_poi','poi_to_percent','poi_from_percent'] 								 


with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
   

# ============================== #
# ============================== #
# Task 2: Remove outliers
# ============================== #
# ============================== #

del data_dict['TOTAL']
del data_dict['THE TRAVEL AGENCY IN THE PARK']
del data_dict['LOCKHART EUGENE E']

		
# ============================== #
# ============================== #
# Task 3: Create new feature(s)
# ============================== #
# ============================== #


# ============================== #
# New Feature Created - POI Interactions To Percentage
# ============================== #

for key in data_dict:
	if (data_dict[key]['to_messages'] != 'NaN' and 
		data_dict[key]['to_messages'] != 0 and 
		data_dict[key]['from_poi_to_this_person'] != 'NaN' and 
		data_dict[key]['from_poi_to_this_person'] != 0) :
			data_dict[key]['poi_to_percent'] = data_dict[key]['from_poi_to_this_person'] / float(data_dict[key]['to_messages'])
	else:
		data_dict[key]['poi_to_percent'] = 'NaN'

# ============================== #
# New Feature Created - POI Interactions From Percentage
# ============================== #

for key in data_dict:
	if (data_dict[key]['from_messages'] != 'NaN' and 
		data_dict[key]['from_messages'] != 0 and 
		data_dict[key]['from_this_person_to_poi'] != 'NaN' and 
		data_dict[key]['from_this_person_to_poi'] != 0) :
			data_dict[key]['poi_from_percent'] = data_dict[key]['from_this_person_to_poi'] / float(data_dict[key]['from_messages'])
	else:
		data_dict[key]['poi_from_percent'] = 'NaN'


my_dataset = data_dict
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)




# ============================== #
# ============================== #
# Task 4: Try a varity of classifiers
# ============================== #
# ============================== #

pca = PCA()
rfc = RandomForestClassifier()
svm = SVC()
dtc = DecisionTreeClassifier()

# ============================== #
# RFC PIPELINE
# ============================== #
# rfc_pipe = Pipeline(steps=[('pca', pca), ('rfc', rfc)])

# rfc_pipe_n_components = [4,5,6,7]
# rfc_pipe_n_estimators = [10,15,20]
# rfc_pipe_max_features = [2,3,4]
# rfc_pipe_min_samples_split = [2,3,4,6]

# estimator = GridSearchCV(rfc_pipe,
#                          dict(pca__n_components=rfc_pipe_n_components,
#                          		rfc__n_estimators=rfc_pipe_n_estimators,
# 								rfc__max_features=rfc_pipe_max_features,
#                          		rfc__min_samples_split=rfc_pipe_min_samples_split),
#                          scoring='average_precision')



# ============================== #
# SVM PIPELINE
# ============================== #

# scaler = preprocessing.MinMaxScaler()

# svm_pipe = Pipeline(steps=[('scalefeatures', scaler), ('pca', pca), ('svm', svm)])

# svm_pipe_n_components = [1,2,3]
# svm_pipe_gamma = [0.01,0.001,0.1,0.0001]
# svm_pipe_C = [100,1000,10000,10, 1, 100000]
# svm_pipe_tol = [0.0008, 0.001]

# estimator = GridSearchCV(svm_pipe,
#                          dict(pca__n_components=svm_pipe_n_components,
#                          		svm__gamma=svm_pipe_gamma,
#                          		svm__tol=svm_pipe_tol,
#                          		svm__C=svm_pipe_C),scoring='recall')

# ============================== #
# DECISION TREE PIPELINE - FORCED
# ============================== #

scaler = preprocessing.MinMaxScaler()
dtc_pipe = Pipeline(steps=[('scalefeatures', scaler),('pca', pca), ('dtc', dtc)])

folds = 1000
cv = StratifiedShuffleSplit(labels, n_iter= folds, random_state = 23)

dtc_pipe_n_components = [4]
dtc_pipe_max_features = [3]
dtc_pipe_min_samples_split = [2]

estimator = GridSearchCV(dtc_pipe,
                         dict(pca__n_components=dtc_pipe_n_components,
                         		dtc__max_features=dtc_pipe_max_features,
                         		dtc__min_samples_split=dtc_pipe_min_samples_split), cv = cv, scoring='recall')

# ============================== #
# DECISION TREE PIPELINE - UNFORCED
# ============================== #

# dtc_pipe = Pipeline(steps=[('pca', pca), ('dtc', dtc)])

# dtc_pipe_n_components = [4,5,6,7]
# dtc_pipe_max_features = [2,3,4]
# dtc_pipe_min_samples_split = [2,3,4,6,8]

# estimator = GridSearchCV(dtc_pipe,
#                          dict(pca__n_components=dtc_pipe_n_components,
#                          		dtc__max_features=dtc_pipe_max_features,
#                          		dtc__min_samples_split=dtc_pipe_min_samples_split),scoring='recall')



# ============================== #
# ============================== #
# Task 5: Tune your classifier to achieve better than .3 precision and recall using our testing script. 
# ============================== #
# ============================== #


estimator.fit(features, labels)
clf = estimator.best_estimator_


# ============================== #
# ============================== #
# Task 6: Dump your classifier, dataset, and features_list so anyone can check your results. 
# ============================== #
# ============================== #

dump_classifier_and_data(clf, my_dataset, features_list)