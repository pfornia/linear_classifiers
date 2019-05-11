from __future__ import division # so that 1/2 = 0.5 and not 0
import urllib.request
import random
import copy
import numpy as np
import pandas as pd
import pickle
import helpers
import nbayes
import logit
import adaline
import sys

#Input Data Set and parameters...

#If inputs not provide, then prompt user.
if len(sys.argv) == 3:
    selection = int(sys.argv[1])
    filepath = sys.argv[2]
else:
    print("Please select classification model type...")
    print("1. Naive Bayes")
    print("2. Logistic")
    print("3. Adaline")
    selection = int(input("Selection: "))
    filepath = input("Enter file path of data: ")

#Read in cleaned dataset.
clean_data = pd.read_pickle(filepath)

#Set seed for train/test splits.
random.seed(3)

#Report the accuracy of simply predicting the most frequent target class.
print("Naive Baseline Accuracy:")
print(np.max(clean_data['target'].value_counts())/len(clean_data['target']))

#Stratified folds if classification, otherwise simple 5-fold split
if type(clean_data['target'][0]) == str:
    folds = helpers.k_folds_stratified(clean_data, k=5, strat_var = 'target')
else:
    folds = helpers.k_folds_split(clean_data, k=5)

performance = [0 for x in folds]

#Train and predict algorithm on each fold
for i,f in enumerate(folds):

    print("Training/testing on fold: " + str(i))
    
    if selection == 1:
        model = nbayes.nb_train(f[0])
        if i == 0:
            print("Naive Bayes probabilities for first fold:")
            print("Prior Probabilities:", model[0])
            columns = copy.deepcopy(f[0].columns.values)
            columns = columns[1:len(columns)]
            for c, col in enumerate(columns):
                print("Posterior probabilities for", col)
                print(model[1][c])
        estimates = nbayes.nb_predict(model, f[1])
    elif selection == 2:
        model = logit.logit_train_multiclass(f[0], eta = 0.01)
        if i == 0:
            print("Logit model output for first fold:")
            columns = copy.deepcopy(f[0].columns.values)
            columns = ["Intercept"] + list(columns[1:len(columns)])
            for key in model:
                print("OVA model for", key)
                for c, col in enumerate(columns):
                    print("    Coefficient for", col, ":", model[key][c])
        estimates = logit.logit_predict_multiclass(model, f[1])
    elif selection == 3:
        model = adaline.adaline_train_multinet(f[0], eta = 0.01)
        if i == 0:
            print("Adaline model output for first fold:")
            columns = copy.deepcopy(f[0].columns.values)
            columns = list(columns[1:len(columns)]) + ["Intercept (-1)"]
            for key in model:
                print("Multi-net coefficients for", key)
                for c, col in enumerate(columns):
                    print("    Coefficient for", col, ":", model[key][c])
        estimates = adaline.adaline_predict_multinet(model, f[1])
    
    actuals = f[1]['target'].values
    #performance[i] = np.mean([actuals[j] == e for j,e in enumerate(estimates)])/len(estimates)
    performance[i] = sum(actuals == estimates)/len(estimates)
    print("Accuracy of this fold:")
    print(performance[i])

#Report the average performance across all 5 folds.
print("Average performance accross folds:")
print(np.mean(performance))


