from __future__ import division # so that 1/2 = 0.5 and not 0
import urllib.request
import random
import copy
import numpy as np
import pandas as pd
import helpers


# See video M10-S03, 7:00 (Widrow-Hoff rule)
def adaline_train(dataset, target = 'target', eta = 0.1, max_iter = 1000):
    """ 
    Train the adaline ANN. See Widrow and Hoff (19060).

    Input: 
        dataset: A training dataset
        target: The target feature name
        eta: A learning rate
        max_iter: Maximum number of iterations to perform gradient descent

    Output:
        A vector of weights that make up the adaline model. 
        The last element is the 'bias', and is to be applied to a feature
        of -1, not included in dataset.
    """        
 
    ys = dataset[target]
    xs = dataset.drop([target], axis = 1).values
    xs = [list(x) + [-1] for x in xs] 

    n,m = len(xs), len(xs[0])

    weights = [random.random()*0.02 - 0.01 for x in range(m)]
    
    for _ in range(max_iter):
        deltas = [0]*m
        for t in range(n):
            y = np.dot(weights, xs[t])
            for j in range(m):
                deltas[j] += (ys[t] - y)*xs[t][j]        

        for j in range(m):
            weights[j] += eta/m*deltas[j]
        epsilon = np.sqrt(np.sum([(eta/m*d)**2 for d in deltas]))

        #print(epsilon)
        if epsilon < 0.01: break

    return weights
    
    
def adaline_predict(adaline_model, dataset, target = 'target', output_binary = False):
    """
    Form predictions using a adaline model.

    Input:
        adaline_model: A model output from adaline_train function.
        dataset: A test dataset.
        target: The name of the target feature.
        output_binary: Flag to indicate whether outputs should be binary.
    Output:
        If output_binary, -1/1 prediction flags for each dataset observation.
        If output_binary == False, raw scores (higher is more confident of positive value) 
    """
 
    xs = dataset.drop([target], axis = 1).values
    xs = [list(x) + [-1] for x in xs]

    if output_binary:
        def sign(x):
            if x < 0: return -1
            return 1
        ypred = [sign(np.dot(x, adaline_model)) for x in xs]
    else:
        ypred = [np.dot(x, adaline_model) for x in xs]        

    return ypred

def adaline_train_multinet(dataset, target = 'target', eta = 0.1):
    """
    Wrapper function for adaline_train to handle multi-net classification.
    For each target class, the function runs adaline_train to train a
    one-vs-all (OVA) classification.

    Input:
        dataset: Training dataset
        target: The name of the target feature.
        eta: Learning rate

    Output:
        A dictionary of adaline models -- one for each target class.
        Can also be interpretted as a single multi-net adaline model, where
        each element of the dictionary contains the weights for one of the
        output nodes.
    """

    #Dictionary of one-vs-all adaline models with key of class label.
    ova_outputs = {}
    
    multi_target = dataset[target]

    classes = multi_target.drop_duplicates()

    for c in classes:
        newdata = dataset.drop([target], axis=1)
        newdata['target'] = 0
        for i,t in enumerate(multi_target):
            if t == c:
                newdata['target'][i] = 1
        ova_outputs[c] = adaline_train(newdata, eta = eta)
    
    return ova_outputs

def adaline_predict_multinet(adaline_model_multi, dataset, target = 'target'):
    """
    Predict the target class from the multi-net adaline model model.

    Inputs:
        adaline_model_multi: Adaline multi-net model, output of adaline_train_multinet
        dataset: Test dataset
        target: The name of the target feature.

    Output:
        Classification predictions for the test set observations.
    """    
 
    classes = []
    predictions = []

    for c in adaline_model_multi:
        classes += [c]
        this_model = adaline_model_multi[c]
        predictions += [adaline_predict(this_model, dataset, target = target)] 
    
    #get highest scored class for each observation
    class_pred = []
    for i in range(len(predictions[0])):
        preds = [x[i] for x in predictions]
        class_pred += [classes[np.argmax(preds)]]

    return class_pred
