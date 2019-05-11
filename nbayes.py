from __future__ import division # so that 1/2 = 0.5 and not 0
import urllib.request
import random
import copy
import numpy as np
import pandas as pd
import helpers


def counts_to_probs(value_counts):
    """
    Convert a list of lists from counts to probability values.

    Input: List of lists, each containing counts of feature frequencies.

    Output: List of lists (same shape as input) that contains the same counts, but
        expressed as proportion of that feature. Numerators and denominators both
        get +1 smoothing.
    """   
 
    value_probs = copy.deepcopy(value_counts)
    for column in value_probs:
        for val in column:
            total_val_count = 0
            for c in column[val]: total_val_count += column[val][c]
            for c in column[val]: 
                column[val][c] = (column[val][c] + 1)/(total_val_count + 1)
                
    
    return value_probs


def normalize_and_sort(pred):
    """
    Since Naive Bayes does not neessarily output true probabilities, this function
        normalizes a list of probabilities to add to 1.

    This function also sorts the probabilities, so that the most probable prediction
        can easily be retrieved from the first element.

    Input: Dictionary of unnormalized "probabilities"

    Output: List of tuples of (Class, Probability), sorted from most to least probable.
    """

    total_prob = 0
    for p in pred: total_prob += pred[p]
    
    pred_remaining = copy.deepcopy(pred)
    output = []
    while len(pred_remaining) > 0:
        max_prob = 0
        for p in pred_remaining:
            if pred_remaining[p] > max_prob:
                max_class = p
                max_prob = pred_remaining[p]
        output += [(max_class, max_prob/total_prob)]
        pred_remaining.pop(max_class)
            
    return output


def nb_train(dataset, target = 'target'):
    """
    Trains the Naive Bayes algorithm, by collecting all the necessary counts and probabilities.

    Input: A Training dataset.

    Ouput: The Naive Bayes model, expressed as a tuple of a) prior probabilities of each class, 
        and b) posterior probabilities of feature values given target classes.

    """    
 
    print("Training Naive Bayes...")
        
    ys = dataset[target]
    xs = dataset.drop([target], axis = 1)
    
    num_obs = xs.shape[0]
    
    #make dict of counts of unique list of classes
    count_labels = {}
    for y in ys:
        if y not in count_labels: count_labels[y] = 0
        count_labels[y] += 1
    
    #calculate prior prob of labels.
    prob_labels = {}
    for c in count_labels: prob_labels[c] = float(count_labels[c])/num_obs

    #count every joint frequency of feature and label
    template_prob = {}
    for k in count_labels: template_prob[k] = 0
        
    counts = []
    for j,col in enumerate(xs):
        column = xs[col]
        counts += [{0: copy.deepcopy(template_prob),
                    1: copy.deepcopy(template_prob)}]
        for i,val in enumerate(column):
            if val not in counts[j]: 
                counts[j][val] = {}
                for k in prob_labels: counts[j][val][k] = 0
            counts[j][val][ys[i]] += 1

    return(prob_labels, counts_to_probs(counts))
    
    
def nb_predict(nb_model, dataset, target = 'target'):
    """
    Creates classification predictions for a test dataset using a Naive Bayes model.

    Input: A Naive Bayes "model" as produced by the nb_train function; a test dataset.

    Output: Class predictions for the test dataset.

    """

    probs_labels = nb_model[0]
    probs_values = nb_model[1]
    
    xs = dataset.drop([target], axis = 1)
    
    predictions = []
    for row in range(xs.shape[0]):
        row_xs = xs.iloc[row]
        this_pred = copy.deepcopy(probs_labels)
        #print(this_pred)
        #print(probs_values)
        for col,x in enumerate(row_xs):
            for c in this_pred: 
                this_pred[c] *= probs_values[col][x][c]
                
        predictions += [normalize_and_sort(this_pred)]
        
        output = [p[0][0] for p in predictions]
        
    return(output)
