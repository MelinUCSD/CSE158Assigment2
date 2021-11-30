from sklearn import linear_model, svm
from sklearn.metrics import mean_absolute_error as MAE

import string
import numpy as np
import scipy.sparse as sps
import matplotlib.pyplot as plt

def readJson(path):
    ''' Return a dictionary from the file '''
    true, false = True, False
    for l in open(path, "rt", encoding="UTF-8"):
        yield eval(l)

def getDataset(path):
    ''' Put the dictionaries into a list '''
    dataset = []
    for d in readJson(path):
        dataset.append(d)
    return dataset

def getPlot(labels):
    ''' Returns a bar graph of the rating distributions '''
    plt.hist(labels, color="Orange")
    plt.xlabel("Review Ratings")
    plt.ylabel("Frequency")
    plt.title("Distribution of Review Ratings")
    plt.show()

def cleanString(review):
    ''' Removes punctuation and extra spaces '''
    tmp = ' '.join(review.split()).replace(" ,",",").replace(" !", "!")
    return tmp.translate(str.maketrans("", "", string.punctuation))

def preprocess(dataset):
    ''' Modify and clean the summary section of the dataset '''
    count = 0
    for d in dataset:
        try:
            d["summary"] = cleanString(d["summary"])
        except:
            dataset.remove(d)
            count += 1
    return dataset, count

def getMostCommonWords(data, N):
    ''' Get the N most common words from the summary field of data '''
    freqs = {}
    for x in data:
        curr = x["summary"].split(" ")
        for w in curr:
            if w not in freqs:
                freqs[w] = 1
            else:
                freqs[w] += 1
    sortedFreqs = {k: v for k, v in sorted(freqs.items(),
        key=lambda item: -item[1])}
    return list(set(list(sortedFreqs.keys())[:N])), sortedFreqs

def getIndices(mostCommon):
    ''' Return the indices of the most common words '''
    indices = {}
    for i, w in enumerate(mostCommon):
        indices[w] = i
    return indices

def bagOfWords(d, mostCommon, indices, N):
    ''' Making a bag of words from the N most common words '''
    f = [0] * N
    tokens = d.split()
    for w in tokens:
        if w in mostCommon:
            f[indices[w]] += 1
    return f + [1]

def getFeatureVector(dataset, mostCommon, indices, N):
    ''' Extract the BOW of the summary field and include the vote field '''
    X, y = [], []
    for d in dataset:
        curr = [1]
        try:
            curr.append(int(d["vote"]))
        except:
            curr.append(0)
        curr += bagOfWords(d["summary"], mostCommon, indices, N)
        X.append(curr)
        y.append(int(d["overall"]))
    return np.array(X).astype(np.float), y

def getFeatureVectorBaseline(dataset, mostCommon, indices, N):
    ''' Extract the BOW of the summary field and exclude the vote field '''
    X, y = [], []
    for d in dataset:
        curr = [1]
        curr += bagOfWords(d["summary"], mostCommon, indices, N)
        X.append(curr)
        y.append(int(d["overall"]))
    return np.array(X).astype(np.float), y

def baselineModelTrain(data):
    ''' Returns the average review rating '''
    count = 0
    for x in data:
        count += x
    return count / len(data)

def baselineModelPredict(data, average):
    ''' Get the predictions for the baseline model '''
    return np.array([average] * len(data))

if __name__ == "__main__":
    # Getting the datasets
    trainPath = "data/train/reviewsTrain.json"
    valPath = "data/val/reviewsVal.json"
    testPath = "data/test/reviewsTest.json"
    
    N_datapoints = 85715
    trainRaw = getDataset(trainPath)[:int(N_datapoints * 0.70)]
    valRaw = getDataset(valPath)[:int(N_datapoints * 0.15)]
    testRaw = getDataset(testPath)[:int(N_datapoints * 0.15)]
    
    # Preprocess the datasets - countX == number of datapoints skipped
    trainProcessed, count1 = preprocess(trainRaw)
    valProcessed, count2 = preprocess(valRaw)
    testProcessed, count3 = preprocess(testRaw)

    # Getting the most common words
    N = 3000
    mcTrain, sortedFreqsTrain = getMostCommonWords(trainProcessed, N)
    mciTrain = getIndices(mcTrain)

    # Getting feature vectors
    print("Getting feature vectors")
    XTrain, yTrain = getFeatureVector(trainProcessed, mcTrain, mciTrain, N)
    XVal, yVal = getFeatureVector(valProcessed, mcTrain, mciTrain, N)
    XTest, yTest = getFeatureVector(testProcessed, mcTrain, mciTrain, N)

    # Assessing: Baseline model
    print("Training baseline model")
    average = baselineModelTrain(yTrain)
    predsBaselineVal = baselineModelPredict(XVal, average)
    predsBaselineTest = baselineModelPredict(XTest, average)
    maeBaselineVal = MAE(predsBaselineVal, yVal)
    maeBaselineTest = MAE(predsBaselineTest, yTest)
    print(f'Baseline Model MAE (Val): {maeBaselineVal}')
    print(f'Baseline Model MAE (Test): {maeBaselineTest}')

    # Assessing: LinearRegression
    print("Training Linear Regression")
    linReg = linear_model.LinearRegression()
    linReg.fit(XTrain, yTrain)
    predsLinRegVal = linReg.predict(XVal)
    predsLinRegTest = linReg.predict(XTest)
    maeLinRegVal = MAE(predsLinRegVal, yVal)
    maeLinRegTest = MAE(predsLinRegTest, yTest)
    print(f'Linear Regression MAE (Val): {maeLinRegVal}')
    print(f'Linear Regression MAE (Test): {maeLinRegTest}')

    # Assessing: Ridge
    print("Training Ridge")
    ridge = linear_model.Ridge()
    ridge.fit(XTrain, yTrain)
    predsRidgeVal = ridge.predict(XVal)
    predsRidgeTest = ridge.predict(XTest)
    maeRidgeVal = MAE(predsRidgeVal, yVal)
    maeRidgeTest = MAE(predsRidgeTest, yTest)
    print(f'Ridge MAE (Val): {maeRidgeVal}')
    print(f'Ridge MAE (Test): {maeRidgeTest}')

    # Assessing: Support Vector Machine
    print("Training Support Vector Machine")
    XTrainSVM = sps.lil_matrix(XTrain)
    XValSVM = sps.lil_matrix(XVal)
    XTestSVM = sps.lil_matrix(XTest)
    svm = svm.LinearSVR()
    svm.fit(XTrainSVM, yTrain)
    predsSVMVal = svm.predict(XValSVM)
    predsSVMTest = svm.predict(XTestSVM)
    maeSVMVal = MAE(predsSVMVal, yVal)
    maeSVMTest = MAE(predsSVMTest, yTest)
    print(f'SVM MAE (Val): {maeSVMVal}')
    print(f'SVM MAE (Test): {maeSVMTest}')

    # Assessing: No vote field
    XTrain_noVote, yTrain = getFeatureVectorBaseline(trainProcessed, mcTrain,
                                mciTrain, N)
    XVal_noVote, yVal = getFeatureVectorBaseline(valProcessed, mcTrain,
                            mciTrain, N)
    XTest_noVote, yTest = getFeatureVectorBaseline(testProcessed, mcTrain,
                            mciTrain, N)
    XTrainSVM_noVote = sps.lil_matrix(XTrain_noVote)
    XValSVM_noVote = sps.lil_matrix(XVal_noVote)
    XTestSVM_noVote = sps.lil_matrix(XTest_noVote)
    svm.fit(XTrain_noVote, yTrain)
    predsSVMVal_noVote = svm.predict(XValSVM_noVote)
    predsSVMTest_noVote = svm.predict(XTestSVM_noVote)
    maeSVMVal_noVote = MAE(predsSVMVal_noVote, yVal)
    maeSVMTest_noVote = MAE(predsSVMTest_noVote, yTest)
    print(f'SVM MAE Baseline (Val): {maeSVMVal_noVote}')
    print(f'SVM MAE Baseline (Test): {maeSVMTest_noVote}')

