from Assignment3 import Decorate
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn import metrics
import time
import numpy as np
import pandas as pd


###########################################
#  start to evaluate the algorithms       #
#                                         #
#                                         #
#                                         #
###########################################
writerResults = pd.ExcelWriter('Decorator_results.xlsx')

dfAllPredDecorateN= pd.DataFrame(
        columns=['dataset_size', 'fit_time', 'pred_time', 'acc_on_test', 'acc_on_train', 'precision', 'recall','fscore'])
dfAllPredDecorateG= pd.DataFrame(
        columns=['dataset_size', 'fit_time', 'pred_time', 'acc_on_test', 'acc_on_train', 'precision', 'recall','fscore'])

def decorate_pred(train,test,data, x_names,y_names,ensamble_max_size=5, max_iteration=10, R_size=0.1,gan_path=None,gan_flag=False):
    start_time = time.time()
    foldEnsamble = Decorate.Decorate(DecisionTreeClassifier(max_depth=5), data.iloc[train], x_names, y_names, ensamble_max_size,
                            max_iteration, R_size, gant_flag=gan_flag, ganPath=gan_path)
    end_time = time.time()
    fit_time = end_time - start_time
    start_time = time.time()
    prediction = Decorate.predict_ensamble(foldEnsamble, data.iloc[test][x_names])  # the predictions labels
    end_time = time.time()
    pred_time = end_time - start_time
    predictionOnTrain = Decorate.predict_ensamble(foldEnsamble, data.iloc[train][x_names])  # the predictions labels
    acu_test = metrics.accuracy_score(pd.Series.tolist(data.iloc[test][y_names]), prediction)
    acu_train = metrics.accuracy_score(pd.Series.tolist(data.iloc[train][y_names]), predictionOnTrain)
    precision, recall, fscore, support = metrics.precision_recall_fscore_support(
        pd.Series.tolist(data.iloc[test][y_names]), prediction, average='macro')
    return np.array(list((fit_time, pred_time, acu_test, acu_train, precision, recall, fscore)))

def normalize_results(pred,index,size):
    pred /= index
    pred = np.insert(pred, 0, size)
    return pred

def predict_decorate_kfold(name, size,data, x_names,y_names,decorateN,decorateG,ensamble_max_size=5, max_iteration=10, R_size=0.1,paramK=10,gan_path=None):
    kfold = KFold(paramK, True)
    index = 0
    allPredDecorateN = 0
    allPredDecorateG = 0
    for train, test in kfold.split(data):
        index += 1
        #decorate normal
        if decorateN:
           allPredDecorateN += decorate_pred(train,test,data, x_names,y_names,ensamble_max_size, max_iteration, R_size,gan_path=None,gan_flag=False)
        #decorate Gan
        if decorateG:
            allPredDecorateG += decorate_pred(train,test,data, x_names,y_names,ensamble_max_size, max_iteration, R_size,gan_path=gan_path,gan_flag=True)

    if decorateN:
        results = normalize_results(allPredDecorateN,index,size)
        dfAllPredDecorateN.loc[name] = results
    if decorateG:
        results = normalize_results(allPredDecorateG,index,size)
        dfAllPredDecorateG.loc[name] = results

def write_to_excel(decorateN,decorateG):
    if decorateN:
        dfAllPredDecorateN.to_excel(writerResults,'DecorateNormal')
    if decorateG:
        dfAllPredDecorateG.to_excel(writerResults,'DecorateGan')
    writerResults.save()
