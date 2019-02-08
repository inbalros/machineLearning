from Assignment3 import Decorate
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import  KFold
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
        columns=['dataset_size', 'fit_time', 'pred_time','train_size_per_fold','test_size_per_fold','avg_enabmble_size', 'acc_on_test', 'acc_on_train', 'precision', 'recall','fscore'])
dfAllPredDecorateG= pd.DataFrame(
        columns=['dataset_size', 'fit_time', 'pred_time','train_size_per_fold','test_size_per_fold','avg_enabmble_size', 'acc_on_test', 'acc_on_train', 'precision', 'recall','fscore'])


def decorate_pred(train,test,data, x_names,y_names,ensamble_max_size=10, max_iteration=20, R_size=0.1,gan_path=None,gan_flag=False):
    """
    creating one decorate ensemble using the parameter that are given.
    :param train:
    :param test:
    :param data:
    :param x_names:
    :param y_names:
    :param ensamble_max_size:
    :param max_iteration:
    :param R_size:
    :param gan_path:
    :param gan_flag:
    :return:
    """
    start_time = time.time()
    foldEnsamble = Decorate.Decorate(DecisionTreeClassifier(max_depth=4), data.iloc[train], x_names, y_names, ensamble_max_size,
                            max_iteration, R_size, gant_flag=gan_flag, ganPath=gan_path)
    end_time = time.time()
    fit_time = end_time - start_time
    start_time = time.time()
    prediction = Decorate.predict_ensemble(foldEnsamble, data.iloc[test][x_names])  # the predictions labels
    end_time = time.time()
    pred_time = end_time - start_time
    predictionOnTrain = Decorate.predict_ensemble(foldEnsamble, data.iloc[train][x_names])  # the predictions labels
    acu_test = metrics.accuracy_score(pd.Series.tolist(data.iloc[test][y_names]), prediction)
    acu_train = metrics.accuracy_score(pd.Series.tolist(data.iloc[train][y_names]), predictionOnTrain)
    ensamble_size = len(foldEnsamble)
    train_size = len(train)
    test_size = len(test)
    precision, recall, fscore, support = metrics.precision_recall_fscore_support(
        pd.Series.tolist(data.iloc[test][y_names]), prediction, average='macro')
    return np.array(list((fit_time, pred_time,train_size,test_size,ensamble_size, acu_test, acu_train, precision, recall, fscore)))


def normalize_results(pred,index,size):
    pred /= index
    pred = np.insert(pred, 0, size)
    return pred


def predict_decorate_kfold(name,data, x_names,y_names,decorateN,decorateG,ensamble_max_size=5, max_iteration=10, R_size=0.1,paramK=10,gan_path=None):
    """
    using k fold validation to evaluate the decorate results and compare the results with gan if the correct flag is on
    :param name:
    :param data:
    :param x_names:
    :param y_names:
    :param decorateN:
    :param decorateG:
    :param ensamble_max_size:
    :param max_iteration:
    :param R_size:
    :param paramK:
    :param gan_path:
    :return:
    """
    kfold = KFold(paramK, True)
    index = 0
    size = data.shape[0]
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
