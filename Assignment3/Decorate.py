from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn import metrics
from sklearn.datasets import load_iris
import time
import numpy as np
import pandas as pd
from random import choices
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

###########################################
#  DECORATE algorithm                     #
#                                         #
#                                         #
#                                         #
###########################################

def get_random_from_dist_cat(tupel):
    return (choices(tupel[0],tupel[1]))[0]

def get_random_from_dist_num(tupel):
    return np.random.normal(tupel[0],tupel[1])

def calc_ensamble_error(ensamble, data,labels):
    prediction = predict_ensamble(ensamble, data)
    return 1- (metrics.accuracy_score(labels,prediction))

def predict_ensamble(ensamble, data):
    prediction = 0
    for model in ensamble:
        prediction += model.predict_proba(data)

    prediction /= len(ensamble)

    index_prd = np.argmax(prediction,axis=1)
    the_prd = [(ensamble[0].classes_)[index] for index in index_prd]
    return the_prd

def inverse_presictions(labels,original_lables):
    original_unique, original_counts = np.unique(original_lables, return_counts=True)
    unique, counts = np.unique(labels, return_counts=True)
    laplas = abs(len(original_unique) - len(unique))

    #counts = np.append(counts,[1 for i in range(laplas)])

    innerIndex = 0
    for index in range(len(original_unique)):
        if innerIndex>=len(unique):
            counts = np.insert(counts,index,1)
        elif (original_unique[index] != unique[innerIndex]):
            counts = np.insert(counts,index,1)
        else:
            innerIndex+=1


    counts = counts/(len(labels)+laplas)
    machane = sum(1/counts)
    counts = ((1/counts)/machane)
    return (pd.Series(choices(original_unique,counts,k=len(labels)))).values

def generate_gant_data(R_size, data,x_names ,y_names,ensamble,ganPath):
    allNames=x_names.copy()
    allNames.append(y_names)
    syn_data = pd.read_csv(ganPath,names=allNames)
    indexes = np.random.randint(0,syn_data.shape[0],int(R_size*len(data)))
    new_data = syn_data.iloc[indexes]
    encode_categorial(new_data)
    pred = predict_ensamble(ensamble, new_data[x_names])
    new_data[y_names] = (inverse_presictions(pred, data[y_names]))
    data = data.append(new_data, ignore_index=True)
    return data

def generate_random_data(R_size, data ,x_names,y_names,ensamble):
    colProp = {}
    for col in x_names:
        #categorial
        if data[col].dtype =='category' :
            unique, counts = np.unique(data[col], return_counts=True)
            colProp[col] = (get_random_from_dist_cat,(unique,counts/len(data[col])))
        else:
            colProp[col] = (get_random_from_dist_num,(data[col].mean(),data[col].std()))

    new_data = pd.DataFrame(columns=x_names)
    for index in range(int(R_size*len(data))):
        newRow = [value[0](value[1]) for key,value in colProp.items()]
        new_data = new_data.append(pd.Series(newRow,index = x_names),ignore_index=True)

    pred = predict_ensamble(ensamble, new_data)
    new_data[y_names] = (inverse_presictions(pred,data[y_names]))
    data = data.append(new_data,ignore_index=True)
    return data

def encode_categorial(data):
    le = LabelEncoder()
    for col in data:
        # categorial
        if data[col].dtype == 'object':
            data[col] = data[col].astype('category')
            data[col] = data[col].cat.codes
            data[col] = data[col].astype('category')

def Decorate(base_learn, data,x_names,y_names,ensamble_max_size, max_iteration, R_size,gant_flag=False,ganPath=None):
    original_data = data.copy()
    # original_lables = lables.copy()
    ensamble_size = 1
    trails = 1
    base_learn.fit(original_data[x_names],original_data[y_names])
    model = base_learn
    ensamble = [model]
    ensamble_error = 1 - model.score(original_data[x_names],original_data[y_names])
    while ensamble_size < ensamble_max_size and trails < max_iteration:
        if gant_flag:
            data = generate_gant_data(R_size, original_data ,x_names,y_names,ensamble,ganPath)
        else:
            data = generate_random_data(R_size, original_data ,x_names,y_names,ensamble)
        model = base_learn.fit(data[x_names],data[y_names])
        ensamble.append(model)
        ensamble_new_error = calc_ensamble_error(ensamble, original_data[x_names],original_data[y_names])
        if ensamble_new_error <= ensamble_error:
            ensamble_size += 1
            ensamble_error= ensamble_new_error
        else:
            ensamble.remove(model)
        trails += 1
    return ensamble

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
dfAllPredTree = pd.DataFrame(
        columns=['dataset_size', 'fit_time', 'pred_time', 'acc_on_test', 'acc_on_train', 'precision', 'recall','fscore'])


def predict_decorate_kfold(name, size,data, x_names,y_names,decorateN,decorateG,tree,ensamble_max_size=5, max_iteration=10, R_size=0.1,paramK=10,gan_path=None):
    kfold = KFold(paramK, True)
    index = 0
    allPredDecorateN = 0
    allPredDecorateG = 0
    allPredTree = 0
    for train, test in kfold.split(data):
        index += 1
        #decorate normal
        if decorateN:
            start_time = time.time()
            foldEnsamble = Decorate(DecisionTreeClassifier(max_depth=5),data.iloc[train],x_names,y_names,ensamble_max_size,max_iteration,R_size,gant_flag = False)
            end_time = time.time()
            fit_time = end_time - start_time
            start_time = time.time()
            prediction = predict_ensamble(foldEnsamble,data.iloc[test][x_names]) #the predictions labels
            end_time = time.time()
            pred_time = end_time - start_time
            predictionOnTrain = predict_ensamble(foldEnsamble,data.iloc[train][x_names]) #the predictions labels
            acu_test = metrics.accuracy_score(pd.Series.tolist(data.iloc[test][y_names]), prediction)
            acu_train = metrics.accuracy_score(pd.Series.tolist(data.iloc[train][y_names]), predictionOnTrain)
            precision, recall, fscore, support = metrics.precision_recall_fscore_support(pd.Series.tolist(data.iloc[test][y_names]), prediction, average='macro')
            allPredDecorateN += np.array(list((fit_time,pred_time,acu_test,acu_train,precision, recall, fscore)))
        #decorate Gan
        if decorateG:
            start_time = time.time()
            foldEnsamble = Decorate(DecisionTreeClassifier(max_depth=5), data.iloc[train], x_names, y_names,ensamble_max_size,max_iteration,R_size, gant_flag=True,ganPath=gan_path)
            end_time = time.time()
            fit_time = end_time - start_time
            start_time = time.time()
            prediction = predict_ensamble(foldEnsamble, data.iloc[test][x_names])  # the predictions labels
            end_time = time.time()
            pred_time = end_time - start_time
            predictionOnTrain = predict_ensamble(foldEnsamble, data.iloc[train][x_names])  # the predictions labels
            acu_test = metrics.accuracy_score(pd.Series.tolist(data.iloc[test][y_names]), prediction)
            acu_train = metrics.accuracy_score(pd.Series.tolist(data.iloc[train][y_names]), predictionOnTrain)
            precision, recall, fscore, support = metrics.precision_recall_fscore_support(
                pd.Series.tolist(data.iloc[test][y_names]), prediction, average='macro')
            allPredDecorateG += np.array(list((fit_time, pred_time, acu_test, acu_train, precision, recall, fscore)))
        #normal decision tree
        if tree:
            start_time = time.time()
            model = DecisionTreeClassifier(max_depth=5).fit(data.iloc[train][x_names], data.iloc[train][y_names])
            end_time = time.time()
            fit_time = end_time - start_time
            start_time = time.time()
            prediction = model.predict(data.iloc[test][x_names])  # the predictions labels
            end_time = time.time()
            pred_time = end_time - start_time
            predictionOnTrain = model.predict(data.iloc[train][x_names])  # the predictions labels
            acu_test = metrics.accuracy_score(pd.Series.tolist(data.iloc[test][y_names]), prediction)
            acu_train = metrics.accuracy_score(pd.Series.tolist(data.iloc[train][y_names]), predictionOnTrain)
            precision, recall, fscore, support = metrics.precision_recall_fscore_support(
                pd.Series.tolist(data.iloc[test][y_names]), prediction, average='macro')
            allPredTree += np.array(list((fit_time, pred_time, acu_test, acu_train, precision, recall, fscore)))

    if decorateN:
        allPredDecorateN /= index
        allPredDecorateN = np.insert(allPredDecorateN, 0, size)
        dfAllPredDecorateN.loc[name] = allPredDecorateN
    if decorateG:
        allPredDecorateG /= index
        allPredDecorateG = np.insert(allPredDecorateG, 0, size)
        dfAllPredDecorateG.loc[name] = allPredDecorateG
    if tree:
        allPredTree /= index
        allPredTree = np.insert(allPredTree, 0, size)
        dfAllPredTree.loc[name] = allPredTree

def write_to_excel(decorateN,decorateG,tree):
    if decorateN:
        dfAllPredDecorateN.to_excel(writerResults,'DecorateNormal')
    if decorateG:
        dfAllPredDecorateG.to_excel(writerResults,'DecorateGan')
    if tree:
        dfAllPredTree.to_excel(writerResults,'DecisionTree')
    writerResults.save()


# cars = pd.read_csv('../Assignment1/cars.csv', names=["buying", "maint", "doors", "persons", "lug_boot", "safety", "acceptability"])
# encode_categorial(cars)
# X_names =  ["buying", "maint", "doors", "persons", "lug_boot", "safety"]
# y_names = "acceptability"
# predict_decorate_kfold("cars",cars.size,cars,X_names,y_names,True,False,True)
# write_to_excel(True,False,True)

gan_path = r'C:\Users\USER\Documents\machineLearning\Assignment3\GanDataSampling\letter_recognition\syn_letter_recognition.csv'
data_path = r'C:\Users\USER\Documents\machineLearning\Assignment3\GanDataSampling\letter_recognition\letter_recognition_data.csv'
letters = pd.read_csv(data_path, names=["att"+str(i) for i in range(1,18)])
X_names= ["att"+str(i) for i in range(1,17)]
y_names = "att17"
encode_categorial(letters)
predict_decorate_kfold("letters",letters.size,letters,X_names,y_names,False,True,False,gan_path=gan_path)
write_to_excel(False,True,False)


