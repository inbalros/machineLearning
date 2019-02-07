from sklearn import metrics
import numpy as np
import pandas as pd
from random import choices
from sklearn.preprocessing import LabelEncoder

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
        if data[col].dtype !='float64'  and data[col].dtype !='int64'  and  data[col].dtype == 'category' :
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





