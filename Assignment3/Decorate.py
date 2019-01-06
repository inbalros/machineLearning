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
    return prediction / len(ensamble)


def inverse_presictions(labels):
    unique, counts = np.unique(labels, return_counts=True)
    counts = counts/len(labels)
    machane = sum(1/counts)
    counts = ((1/counts)/machane)
    return (pd.Series(choices(unique,counts,k=len(labels)))).values


def generate_gant_data(R_size, data ,x_names,y_names,ensamble):
    return data,data


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
    new_data[y_names] = (inverse_presictions(pred))
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


def Decorate(base_learn, data,x_names,y_names,gant_flag=False, ensamble_max_size=3, max_iteration=10, R_size=0.1):
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
            data = generate_gant_data(R_size, data ,x_names,y_names,ensamble)
        else:
            data = generate_random_data(R_size, data ,x_names,y_names,ensamble)
        model = base_learn.fit(data[x_names],data[y_names])
        ensamble.append(model)
        ensamble_new_error = calc_ensamble_error(ensamble, original_data[x_names],original_data[y_names])
        if ensamble_new_error < ensamble_error:
            ensamble_size += 1
            ensamble_error= ensamble_new_error
        else:
            ensamble.remove(model)
        trails += 1
    return ensamble


cars = pd.read_csv('../Assignment1/cars.csv', names=["buying", "maint", "doors", "persons", "lug_boot", "safety", "acceptability"])
#for col in cars.columns:
    #if cars[col].dtype == object:
    #    cars[col] = cars[col].astype('category')
    #    cars[col] = cars[col].cat.codes
encode_categorial(cars)
X_names =  ["buying", "maint", "doors", "persons", "lug_boot", "safety"]
y_names = "acceptability"

Decorate(DecisionTreeClassifier(),cars,X_names,y_names)

