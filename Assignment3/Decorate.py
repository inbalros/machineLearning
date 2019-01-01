from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn import metrics
from sklearn.datasets import load_iris
import time
import numpy as np
import pandas as pd


def Decorate(base_learn, data, gant_flag=False, ensamble_max_size=3, max_iteration=10, R_size=0.1):
    original_data = data.copy()
    ensamble_size = 1
    trails = 1
    model = base_learn.fit(data)
    ensamble = [model]
    ensamble_error = calc_ensamble_error(ensamble,original_data)
    while ensamble_size < ensamble_max_size and trails < max_iteration:
        if gant_flag:
            data = generate_gant_data(R_size, data)
        else:
            data = generate_random_data(R_size, data)
        model = base_learn.fit(data)
        ensamble.append(model)
        ensamble_new_error = calc_ensamble_error(ensamble, original_data)
        if ensamble_new_error < ensamble_error:
            ensamble_size += 1
            ensamble_error= ensamble_new_error
        else:
            ensamble.remove(model)
        trails += 1
    return ensamble


