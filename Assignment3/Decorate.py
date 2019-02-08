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


def get_random_from_dist_cat(tuple):
    """
    creating a random feature value from the feature distribution for categorical features
    :param tuple: get the avg and std from the categorical feature
    :return:
    """
    return (choices(tuple[0],tuple[1]))[0]


def get_random_from_dist_num(tuple):
    """
    creating a random feature value from the feature distribution for numerical features
    :param tuple: get the avg and std from the numerical feature
    :return:
     """
    return np.random.normal(tuple[0],tuple[1])


def calc_ensemble_error(ensemble, data, labels):
    """
    calculating the ensemble error using the accuracy score from the sklearn
    :param ensemble: that want to know his error
    :param data: using the data to predict from the ensemble to create the labels to compare
    :param labels: the correct labels to compare to the ensemble
    :return: the ensemble error
    """
    prediction = predict_ensemble(ensemble, data)
    return 1 - (metrics.accuracy_score(labels,prediction))


def predict_ensemble(ensemble, data):
    """
    predict the ensemble classification
    :param ensemble: to get his classification
    :param data: the data to get the classification for
    :return: the ensemble classification for the data
    """
    prediction = 0
    for model in ensemble:
        prediction += model.predict_proba(data)

    prediction /= len(ensemble)

    index_prd = np.argmax(prediction,axis=1)
    the_prd = [(ensemble[0].classes_)[index] for index in index_prd]
    return the_prd


def inverse_predictions(labels, original_labels):
    """
    get the correct labels for the data calculate the classes distributions and get from the inverse
    distribution random classes as the labels for the data  -  adding noise to the data so every tree will learn differently
    :param labels: the correct labels to be inverse
    :param original_labels: all the labels of the data set
    :return: the inverse predictions
    """
    original_unique, original_counts = np.unique(original_labels, return_counts=True)
    unique, counts = np.unique(labels, return_counts=True)
    laplas = abs(len(original_unique) - len(unique))

    # for laplas correction- adding the missing classes in their correct position
    inner_index = 0
    for index in range(len(original_unique)):
        if inner_index >= len(unique):
            counts = np.insert(counts,index,1)
        elif (original_unique[index] != unique[inner_index]):
            counts = np.insert(counts,index,1)
        else:
            inner_index += 1

    counts = counts/(len(labels)+laplas)
    machane = sum(1/counts)
    counts = ((1/counts)/machane)
    return (pd.Series(choices(original_unique,counts,k=len(labels)))).values


def generate_gant_data(R_size, data,x_names ,y_names,ensemble,ganPath):
    """
    generate the synthetic records using the gan records that was created with TGAN.
    the record is pre-prepend and saved in the given path
    :param R_size: the proportion of record to generate from the data
    :param data: the original data set that we want to add the synthetic records to
    :param x_names: the features names
    :param y_names: the class names
    :param ensemble:
    :param ganPath: the path to the file with the gan records
    :return: the data set with the synthetic records
    """
    allNames=x_names.copy()
    allNames.append(y_names)
    syn_data = pd.read_csv(ganPath,names=allNames)
    indexes = np.random.randint(0,syn_data.shape[0],int(R_size*len(data)))
    new_data = syn_data.iloc[indexes]
    encode_categorial(new_data)
    pred = predict_ensemble(ensemble, new_data[x_names])
    new_data[y_names] = (inverse_predictions(pred, data[y_names]))
    data = data.append(new_data, ignore_index=True)
    return data


def generate_random_data(R_size, data ,x_names,y_names,ensemble):
    """
    generate the synthetic records using the algorithm that was mentioned in the article.
    the record are generated from the features distributions
    :param R_size: the proportion of record to generate from the data
    :param data:the original data set that we want to add the synthetic records to
    :param x_names:the features names
    :param y_names:the class names
    :param ensemble:
    :return:the data set with the synthetic records
    """
    colProp = {}
    for col in x_names:
        #categorial
        if data[col].dtype != 'float64' and data[col].dtype != 'int64' and data[col].dtype == 'category' :
            unique, counts = np.unique(data[col], return_counts=True)
            colProp[col] = (get_random_from_dist_cat,(unique,counts/len(data[col])))
        else:
            colProp[col] = (get_random_from_dist_num,(data[col].mean(),data[col].std()))

    new_data = pd.DataFrame(columns=x_names)
    for index in range(int(R_size*len(data))):
        newRow = [value[0](value[1]) for key,value in colProp.items()]
        new_data = new_data.append(pd.Series(newRow,index = x_names),ignore_index=True)

    pred = predict_ensemble(ensemble, new_data)
    new_data[y_names] = (inverse_predictions(pred, data[y_names]))
    data = data.append(new_data,ignore_index=True)
    return data


def encode_categorial(data):
    """
    change the category data to numbers represent the value
    :param data:
    :return:
    """
    le = LabelEncoder()
    for col in data:
        # categorial
        if data[col].dtype == 'object':
            data[col] = data[col].astype('category')
            data[col] = data[col].cat.codes
            data[col] = data[col].astype('category')


def Decorate(base_learn, data,x_names,y_names,ensemble_max_size, max_iteration, R_size,gant_flag=False,ganPath=None):
    """
    the basic decorate algorithm as described in the article and in the paper we wrote as part of the
    submition of the assignment
    :param base_learn: the basic learner that the ensemble going to build from
    :param data: the data set to learn from
     :param x_names:the features names
    :param y_names:the class names
    :param ensemble_max_size:  the max size the ensemble can grow to
    :param max_iteration: the max iteration try to add a learner to the ensemble
    :param R_size: the proportion of record to generate from the data
    :param gant_flag: flag if to use the gan method for synthetic records
    :param ganPath: if the flag is on, using this path to read the synthetic records
    :return:
    """
    original_data = data.copy()
    ensemble_size = 1
    trails = 1
    base_learn.fit(original_data[x_names],original_data[y_names])
    model = base_learn
    ensemble = [model]
    ensemble_error = 1 - model.score(original_data[x_names],original_data[y_names])
    while ensemble_size < ensemble_max_size and trails < max_iteration:
        if gant_flag:
            data = generate_gant_data(R_size, original_data ,x_names,y_names,ensemble,ganPath)
        else:
            data = generate_random_data(R_size, original_data ,x_names,y_names,ensemble)
        model = base_learn.fit(data[x_names],data[y_names])
        ensemble.append(model)
        ensemble_new_error = calc_ensemble_error(ensemble, original_data[x_names], original_data[y_names])
        if ensemble_new_error <= ensemble_error:
            ensemble_size += 1
            ensemble_error= ensemble_new_error
        else:
            ensemble.remove(model)
        trails += 1
    return ensemble





