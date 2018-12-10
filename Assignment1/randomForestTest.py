import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_iris, load_diabetes, load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import time

clf_default = RandomForestClassifier()
clf_25_10_3 = RandomForestClassifier(n_estimators=25, min_samples_leaf=10, max_depth=3)
clf_25_10_2 = RandomForestClassifier(n_estimators=25, min_samples_leaf=10, max_depth=2)
clf_25_20_3 = RandomForestClassifier(n_estimators=25, min_samples_leaf=20, max_depth=3)
clf_25_20_2 = RandomForestClassifier(n_estimators=25, min_samples_leaf=20, max_depth=2)
clf_25_30_3 = RandomForestClassifier(n_estimators=25, min_samples_leaf=30, max_depth=3)
clf_25_30_2 = RandomForestClassifier(n_estimators=25, min_samples_leaf=30, max_depth=2)
clf_50_10_3 = RandomForestClassifier(n_estimators=50, min_samples_leaf=10, max_depth=3)
clf_50_10_2 = RandomForestClassifier(n_estimators=50, min_samples_leaf=10, max_depth=2)
clf_50_20_3 = RandomForestClassifier(n_estimators=50, min_samples_leaf=20, max_depth=3)
clf_50_20_2 = RandomForestClassifier(n_estimators=50, min_samples_leaf=20, max_depth=2)
clf_50_30_3 = RandomForestClassifier(n_estimators=50, min_samples_leaf=30, max_depth=3)
clf_50_30_2 = RandomForestClassifier(n_estimators=50, min_samples_leaf=30, max_depth=2)
clf_100_10_3 = RandomForestClassifier(n_estimators=100, min_samples_leaf=10, max_depth=3)
clf_100_10_2 = RandomForestClassifier(n_estimators=100, min_samples_leaf=10, max_depth=2)
clf_100_20_3 = RandomForestClassifier(n_estimators=100, min_samples_leaf=20, max_depth=3)
clf_100_20_2 = RandomForestClassifier(n_estimators=100, min_samples_leaf=20, max_depth=2)
clf_100_30_3 = RandomForestClassifier(n_estimators=100, min_samples_leaf=30, max_depth=3)
clf_100_30_2 = RandomForestClassifier(n_estimators=100, min_samples_leaf=30, max_depth=2)
writer = pd.ExcelWriter('NB_randomforest_results.xlsx')
df_dic ={}

clf_dic = {
    'clf_default': clf_default,
    'clf_25_10_3': clf_25_10_3,
    'clf_25_10_2': clf_25_10_2,
    'clf_25_20_3': clf_25_20_3,
    'clf_25_20_2': clf_25_20_2,
    'clf_25_30_3': clf_25_30_3,
    'clf_25_30_2': clf_25_30_2,
    'clf_50_10_3': clf_50_10_3,
    'clf_50_10_2': clf_50_10_2,
    'clf_50_20_3': clf_50_20_3,
    'clf_50_20_2': clf_50_20_2,
    'clf_50_30_3': clf_50_30_3,
    'clf_50_30_2': clf_50_30_2,
    'clf_100_10_3': clf_100_10_3,
    'clf_100_10_2': clf_100_10_2,
    'clf_100_20_3': clf_100_20_3,
    'clf_100_20_2': clf_100_20_2,
    'clf_100_30_3': clf_100_30_3,
    'clf_100_30_2': clf_100_30_2}


def fit_and_prdict(X, y, dataset_name):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    train_size = X.shape[0]
    clf_dic['clf_25_0.1C_3'] = (RandomForestClassifier(n_estimators=25, min_samples_leaf=int(train_size * 0.1), max_depth=3))
    clf_dic['clf_50_0.1C_3'] = (RandomForestClassifier(n_estimators=50, min_samples_leaf=int(train_size * 0.1), max_depth=3))
    clf_dic['clf_100_0.1C_3'] = (RandomForestClassifier(n_estimators=100, min_samples_leaf=int(train_size * 0.1), max_depth=3))
    df = pd.DataFrame(columns=['dataset_size','fit_time', 'pred_time', 'acc_on_test', 'acc_on_train', 'precision', 'recall'])
    for clf_name, clf in clf_dic.items():
        start_time = time.time()
        clf.fit(X_train, y_train)
        end_time = time.time()
        fit_time = end_time - start_time
        start_time = time.time()
        y_pred = clf.predict(X_test)
        end_time = time.time()
        pred_time = end_time - start_time
        precision, recall, fscore, support = metrics.precision_recall_fscore_support(y_test, y_pred, average='macro')
        df.loc[clf_name] = [train_size, fit_time, pred_time, metrics.accuracy_score(y_test, y_pred), clf.score(X_train, y_train),
                            precision, recall]
    df_dic[dataset_name] = df


##########Caesarian Section ##########
df = pd.read_csv('csv_result-caesarian.csv', index_col=0)
X = df[["'Age'", "'Delivery", "'Delivery.1", "'Blood", "'Heart"]]
y = df['Caesarian']
fit_and_prdict(X,y,'Caesarian')

##### Cryotherapy dataset ########
df = pd.read_excel('Cryotherapy.xlsx', index_col=0)
X = df[["age", "Time", "Number_of_Warts", "Type", "Area"]]
y = df['Result_of_Treatment']
fit_and_prdict(X,y,'Cryotherapy')

######iris dataset #################
iris = load_iris()
fit_and_prdict(iris.data, iris.target, 'iris')

##### breast_cancer #######
breast_cancer = load_breast_cancer()
fit_and_prdict(breast_cancer.data, breast_cancer.target,'breast_cancer')
#
# ### wine dataset #####
wine = load_wine()
fit_and_prdict(wine.data, wine.target,'wine')
#


## zoo ###
zoo = pd.read_csv('zoo.csv', index_col=False)
X = zoo[['hair', 'feathers', 'eggs', 'milk', 'airborne', 'aquatic', 'predator', 'toothed', 'backbone', 'breathes',
         'venomous', 'fins', 'legs', 'tail', 'domestic', 'catsize']]
y = zoo[['class_type']]
fit_and_prdict(X, y,'zoo')

## Bank Marketing Data Set #######
bank = pd.read_csv('bank.csv', index_col=False, sep=';')
for col in bank.columns:
    if bank[col].dtype == object:
        bank[col] = bank[col].astype('category')
        bank[col] = bank[col].cat.codes
X = bank[['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month',
          'duration', 'campaign', 'pdays', 'previous', 'poutcome']]
y = bank[['y']]
fit_and_prdict(X, y,'bank')

## cars ###
cars = pd.read_csv('cars.csv', names=["buying", "maint", "doors", "persons", "lug_boot", "safety", "acceptability"])
for col in cars.columns:
    if cars[col].dtype == object:
        cars[col] = cars[col].astype('category')
        cars[col] = cars[col].cat.codes
X = cars[["buying", "maint", "doors", "persons", "lug_boot", "safety"]]
y = cars[["acceptability"]]
fit_and_prdict(X,y,'cars')

### blood ####
blood = pd.read_csv('blood.csv', names=["Recency", "Frequency", "Monetary", "Time",'target'])
X = blood[["Recency", "Frequency", "Monetary", "Time"]]
y = blood[["target"]]

fit_and_prdict(X,y,'blood')


### glass ####
glass = pd.read_csv('glass.csv', names=["Id", "RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe", 'type'])
X = glass[["RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe"]]
y = glass[["type"]]
fit_and_prdict(X,y,'glass')

for name, df in df_dic.items():
    df.to_excel(writer,sheet_name=name)
writer.save()
