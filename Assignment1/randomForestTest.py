import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer,load_iris,load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from scipy.io import arff
import time

clf_default=RandomForestClassifier()
clf_100_10_3=RandomForestClassifier(n_estimators=100, min_samples_leaf=10, max_depth=3)
clf_100_5_3=RandomForestClassifier(n_estimators=100, min_samples_leaf=5, max_depth=3)
clf_50_10_3=RandomForestClassifier(n_estimators=50, min_samples_leaf=10, max_depth=3)
clf_50_5_3=RandomForestClassifier(n_estimators=50, min_samples_leaf=5, max_depth=3)
clf_50_10_2=RandomForestClassifier(n_estimators=100, min_samples_leaf=10, max_depth=2)
clf_50_5_2=RandomForestClassifier(n_estimators=50, min_samples_leaf=5, max_depth=2)

clf_list = [clf_default,clf_50_10_3,clf_50_5_3,clf_100_10_3,clf_100_5_3,clf_50_10_2,clf_50_5_2]

def fit_and_prdict(X,y,dataset_name):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    for clf in clf_list:
        start_time = time.time()
        clf.fit(X_train, y_train)
        end_time = time.time()
        fit_time = end_time-start_time
        start_time = time.time()
        y_pred = clf.predict(X_test)
        end_time = time.time()
        pred_time = end_time-start_time
        print("for {} dataset and {} clf the results are:".format(dataset_name,clf))
        print("time to fit: {}\ntime to predict:{}".format(fit_time,pred_time))
        print("Accuracy on train set:{}\nAccuracy on test set:{}".format(clf.score(X_train,y_train),metrics.accuracy_score(y_test, y_pred)))
        print('-'*20)

##########Caesarian Section ##########
df = pd.read_csv('csv_result-caesarian.csv', index_col=0)
X=df[["'Age'", "'Delivery", "'Delivery.1", "'Blood","'Heart"]]
y=df['Caesarian']
# fit_and_prdict(X,y,'Caesarian')

##### Cryotherapy dataset ########
df = pd.read_excel('Cryotherapy.xlsx', index_col=0)
X=df[["age", "Time", "Number_of_Warts", "Type","Area"]]
y=df['Result_of_Treatment']
# fit_and_prdict(X,y,'Cryotherapy')

######iris dataset #################
iris= load_iris()
# fit_and_prdict(iris.data,iris.target,'iris')


##### breast_cancer #######
breast_cancer = load_breast_cancer()
# fit_and_prdict(breast_cancer.data, breast_cancer.target,'breast_cancer')
#
# ### wine dataset #####
wine = load_wine()
# fit_and_prdict(wine.data, wine.target,'wine')
#

## zoo ###
zoo = pd.read_csv('zoo.csv',index_col = False)
X = zoo[['hair','feathers','eggs','milk','airborne','aquatic','predator','toothed','backbone','breathes','venomous','fins','legs','tail','domestic','catsize']]
y = zoo[['class_type']]
# fit_and_prdict(X, y,'zoo')

## Bank Marketing Data Set #######
bank = pd.read_csv('bank.csv',index_col = False,sep=';')
# print(bank.columns)
# print(bank.head())
for col in bank.columns:
    if bank[col].dtype == object:
        bank[col] = bank[col].astype('category')
        bank[col] = bank[col].cat.codes
X = bank[['age','job','marital','education','default','balance','housing','loan','contact','day','month','duration','campaign','pdays','previous','poutcome']]
y = bank[['y']]
# fit_and_prdict(X, y,'bank')

## cars ###
cars = pd.read_csv('cars.csv',names = ["buying", "maint", "doors", "persons","lug_boot","safety","acceptability"])
for col in cars.columns:
    if cars[col].dtype == object:
        cars[col] = cars[col].astype('category')
        cars[col] = cars[col].cat.codes
X = cars[["buying", "maint", "doors", "persons","lug_boot","safety"]]
y= cars[["acceptability"]]
# fit_and_prdict(X,y,'cars')

### poker hand ####
poker = pd.read_csv('poker_hand_train.csv',names = ["S1", "C1","S2", "C2","S3", "C3","S4", "C4","S5", "C5",'class'])
X = poker[["S1", "C1","S2", "C2","S3", "C3","S4", "C4","S5", "C5"]]
y= poker[["class"]]
# fit_and_prdict(X,y,'poker')

### glass ####
glass = pd.read_csv('glass.csv',names = ["Id","RI","Na", "Mg","Al", "Si","K", "Ca","Ba", "Fe",'type'])
X = glass[["RI","Na", "Mg","Al", "Si","K", "Ca","Ba", "Fe"]]
y= glass[["type"]]
fit_and_prdict(X,y,'glass')
