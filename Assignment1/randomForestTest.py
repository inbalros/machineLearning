import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer,load_iris,load_diabetes,load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from scipy.io import arff

clf=RandomForestClassifier(n_estimators=100,min_samples_leaf=25)

def fit_and_prdict(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    # Train the model using the training sets y_pred=clf.predict(X_test)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

##########Caesarian Section ##########
df = pd.read_csv('csv_result-caesarian.csv', index_col=0)
X=df[["'Age'", "'Delivery", "'Delivery.1", "'Blood","'Heart"]]
y=df['Caesarian']
fit_and_prdict(X,y)

##### Cryotherapy dataset ########
df = pd.read_excel('Cryotherapy.xlsx', index_col=0)
X=df[["age", "Time", "Number_of_Warts", "Type","Area"]]
y=df['Result_of_Treatment']
fit_and_prdict(X,y)

######iris dataset #################
iris= load_iris()
fit_and_prdict(iris.data,iris.target)


##### breast_cancer #######
breast_cancer = load_breast_cancer()
fit_and_prdict(breast_cancer.data, breast_cancer.target)

# ### wine dataset #####
wine = load_wine()
fit_and_prdict(wine.data, wine.target)
#
# ### diabetes #####
diabetes = load_diabetes()
fit_and_prdict(diabetes.data, diabetes.target)

## zoo ###
zoo = pd.read_csv('zoo.csv',index_col = False)
X = zoo[['hair','feathers','eggs','milk','airborne','aquatic','predator','toothed','backbone','breathes','venomous','fins','legs','tail','domestic','catsize']]
y = zoo[['class_type']]
fit_and_prdict(X, y)

## Bank Marketing Data Set #######
bank = pd.read_csv('bank.csv',index_col = False,sep=';')
for col in bank.columns:
    if bank[col].dtype == object:
        bank[col] = bank[col].astype('category')
        bank[col] = bank[col].cat.codes
X = bank[['age','job','marital','education','default','balance','housing','loan','contact','day','month','duration','campaign','pdays','previous','poutcome']]
y = bank[['y']]
fit_and_prdict(X, y)

