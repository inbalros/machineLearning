from Assignment3 import Decorate,decorate_evaluation
from sklearn import metrics
import numpy as np
import pandas as pd
from random import choices
from sklearn.preprocessing import LabelEncoder

#############################
#                           #
#  gan and normal decorator #
#  data sets - to evaluate  #
#                           #
#############################

# #1) data set = letter recognition: https://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/
# gan_path = r'C:\Users\USER\Documents\machineLearning\Assignment3\GanDataSampling\letter_recognition\letter_recognition_syn.csv'
# data_path = r'C:\Users\USER\Documents\machineLearning\Assignment3\GanDataSampling\letter_recognition\letter_recognition.csv'
# letters = pd.read_csv(data_path, names=["att"+str(i) for i in range(1,18)])
# X_names = ["att"+str(i) for i in range(1,17)]
# y_names = "att17"
# Decorate.encode_categorial(letters)
# decorate_evaluation.predict_decorate_kfold("letters",letters.size,letters,X_names,y_names,True,True,gan_path=gan_path,max_iteration=2,ensamble_max_size=2,R_size=0.001)


# #2) data set = magic04 https://archive.ics.uci.edu/ml/datasets/MAGIC+Gamma+Telescope
# gan_path = r'C:\Users\USER\Documents\machineLearning\Assignment3\GanDataSampling\magic04\magic_syn.csv'
# data_path = r'C:\Users\USER\Documents\machineLearning\Assignment3\GanDataSampling\magic04\magic.csv'
# magic = pd.read_csv(data_path, names=["att"+str(i) for i in range(1,12)])
# X_names = ["att"+str(i) for i in range(1,11)]
# y_names = "att11"
# Decorate.encode_categorial(magic)
# decorate_evaluation.predict_decorate_kfold("magic04",magic.size,magic,X_names,y_names,True,True,gan_path=gan_path,max_iteration=2,ensamble_max_size=2,R_size=0.001)


# # 3) data set = skin_NonSkim https://archive.ics.uci.edu/ml/datasets/skin+segmentation#
# gan_path = r'C:\Users\USER\Documents\machineLearning\Assignment3\GanDataSampling\skin_NonSkin\skin_NoSkin_syn.csv'
# data_path = r'C:\Users\USER\Documents\machineLearning\Assignment3\GanDataSampling\skin_NonSkin\skin_NoSkin.csv'
# skin = pd.read_csv(data_path, names=["att"+str(i) for i in range(1,5)])
# X_names = ["att"+str(i) for i in range(1,4)]
# y_names = "att4"
# Decorate.encode_categorial(skin)
# decorate_evaluation.predict_decorate_kfold("skin_nonSkin",skin.size,skin,X_names,y_names,True,True,gan_path=gan_path,max_iteration=2,ensamble_max_size=2,R_size=0.001)


# # 4) data set = car https://archive.ics.uci.edu/ml/machine-learning-databases/car/
# gan_path = r'C:\Users\USER\Documents\machineLearning\Assignment3\GanDataSampling\car\car_syn.csv'
# data_path = r'C:\Users\USER\Documents\machineLearning\Assignment3\GanDataSampling\car\car.csv'
# cars = pd.read_csv(data_path, names=["att"+str(i) for i in range(1,8)])
# X_names = ["att"+str(i) for i in range(1,7)]
# y_names = "att7"
# Decorate.encode_categorial(cars)
# decorate_evaluation.predict_decorate_kfold("cars",cars.size,cars,X_names,y_names,True,True,gan_path=gan_path,max_iteration=2,ensamble_max_size=2,R_size=0.001)
#


# #5) missing one
#
#
#


#############################
#                           #
#  only normal decorator    #
#  data sets - to evaluate  #
#                           #
#############################

# # 6) data set = bank  https://archive.ics.uci.edu/ml/datasets/Bank+Marketing
# data_path = r'C:\Users\USER\Documents\machineLearning\Assignment3\GanDataSampling\otherDataSets\bank\bank.csv'
# bank = pd.read_csv(data_path, names=["att"+str(i) for i in range(1,18)])
# X_names = ["att"+str(i) for i in range(1,17)]
# y_names = "att17"
# Decorate.encode_categorial(bank)
# decorate_evaluation.predict_decorate_kfold("bank",bank.size,bank,X_names,y_names,True,False,max_iteration=2,ensamble_max_size=2,R_size=0.001)


# # 7) data set = wine  https://archive.ics.uci.edu/ml/datasets/Wine+Quality
# data_path = r'C:\Users\USER\Documents\machineLearning\Assignment3\GanDataSampling\otherDataSets\wine\winequality-white.csv'
# wine = pd.read_csv(data_path, names=["att"+str(i) for i in range(1,13)])
# X_names = ["att"+str(i) for i in range(1,12)]
# y_names = "att12"
# Decorate.encode_categorial(wine)
# decorate_evaluation.predict_decorate_kfold("wine",wine.size,wine,X_names,y_names,True,False,max_iteration=2,ensamble_max_size=2,R_size=0.001)

# # 8) data set = blood  https://archive.ics.uci.edu/ml/datasets/Blood+Transfusion+Service+Center
# data_path = r'C:\Users\USER\Documents\machineLearning\Assignment3\GanDataSampling\otherDataSets\blood_transfusion\blood.csv'
# blood = pd.read_csv(data_path, names=["att"+str(i) for i in range(1,6)])
# X_names = ["att"+str(i) for i in range(1,5)]
# y_names = "att5"
# Decorate.encode_categorial(blood)
# decorate_evaluation.predict_decorate_kfold("blood",blood.size,blood,X_names,y_names,True,False,max_iteration=2,ensamble_max_size=2,R_size=0.1)


# # 9) data set = wifi  https://archive.ics.uci.edu/ml/datasets/Wireless+Indoor+Localization
# data_path = r'C:\Users\USER\Documents\machineLearning\Assignment3\GanDataSampling\otherDataSets\wifi\wifi.csv'
# wifi = pd.read_csv(data_path, names=["att"+str(i) for i in range(1,9)])
# X_names = ["att"+str(i) for i in range(1,8)]
# y_names = "att8"
# Decorate.encode_categorial(wifi)
# decorate_evaluation.predict_decorate_kfold("wifi",wifi.size,wifi,X_names,y_names,True,False,max_iteration=2,ensamble_max_size=2,R_size=0.1)


# # 10) data set = chess https://archive.ics.uci.edu/ml/datasets/Chess+%28King-Rook+vs.+King%29
# data_path = r'C:\Users\USER\Documents\machineLearning\Assignment3\GanDataSampling\otherDataSets\chess\chess.csv'
# chess = pd.read_csv(data_path, names=["att"+str(i) for i in range(1,8)])
# X_names = ["att"+str(i) for i in range(1,7)]
# y_names = "att7"
# Decorate.encode_categorial(chess)
# decorate_evaluation.predict_decorate_kfold("chess",chess.size,chess,X_names,y_names,True,False,max_iteration=2,ensamble_max_size=2,R_size=0.1)
#



decorate_evaluation.write_to_excel(True,False)






