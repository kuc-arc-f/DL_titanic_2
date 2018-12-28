# -*- coding: utf-8 -*-
# 評価
#

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from util_dt import *
import time
import pickle
import sklearn
from keras.utils import np_utils
from class_net import ClassNet

# 学習データ
global_start_time = time.time()


#
def get_subData(src ):
    sub=src
    sub["Age"] = src["Age"].fillna( src["Age"].median())
    sub = sub.dropna()
#    sub["Embark_flg"] = sub["Embarked"].values
    sub["Embark_flg"] = sub["Embarked"]
    sub["Embark_flg"] = sub["Embark_flg"].map(lambda x: str(x).replace('C', '0') )    
    sub["Embark_flg"] = sub["Embark_flg"].map(lambda x: str(x).replace('Q', '1') )
    sub["Embark_flg"] = sub["Embark_flg"].map(lambda x: str(x).replace('S', '2') )
    sub.groupby("Embark_flg").size()
    # convert, num
    sub = sub.assign( Embark_flg=pd.to_numeric( sub.Embark_flg ))
    sub["Sex_flg"] = sub["Sex"].map(lambda x: 0 if x =='female' else 1)    
    return sub

# 標準化対応、学習。
# 学習データ
train_data = pd.read_csv("train.csv" )
test_data = pd.read_csv("test.csv" )
print( train_data.shape )
#print( train_data.head() )
#
# 前処理 ,欠損データ 中央値で置き換える
train2  = train_data[["PassengerId","Survived","Sex","Age" , "Embarked" ,"SibSp" ,"Parch" ]]
test2   = test_data[ ["PassengerId"           ,"Sex","Age" , "Embarked" ,"SibSp" ,"Parch" ]]
#
age_mid=train2["Age"].median()
#print(age_mid )
print(train2.info() )
print(train2.head(10 ) )
#train2 = train2.dropna()
#train2["Embark_flg"] = train2["Embarked"].map(lambda x: str(x).replace('C', '0') )

train_sub =get_subData(train2 )
test_sub =get_subData(test2 )
print(train_sub.info() )
print(test_sub.info() )
#quit()

# 説明変数と目的変数
x_train= train_sub[["Sex_flg","Age" , "Embark_flg" ,"SibSp" ,"Parch" ]]
y_train= train_sub['Survived']
x_test = test_sub[["Sex_flg","Age" , "Embark_flg" ,"SibSp" ,"Parch" ]]

#conv
num_max_y=10
colmax_x =x_train[ "Age" ].max()
#x_train = x_train / colmax_x
#print(x_train[: 10 ])
#quit()

#print("#check-df")
#col_name="Age"
#print(x_train[ col_name ].max() )
#print(x_train[ col_name ].min() )
#quit()
#np
x_train = np.array(x_train, dtype = np.float32).reshape(len(x_train), 5)
y_train = np.array(y_train, dtype = np.float32).reshape(len(y_train), 1)
#正解ラベルをOne-Hot表現に変換
#y_train = y_train / num_max_y
#x_test  = x_test / num_max_y
y_train=np_utils.to_categorical(y_train, 2)

#


# 学習データとテストデータに分ける
print(x_train.shape, y_train.shape )
print(x_test.shape  )
#quit()


# load
#network = SimpleNet(input_size=1 , hidden_size=10, output_size=1 )
network = ClassNet(input_size=5 , hidden_size=10, output_size=2 )
network.load_params("params.pkl" )


train_acc = network.accuracy(x_train, y_train)
#test_acc  = network.accuracy(x_test, y_test)
#
#print("train acc | " + str(train_acc) +  " , loss=" +str(loss) )
print("train acc | " + str(train_acc) )
print ('time : ', time.time() - global_start_time)

#
# 予測をしてCSVへ書き出す
pred = network.predict( x_test)
print(pred.shape )
outList=[]
for item in pred:
    y = np.argmax(item )
    outList.append(y )

#print(outList[: 10])
pred_y= np.array( outList )
#quit()

PassengerId = np.array( test_data["PassengerId"]).astype(int)
#df = pd.DataFrame(outList , PassengerId, columns=["Survived"])
df = pd.DataFrame(pred_y , PassengerId, columns=["Survived"])
df.head()

#
df.to_csv("out2.csv", index_label=["PassengerId"])
quit()

#
# pred
pred =network.predict(x_train)
print(pred[: 10])
#quit()
for item in pred[: 10]:
    y = np.argmax(item )
    #y = np.argmax(item, axis=1)
    #print(item)
    print(y)
quit()

