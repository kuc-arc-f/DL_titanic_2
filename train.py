# encoding: utf-8
# titanic 問題。 通常データ train/ test.csv の使用して検証する。
# ロジスティック回帰、
# 標準化の処理、なし。
#
# 評価
# train : % 
# test  : %

# 途中で使用するため、あらかじめ読み込んでおいてください。
# データ加工・処理・分析モジュール
import numpy as np
import numpy.random as random
import scipy as sp
from pandas import Series, DataFrame
import pandas as pd
import time
# 可視化モジュール
import matplotlib.pyplot as plt
import matplotlib as mpl
# 機械学習モジュール
import sklearn
from keras.utils import np_utils
from class_net import ClassNet

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
y_train=np_utils.to_categorical(y_train, 2)

#


# 学習データとテストデータに分ける
print(x_train.shape, y_train.shape )
print(x_test.shape  )
#print( y_train[: 10 ])
#print(type(x_train) )
#quit()
#
global_start_time = time.time()
network = ClassNet(input_size=5 , hidden_size=10, output_size=2 )

#iters_num = 5000  # 繰り返しの回数を適宜設定する    
iters_num = 50000  # 繰り返しの回数を適宜設定する    
train_size = x_train.shape[0]
print( train_size )
#quit()
#
global_start_time = time.time()

#    batch_size = 100
batch_size = 32
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

#    iter_per_epoch = max(train_size / batch_size, 1)
iter_per_epoch =1000
#print(iter_per_epoch)
#quit()

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
#    quit()
    t_batch = y_train[batch_mask]
    
    # 勾配の計算
    grad = network.gradient(x_batch, t_batch)
    
    # パラメータの更新
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]
    
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, y_train)
#        test_acc  = network.accuracy(x_test, y_test)
        train_acc_list.append(train_acc)
        print("i=" +str(i) + ", train acc | " + str(train_acc) + " , loss=" +str(loss) )
        print ('time : ', time.time() - global_start_time)
        #print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))
#pred
train_acc = network.accuracy(x_train, y_train)
#test_acc  = network.accuracy(x_test, y_test)
#
#print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc) + " , loss=" +str(loss) )
print("train acc | " + str(train_acc) +  " , loss=" +str(loss) )
print ('time : ', time.time() - global_start_time)
#
# パラメータの保存
network.save_params("params.pkl")
print("Saved Network Parameters!")

#print(train_acc_list[: 10])
#quit()
#plt
a1=np.arange(len(train_acc_list) )
#plt.plot(a1 , y_test *num_max_y , label = "y_test")
plt.plot(a1 , train_acc_list , label = "predict")
plt.legend()
plt.grid(True)
plt.title("price pred")
plt.xlabel("x")
plt.ylabel("price")
plt.show()
quit()

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

#
# 予測をしてCSVへ書き出す
pred = model.predict(X_test)
PassengerId = np.array( test_data["PassengerId"]).astype(int)
df = pd.DataFrame(pred, PassengerId, columns=["Survived"])
df.head()

#
df.to_csv("out_res.csv", index_label=["PassengerId"])


