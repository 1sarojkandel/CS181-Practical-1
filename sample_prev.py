import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

"""
Read in train and test as Pandas DataFrames
"""
df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

df_train.head()
df_test.head()

#store gap values
Y_train = df_train.gap.values
#row where testing examples start
test_idx = df_train.shape[0]
#delete 'Id' column
df_test = df_test.drop(['Id'], axis=1)
#delete 'gap' column
df_train = df_train.drop(['gap'], axis=1)

#DataFrame with all train and test examples so we can more easily apply feature engineering on
df_all = pd.concat((df_train, df_test), axis=0)
df_all.head()

"""
Example Feature Engineering
this calculates the length of each smile string and adds a feature column with those lengths
Note: this is NOT a good feature and will result in a lower score!
"""
#smiles_len = np.vstack(df_all.smiles.astype(str).apply(lambda x: len(x)))
#df_all['smiles_len'] = pd.DataFrame(smiles_len)


#Drop the 'smiles' column
df_all = df_all.drop(['smiles'], axis=1)
vals = df_all.values
X_train = vals[:test_idx]
X_test = vals[test_idx:]
print ("Train features:", X_train.shape)
print ("Train gap:", Y_train.shape)
print ("Test features:", X_test.shape)


LR = LinearRegression()
LR.fit(X_train, Y_train)
LR_pred = LR.predict(X_test)

RF = RandomForestRegressor()
RF.fit(X_train, Y_train)
RF_pred = RF.predict(X_test)

clf = linear_model.Lasso(alpha=0.000001)
clf.fit(X_train, Y_train)
Lasso_pred=clf.predict(X_test)


def write_to_file(filename, predictions):
    with open(filename, "w") as f:
        f.write("Id,Prediction\n")
        for i,p in enumerate(predictions):
            f.write(str(i+1) + "," + str(p) + "\n")


#write_to_file("sample1.csv", LR_pred)
#write_to_file("sample2.csv", RF_pred)
#write_to_file("sample3.csv",Lasso_pred)

def scorefunc(predictions,truth):
    sum=0
    N=len(predictions)
    for i in range(len(predictions)):
        sum+=np.square(predictions[i]-truth[i])
    ans=np.sqrt(sum/N)
    return ans



scorefunc(Lasso_pred,Y_train)


# In[107]:

#scorefunc(LR_pred,Y_train)


# In[ ]:

test_data_array = []
train_data_array = []

def cross_validate(test_data, train_data, x, y, num):
    length = len(x)/num

    start = 0
    end = length
    for i in range(num):
        test_data.append([x[start:end], y[start:end]])
        train_data.append([x[0:start] + x[end:], y[0:start] + y[end:]])
        start = end
        end += length


cross_validate(test_data_array, train_data_array, X_train, Y_train, 5)

#print "test data is ", test_data_array
#print "training data is ", train_data_array




# In[ ]:

def output_errors(test_array, train_array):
    result = []
    for i in range(len(train_array)):
        # train data
        clf.fit(train_array[i][0], train_array[i][1])
        
        # predict data
        lasso_pred=clf.predict(test_array[i][0])
        
        # add the least square error of this test data
        error = scorefunc(lasso_pred,test_array[i][1])
        result.append(error)
        
    return result

print "errors are " , output_errors(test_data_array, train_data_array)
        
        


# In[ ]:

#

