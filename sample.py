
# coding: utf-8

# In[ ]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import AllChem


# In[ ]:

from sklearn import linear_model


# In[ ]:

"""'
Read in train and test as Pandas DataFrames
"""
df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")


# In[ ]:

df_train.head()


# In[ ]:

df_test.head()


# In[ ]:

#store gap values
Y_train = df_train.gap.values
#row where testing examples start
test_idx = df_train.shape[0]
#delete 'Id' column
df_test = df_test.drop(['Id'], axis=1)
#delete 'gap' column
df_train = df_train.drop(['gap'], axis=1)


# In[ ]:

#DataFrame with all train and test examples so we can more easily apply feature engineering on
df_all = pd.concat((df_train, df_test), axis=0)
df_all.head()


# In[ ]:

"""
Example Feature Engineering

this calculates the length of each smile string and adds a feature column with those lengths
Note: this is NOT a good feature and will result in a lower score!
"""
# smiles_len = np.vstack(df_all.smiles.astype(str).apply(lambda x: len(x)))
# df_all['smiles_len'] = pd.DataFrame(smiles_len)

# countC = np.vstack(df_all.smiles.astype(str).apply(lambda x: x.count('c')))
# df_all['countC'] = pd.DataFrame(countC)

def numAromatic(x):
#     moleculeString = (df_all['smiles'][1])
#     mS = moleculeString.tolist()[0]
    molecule = Chem.MolFromSmiles(str(x))
    return rdMolDescriptors.CalcNumAromaticRings(molecule)

countAromatic = np.vstack(df_all.smiles.astype(str).apply(lambda x: numAromatic(x)))
df_all['countAromatic'] = pd.DataFrame(countAromatic)

df_all.head()

def getBitVector(x):
    molecule = Chem.MolFromSmiles(str(x))
    feature = AllChem.GetMorganFingerprintAsBitVect(molecule, 2, nBits=512, useFeatures=True)
    return feature

bitVector = np.vstack(df_all.smiles.astype(str).apply(lambda x: getBitVector(x)))
df_all['bitVector'] = pd.DataFrame(bitVector)

df_all.head()

# def func():
#     for elem in df_all.smiles:
#         print elem
# #         num = elem.count('c')
# #         print num
# func()


# In[ ]:

#     print (Chem.CalcNumAromaticRings(molecule))
#     print(str(moleculeString[10:]))
#     print(type(str(moleculeString[10:])))
#     molecule = Chem.MolFromSmiles(str(moleculeString[10:]))
#     print (molecule)
#     print (Chem.lipinski.CalcNumAromaticRings(molecule))
    
#     np.vstack(df_all.smiles.astype(str).apply(lambda x: print (x)))
#     for elem in df_all.smiles.astype(str):
#         print (elem)
# #     for elem in df_all.smiles:
# #         print (elem)
# #         num = elem.count('c')
# #         print num


# In[ ]:

#print (Y_train)


# In[ ]:

#Drop the 'smiles' column
df_all = df_all.drop(['smiles'], axis=1)
vals = df_all.values
X_train = vals[:test_idx]
X_test = vals[test_idx:]
print ("Train features:", X_train.shape)
print ("Train gap:", Y_train.shape)
print ("Test features:", X_test.shape)


# In[ ]:

#LR = LinearRegression()
#LR.fit(X_train, Y_train)
#LR_pred = LR.predict(X_test)


# In[ ]:

'''def scorefunc(predictions,truth):
    sum=0
    N=len(predictions)
    for i in range(len(predictions)):
        sum+=np.square(predictions[i]-truth[i])
    ans=np.sqrt(sum/N)
    return ans
'''

# In[ ]:

'''def runPrediction():
    
    def cross_validate(x, y, num):
        length = len(x)/num

        start = 0
        end = length
        for i in range(num):
            testData = []
            testData.append([x[start:end], y[start:end]])
            trainData = []
            trainData.append([np.concatenate([x[0:start], x[end:]]), np.concatenate([y[0:start], y[end:]])])
            
            print ("errors are " , output_errors(testData, trainData))

            start = end
            end += length
            
    def output_errors(test_array, train_array):
        result = []

        clf.fit(train_array[0][0], train_array[0][1])

        # predict data
        curPred=clf.predict(test_array[0][0])

        # add the least square error of this test data
        error = scorefunc(curPred,test_array[0][1])
        result.append(error)

        return result
    
    alphaStep = 1
    alphaParam = 0.000001
    l1_ratioParam = 0.5
    while(alphaParam < 0.1):
        clf = linear_model.Lasso(alpha=alphaParam)
        clf2 = linear_model.LassoLars(alpha=alphaParam)
        clf3= linear_model.Ridge(alpha=alphaParam)
        clf4 = linear_model.ElasticNet(alpha=alphaParam, l1_ratio=l1_ratioParam)
        print (alphaParam)
        cross_validate(X_train, Y_train, 5)
        alphaParam += alphaStep 
    

runPrediction()'''


# In[ ]:

RF = RandomForestRegressor()
RF.fit(X_train, Y_train)
RF_pred = RF.predict(X_test)


# In[ ]:




# In[ ]:

def write_to_file(filename, predictions):
    with open(filename, "w") as f:
        f.write("Id,Prediction\n")
        for i,p in enumerate(predictions):
            f.write(str(i+1) + "," + str(p) + "\n")


# In[ ]:

write_to_file("sample111.csv", RF_pred)
#write_to_file("sample2.csv", RF_pred)


# In[ ]:


#print(clf.coef_)
#print(clf.intercept_)


# In[ ]:




# In[ ]:

#write_to_file("sample3.csv",Lasso_pred)


# In[ ]:

#print(Lasso_pred)


# In[ ]:




# In[ ]:

#scorefunc(Lasso_pred,Y_train)


# In[ ]:

#scorefunc(LR_pred,Y_train)


# In[ ]:





# In[ ]:


        
        


# In[ ]:

#

