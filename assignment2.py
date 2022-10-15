import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import Normalizer
from sklearn.neural_network import MLPClassifier


data =  pd.read_csv('abalone.data', sep=",")
data1 =data
data = data.replace(to_replace=['M', 'F','I'], value=[0, 1, 2])
matrix = data.corr()
mask = np.triu(np.ones_like(matrix, dtype=bool))
sns.heatmap(matrix, annot=True, vmax=1, vmin=-1, center=0, cmap='vlag')
plt.show()

x1 = data['Length']
x2 = data['Diameter']
y = data['Rings']
data.plot.scatter(x='Length',
                      y='Diameter',
                      c='Rings')
plt.show()
x1.hist(bins=25)
plt.show()
x2.hist(bins=25)
plt.show()
y.hist(bins=25)
plt.show()

X =np.array(data.drop(columns=('Rings')))
y = np.array(data['Rings'])

Q1_r2 = []
Q1_rmse = []

for i in range(30):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, shuffle=True)
    reg = linear_model.LinearRegression()
    reg.fit(X_train,y_train)
    regPred = reg.predict(X_test)
    Q1_r2.append(r2_score(y_test,regPred))
    Q1_rmse.append(mean_squared_error(y_test, regPred, squared=False))

print("For linear regress model using all features, the standard deviation of the R-squared score is :",np.std(Q1_r2),"\n" )
print("For linear regress model using all features, the standard deviation of the RMSE is :",np.std(Q1_rmse),"\n" )
print("For linear regress model using all features, the mean of the R-squared score is :",np.mean(Q1_r2),"\n" )
print("For linear regress model using all features, the mean of the RMSE is :",np.mean(Q1_rmse),"\n" )



Q2_r2 = []
Q2_rmse = []

transformer = Normalizer().fit(X)
X= transformer.transform(X)

for i in range(30):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, shuffle=True)
    reg_Q2 = linear_model.LinearRegression()
    reg_Q2.fit(X_train,y_train)
    regPred_Q2 = reg_Q2.predict(X_test)
    Q2_r2.append(r2_score(y_test,regPred_Q2))
    Q2_rmse.append(mean_squared_error(y_test, regPred_Q2, squared=False))

print("For linear regress model using all features with normalisation, the standard deviation of the R-squared score is :",np.std(Q2_r2),"\n" )
print("For linear regress model using all features with normalisation, the standard deviation of the RMSE is :",np.std(Q2_rmse),"\n" )
print("For linear regress model using all features with normalisation, the mean of the R-squared score is :",np.mean(Q2_r2),"\n" )
print("For linear regress model using all features with normalisation, the mean of the RMSE is :",np.mean(Q2_rmse),"\n" )


Q3_r2 = []
Q3_rmse = []
X_Q3 = data1.drop(columns=('Height'))
X_Q3 =X_Q3.drop(columns=('Sex'))
X_Q3 =X_Q3.drop(columns=('She_wgt'))
X_Q3 =X_Q3.drop(columns=('Whl_wgt'))
X_Q3 =X_Q3.drop(columns=('Rings'))
X_Q3 =X_Q3.drop(columns=('Shu_wgt'))
X_Q3 =X_Q3.drop(columns=('Vis_wgt'))

for i in range(30):
    X_train, X_test, y_train, y_test = train_test_split(
         X_Q3, y, test_size=0.4, shuffle=True)

    reg_Q3 = linear_model.LinearRegression()
    reg_Q3.fit(X_train,y_train)
    regPred_Q3 = reg_Q3.predict(X_test)
    Q3_r2.append(r2_score(y_test,regPred))
    Q3_rmse.append(mean_squared_error(y_test, regPred_Q3, squared=False))
#print(np.subtract(Q2_r2,Q1_r2))
print("For linear regress model using 2 features, the standard deviation of the R-squared score is :",np.std(Q3_r2),"\n" )
print("For linear regress model using 2 features, the standard deviation of the RMSE is :",np.std(Q3_rmse),"\n" )
print("For linear regress model using 2 features, the mean of the R-squared score is :",np.mean(Q3_r2),"\n" )
print("For linear regress model using 2 features, the mean of the RMSE is :",np.mean(Q3_rmse),"\n" )




Q4_r2=[]
Q4_rmse=[]

for i in range(30):
    X_train, X_test, y_train, y_test = train_test_split(
         X, y, test_size=0.4, shuffle=True)

    nn = MLPClassifier(solver='sgd', alpha=1e-5,learning_rate ='adaptive', learning_rate_init=0.06,
                        hidden_layer_sizes=(20,10,20),  max_iter =1000,momentum=0.08)
    nn.fit(X_train,y_train)
    nnPred=nn.predict(X_test)
    Q4_r2.append(r2_score(y_test,nnPred))
    Q4_rmse.append(mean_squared_error(y_test,nnPred))

print("For neural network model using all features with normalisation, the standard deviation of the R-squared score is :",np.std(Q4_r2),"\n" )
print("For neural network model using all features with normalisation, the standard deviation of the RMSE is :",np.std(Q4_rmse),"\n" )
print("For neural network model using all features with normalisation, the mean of the R-squared score is :",np.mean(Q4_r2),"\n" )
print("For neural network model using all features with normalisation, the mean of the RMSE is :",np.mean(Q4_rmse),"\n" )
