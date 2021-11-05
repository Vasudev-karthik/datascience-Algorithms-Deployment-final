import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
from PIL import Image 

 
# Load the csv file
df = pd.read_csv("https://raw.githubusercontent.com/RupeshMohan/Linear_Regression/master/headbrain.csv")
 
#print(df.head())
 
# Select independent and dependent variable
X = df[["Gender", "Age Range", "Head Size(cm^3)"]]
y = df["Brain Weight(grams)"]
 
# Split the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=50)
 
# Feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test= sc.transform(X_test)
 
# Instantiate the model
regressor = LinearRegression()
 
# Fit the model
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)


X=df['Head Size(cm^3)'].values
Y = df['Brain Weight(grams)'].values
np.corrcoef(X, Y)

plt.scatter(X, Y, c='green', label='Data points')
plt.xlabel('Head Size in cm3')
plt.ylabel('Brain Weight in grams')
plt.title('MSE = mean_squared_error(y_actual, y_predicted) \n RMSE = sqrt(MSE)')

plt.legend()
#plt.show()

mean_x = np.mean(X)
mean_y = np.mean(Y)

n = len(X)
#print(n)

numer = 0
denom = 0
for i in range(n):
    numer += (X[i] - mean_x) * (Y[i] - mean_y)
    denom += (X[i] - mean_x) ** 2
b1 = numer / denom
b0 = mean_y - (b1 * mean_x)

print("coefficients for regression",b1, b0)

plt.rcParams['figure.figsize'] = (10.0, 5.0)

y = b0 + b1 * X

plt.plot(X, y, color='blue', label='Regression Line')

plt.scatter(X, Y, c='green', label='Scatter data')
plt.xlabel('Head Size in cm3')
plt.ylabel('Brain Weight in grams')
plt.legend()
#plt.show()

rmse = 0
for i in range(n):
    y_pred = b0 + b1 * X[i]
    rmse += (Y[i] - y_pred) ** 2
    
rmse = np.sqrt(rmse/n)
print("Root Mean Square Error is",rmse)

plt.rcParams['figure.figsize'] = (10.0, 5.0)

y = b0 + b1 * X

plt.scatter(y_pred,rmse,  c='green', label='Scatter data')
plt.xlabel('Y_pred')
plt.ylabel('rmse')
plt.legend()
#plt.show()

ss_tot = 0
ss_res = 0
for i in range(n):
    y_pred = b0 + b1 * X[i]
    ss_tot += (Y[i] - mean_y) ** 2
    ss_res += (Y[i] - y_pred) ** 2
r2 = 1 - (ss_res/ss_tot)
print("R2 Score",r2)

y_pred

im=Image.open(r'C:\Users\vasud\Downloads\MicrosoftTeams-image.png') 
plt.imshow(im)
#plt.show()

# Make pickle file of our model
pickle.dump(regressor, open("linearmodel.pkl", "wb"))










#pickle.dump(dtc, open("model.pkl", "wb"))



