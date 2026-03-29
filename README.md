# BLENDED_LEARNING
# Implementation-of-Linear-Regression-for-Predicting-Car-Prices
## AIM:
To write a program to predict car prices using a linear regression model and test the assumptions for linear regression.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Import Libraries: Bring in essential libraries such as pandas, numpy, matplotlib, and sklearn.
2. Load Dataset: Import the dataset containing car prices along with relevant features.
3. Data Preprocessing: Manage missing data and select key features for the model, if required.
4. Split Data: Divide the dataset into training and testing subsets.
5. Train Model: Build a linear regression model and train it using the training data.
6. Make Predictions: Apply the model to predict outcomes for the test set.
7. Evaluate Model: Measure the model's performance using metrics like R² score, Mean Absolute Error (MAE), etc.
8. Check Assumptions: Plot residuals to verify assumptions like homoscedasticity, normality, and linearity.
9. Output Results: Present the predictions and evaluation metrics.

## Program:
```
import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
df=pd.read_csv('CarPrice_Assignment.csv')
x=df[['enginesize','horsepower','citympg','highwaympg']]
y=df['price']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
scaler=StandardScaler()
x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.transform(x_test)
model=LinearRegression()
model.fit(x_train_scaled,y_train)
y_pred=model.predict(x_test_scaled)
print("="*50)
print("MODEL COEFFICIENTS:")
for feature, coef in zip(x.columns, model.coef_):
    print(f"{feature:>12}: {coef:>10.2f}")
print(f"{'Intercept':>12}:{model.intercept_:>10.2f}")
print("="*50)

print("\nMODEL PERFORMANCE:")
mse=mean_squared_error(y_test, y_pred)
print('Mean squared error=',mse)
rmse=np.sqrt(mse)
print('Root mean squared error=',rmse)
r2score=r2_score(y_test, y_pred)
print('R-squared=',r2score)

# 1. Linearity check
plt.figure(figsize=(10,5))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y.min(),y.max()],[y.min(),y.max()],'r--')
plt.title("Linearity check: Actual vs Predicted prices")
plt.xlabel("Actual Price($)")
plt.ylabel("Predicted Price($)")
plt.grid(True)
plt.show()

# 2. Independence (Durbin-Watson)
residuals=y_test-y_pred
dw_test=sm.stats.durbin_watson(residuals)
print(f"\nDurbin-Watson Statistic:{dw_test:.2f}","\n(Values close to 2 indicate no autocorrelation)")

# 3. Homoscedasticity
plt.figure(figsize=(10,5))
sns.residplot(x=y_pred, y=residuals, lowess=True, line_kws={'color':'red'})
plt.title("Homoscedasticity Check: Residuals vs Predicted")
plt.xlabel("Predicted Price ($)")
plt.ylabel("Residuals ($)")
plt.grid(True)
plt.show()

# 4. Normality of residuals
fig, (ax1,ax2)=plt.subplots(1,2,figsize=(12,5))
sns.histplot(residuals, kde=True, ax=ax1)
ax1.set_title("Residuals Distribution")
sm.qqplot(residuals, line='45', fit=True, ax=ax2)
ax2.set_title("Q-Q Plot")
plt.tight_layout()
plt.show()
```

## Output:

<img width="1426" height="696" alt="Screenshot 2026-02-04 090616" src="https://github.com/user-attachments/assets/859f31fe-9de9-4d3f-9485-168154d5bfef" />

<img width="1425" height="435" alt="Screenshot 2026-02-04 090833" src="https://github.com/user-attachments/assets/83a15aa9-e68b-4c6c-b456-dcbef84fa540" />

<img width="1423" height="557" alt="Screenshot 2026-02-04 090903" src="https://github.com/user-attachments/assets/dbc65d01-a2ec-4e67-a5ea-a86cfa62a3bd" />

<img width="1425" height="823" alt="Screenshot 2026-02-04 090930" src="https://github.com/user-attachments/assets/ce6c90e8-ca4e-4cac-b9dc-a75423ee9531" />

<img width="1425" height="943" alt="Screenshot 2026-02-04 091054" src="https://github.com/user-attachments/assets/4ba92a33-73fc-4308-ac36-f11822708896" />

<img width="1425" height="766" alt="Screenshot 2026-02-04 091110" src="https://github.com/user-attachments/assets/56a2a7af-5e7a-4608-b9f3-dfa7354339cf" />

## Result:
Thus, the program to implement a linear regression model for predicting car prices is written and verified using Python programming, along with the testing of key assumptions for linear regression.
