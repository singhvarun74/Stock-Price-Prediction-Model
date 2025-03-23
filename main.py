import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential

# Fetching the data of last 5 years of Apple stock
stock_symbol = "AAPL"
df = yf.download(stock_symbol, start="2018-01-01", end="2023-01-01")

#print(df.head())
df = df.reset_index()
# Convert 'Date' column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Fill missing values if any exist (ffill means-forward fill means fill the missing values with the previous value)
df.ffill(inplace=True)

# Create time-based features
df['year'] = df['Date'].dt.year
df['month'] = df['Date'].dt.month
df['day'] = df['Date'].dt.day
# Calculate moving averages
df['SMA50'] = df['Close'].rolling(window=50).mean()
df['SMA200'] = df['Close'].rolling(window=200).mean()

# Define independent (X) and dependent (y) variables
x = df[['year', 'month', 'day']]
y = df['Close']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

model = LinearRegression()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)

rmse = np.sqrt(mean_squared_error(y_test,y_pred))
mae = mean_absolute_error(y_test,y_pred)
print(f"RMSE: {rmse}, MAE: {mae}")
'''
plt.scatter(y_test, y_pred)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Linear Regression Predictions")
plt.show()

'''



print(df)
