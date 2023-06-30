#including libraries
import math
import datetime as dt
import pandas
from pandas_datareader import data as pdr
import yfinance as yfin
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')



yfin.pdr_override()
df = pdr.get_data_yahoo('AAPL', start='2012-10-24', end='2020-12-23')
df

#Draw plot for closing price
plt.figure(figsize=(16,8))
plt.title('Closing Price')
plt.plot(df['Close'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price in USD ($)', fontsize=18)
#Creating a new dataframe with only the closing column

data=df.filter(['Close'])

#Convert the dataframe to a numpy

dataset=data.values

#Get the number of rows to train the model, we train on 80% of the dataset

training_data_len = math.ceil(len(dataset)*.8)
training_data_len
#Scale the data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)
scaled_data
#Create the training data set
#Create the scaled training data sets
train_data = scaled_data[0:training_data_len, :]
x_train =[]
y_train = []
for i in range(60, len(train_data)):
  x_train.append(train_data[i-60:i,0])
  y_train.append(train_data[i,0])



# Convert the x_train and y_train to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)

#Reshape the data
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_train.shape




# Building the LSTM model

model = Sequential()
model.add(LSTM(50,return_sequences=True, input_shape = (x_train.shape[1],1))) #50 neurons
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))





#Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

#Create the testing data set -> Create a new array from index 1584 to 2054
test_data = scaled_data[training_data_len-60:, :]
#Creating the data sets
x_test =[]
y_test = dataset[training_data_len:, :]
for i in range(60, len(test_data)):
  x_test.append(test_data[i-60:i, 0])

#Create the testing data set -> Create a new array from index 1584 to 2054
test_data = scaled_data[training_data_len-60:, :]
#Creating the data sets
x_test =[]
y_test = dataset[training_data_len:, :]
for i in range(60, len(test_data)):
  x_test.append(test_data[i-60:i, 0])



# Convert and reshape
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))


#Get the models predicted price values
prediction = model.predict(x_test)
predictions = scaler.inverse_transform(prediction)


# Getting rms of error
rmse=np.sqrt(np.mean(((prediction-y_test)**2)))

#Plotting the data
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions
#Visualize the data
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train['Close'])
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Prediction'], loc='lower right')
plt.show()