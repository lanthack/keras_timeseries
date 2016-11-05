# Fit a shallow nueral network with 1 input, 1 hidden layer with 8 neurons, and an output layer.
# A time series problem is converted into a regression problem by fitting y(t) to y(t-lag)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense

# fix random seed for reproducibility
np.random.seed(7)

# load the ds
df = pd.read_csv('data/international-airline-passengers.csv', 
	             usecols=[1], engine='python', skipfooter=3)
ds = df.values
ds = ds.astype('float32')

# split into train and test sets
train_size = int(len(ds) * 0.67)
test_size = len(ds) - train_size
train, test = ds[0:train_size,:], ds[train_size:len(ds),:]
print(len(train), len(test))

# convert an array of values into a ds matrix
def create_ds(ds, lag=1):
	dataX, dataY = [], []
	for i in range(len(ds)-lag-1):
		a = ds[i:(i+lag), 0]
		dataX.append(a)
		dataY.append(ds[i + lag, 0])
	return np.array(dataX), np.array(dataY)

# reshape into X=t and Y=t+1
lag = 1
trainX, trainY = create_ds(train, lag)
testX, testY = create_ds(test, lag)

# create and fit a simple network with: 
# 1 input, 1 hidden layer with 8 neurons, and an output layer
model = Sequential()
model.add(Dense(8, input_dim=lag, activation='relu'))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, nb_epoch=200, batch_size=2, verbose=2)

# Estimate model performance
trainScore = model.evaluate(trainX, trainY, verbose=0)
print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore, np.sqrt(trainScore)))
testScore = model.evaluate(testX, testY, verbose=0)
print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore, np.sqrt(testScore)))

# generate() predictions for training
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# shift train predictions for plotting
trainPredictPlot = np.empty_like(ds)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[lag:len(trainPredict)+lag, :] = trainPredict

# shift test predictions for plotting
testPredictPlot = np.empty_like(ds)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(lag*2)+1:len(ds)-1, :] = testPredict

# plot baseline and predictions
plt.plot(ds)
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()