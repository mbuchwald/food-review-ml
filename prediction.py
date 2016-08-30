import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.regularizers import l1

def mlp(dim):
	model = Sequential()
	model.add(Dense(256, input_dim=dim, init='normal', activation='relu'))
	#model.add(Dropout(0.5))
	model.add(Dense(256, activation='relu'))
	#model.add(Dropout(0.5))
	model.add(Dense(256, activation='relu'))
	#model.add(Dropout(0.5))
	model.add(Dense(1, activation='sigmoid'))

	model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

	return model

def regression(dim):
	# create model
	model = Sequential()
	model.add(Dense(1024, input_dim=dim, init='normal', activation='relu'))	
	model.add(Dense(256, input_dim=dim, init='normal', activation='relu'))
	model.add(Dense(1, init='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy'])
	return model

def regression_with_regularization(dim):
	# create model
	model = Sequential()
	model.add(Dense(1024, input_dim=dim, init='normal', activation='relu', W_regularizer=l1(0.0001)))	
	model.add(Dense(512, input_dim=dim, init='normal', activation='relu'))
	model.add(Dense(1, init='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy'])
	return model


def train(words, predictions, iterations = 300):

	model = mlp(words.shape[1])
	predictions = (predictions.astype('float') - 1.0)/ 4.0

	history = model.fit(words, predictions,
                    nb_epoch=iterations, batch_size=1000,
                    verbose=2, validation_split=0.2)
	
	return model
