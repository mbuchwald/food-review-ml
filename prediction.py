import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout

def mlp(dim):
	model = Sequential()
	model.add(Dense(64, input_dim=dim, init='uniform', activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(64, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(1, activation='sigmoid'))

	model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

	return model

def regression(dim):
	# create model
	model = Sequential()
	model.add(Dense(dim, input_dim=dim, init='normal', activation='relu'))
	model.add(Dense(1, init='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
	return model

def train(words, predictions, iterations = 500):

	model = regression(words.shape[1])

	history = model.fit(words, predictions,
                    nb_epoch=iterations, batch_size=100,
                    verbose=2, validation_split=0.2)
	
	return model