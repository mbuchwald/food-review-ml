import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.regularizers import l2
from keras.optimizers import SGD

def mlp(dim):
	model = Sequential()
	model.add(Dense(256, input_dim=dim, init='normal', activation='relu'))
	#model.add(Dropout(0.5))
	model.add(Dense(256, init='normal', activation='relu'))
	#model.add(Dropout(0.5))
	model.add(Dense(256, init='normal', activation='relu'))
	#model.add(Dropout(0.5))
	model.add(Dense(1, init='normal', activation='sigmoid'))

	sgd = SGD(lr=0.1, decay=1e-7, momentum=0.0, nesterov=True)
	model.compile(#loss='binary_crossentropy',
			loss='mean_squared_error',
              optimizer=sgd, #'rmsprop',
              metrics=['accuracy'])

	return model
	

def regression_with_regularization(dim):
	# Usar L2
	# create model
	model = Sequential()
	model.add(Dense(1024, input_dim=dim, init='normal', activation='relu', W_regularizer=l2(0.0001)))	
	model.add(Dense(512, input_dim=dim, init='normal', activation='relu'))
	model.add(Dense(1, init='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy'])
	return model


def train(words, predictions, iterations = 800, model = None):

	if model is None: 
		model = mlp(words.shape[1])
	predictions = (predictions.astype('float') - 1.0)/ 4.0

	history = model.fit(words, predictions,
                    nb_epoch=iterations, batch_size=5000,
                    verbose=2, validation_split=0.1, shuffle=True)
	
	return model
