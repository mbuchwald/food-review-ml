import sys
import numpy as np
import heapq
import prediction
from keras.models import load_model

TRAINING_SET = "set/train.csv"
TEST_SET = "set/test.csv"
ID_FIELD = 0
PREDICTION_FIELD = 6
SUMMARY_FIELD = 8
TEXT_FIELD_START = 9

FILTERED_CHARACTERS = "'!?()[]-$:#\""
SPACED_CHARACTERS = ",/."

TAM_VECS = 1000
CANT_STOPWORDS = 40
SAVED_STOPWORDS = ['coffee', 'if', 'product', 'one', 'taste', 'very', 'great', 'them', 'are', 'its', 'as', 'just', 'or', 'so', 'at', 'not', 'they', 'that', 'you', 'good', 'have', 'i', 'my', 'the', 'these', 'on', 'like', 'is', 'and', 'for', 'be', 'of', 'in', 'was', 'but', 'it', 'a', 'with', 'this', 'to']

def relevant_training_fields(line):
	no_ids = line[line.index(",") + 1:]
	no_pids = no_ids[no_ids.index(",") + 1:]
	stage3 = no_pids[no_pids.index(",") + 1:]
	stage4 = stage3[stage3.index('",') + 2:]
	stage5 = stage4[stage4.index(",") + 1:]
	stage5 = stage5[stage5.index(",") + 1:]

	prediction = float(stage5[0])
	stage6 = stage5[stage5.index(',', 2) + 2:]
	summary = stage6[:stage6.index('",')]
	text = stage6[stage6.index('",') + 2:]
	return summary, text, prediction

def clean(text):
	text = "".join(filter(lambda x: x not in FILTERED_CHARACTERS, text)).lower()
	text = " ".join(text.split("<br />"))
	text = "".join(map(lambda x: x if x not in SPACED_CHARACTERS else " ", text)).lower()
	text = filter(lambda x: len(x) > 0, text.split(" "))
	return text

def parse():
	infile = open(TRAINING_SET)
	infile.readline()
	texts = []
	predictions = []
	errores = []

	for line in infile:
		try:
			summary, text, prediction = relevant_training_fields(line.strip())
			#Por ahora no le doy bola al summary
			texts.append(clean(text))
			predictions.append(prediction)
 			#if len(predictions) == 200000: break
		except ValueError:
			#Hay solo 5 con errores
			errores.append(line.strip())
	infile.close()

	return texts, predictions

def unigramas(words, vector):
	for word in words:
		vector[hash(word) % TAM_VECS] += 1
	return vector

def bigramas(words, vector):
	for i in range(0, len(words) - 1):
		vector[hash(words[i] + words[i+1]) % TAM_VECS] += 1
	return vector

def word2vec(words):
	return unigramas(words, [0 for i in xrange(TAM_VECS)])

def get_stopwords(texts):
	if SAVED_STOPWORDS:
		return SAVED_STOPWORDS[:]
	words = {}
	for text in texts:
		for word in text:
			words[word] = words.get(word, 0) + 1

	q = []
	for word in words:
		if len(q) < CANT_STOPWORDS:
			heapq.heappush(q, (words[word], word))
		elif q[0][0] < words[word]:
			heapq.heappush(q, (words[word], word))
			heapq.heappop(q)
	return map(lambda x: x[1], q)

def filter_stopwords(texts):
	stop_words = get_stopwords(texts)
	return map(lambda text: filter(lambda word: word not in stop_words, text), texts)

def relevant_test_fields(line):
	last_fields = line.split('","')
	text = last_fields[-1]
	summary = last_fields[-2].split(',"')[-1]
	id = line.split(',')[0]
	return id, summary, text

def parse_tests():
	infile = open(TEST_SET)

	infile.readline()
	texts = []
	ids = []
	errores = []
	for line in infile:
		try:
			id, summary, text = relevant_test_fields(line.strip())
			#Por ahora no le doy bola al summary
			texts.append(clean(text))
			ids.append(id)
		except ValueError:
			#Hay solo 5 con errores
			errores.append(line.strip())
	infile.close()

	return ids, texts

def texts_to_array(texts):
	#Por ahora no pienso en sacar las stopwords, pero queda para probar:
	#texts = filter_stopwords(texts)
	return np.array(map(lambda text: word2vec(text), texts)).astype('float')

def normalize(vec):
	return (vec - vec.mean(axis=0)) / vec.std(axis=0)

def main():
	#Analizar agregar interacciones entre features
	texts, predictions = parse() 
	
	vecs = texts_to_array(texts)
	mean = vecs.mean(axis=0)
	std = vecs.std(axis=0)
	vecs = (vecs - mean) / std

	predictions = np.array(predictions)

	if len(sys.argv) == 1:
		model = prediction.train(vecs, predictions)	
	else:
		model = load_model('model.h5')
		model = prediction.train(vecs, predictions, model=model)
		
	model.save('model.h5')	

	ids, tests = parse_tests()
	tests = texts_to_array(tests)
	tests = (tests - mean) / std
	proba = model.predict_proba(tests, batch_size=1000)

	outfile = open("submit.txt", 'w')
	outfile.write('Id,Prediction\n')
	for i in range(len(ids)):	
		outfile.write(ids[i] + "," + str(round(proba[i][0] * 4 + 1, 2)) + '\n')
	outfile.close()

	

if __name__ == "__main__":
	main()
