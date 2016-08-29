import numpy as np
import heapq
import prediction

TRAINING_SET = "set/train.csv"
ID_FIELD = 0
PREDICTION_FIELD = 6
SUMMARY_FIELD = 8
TEXT_FIELD_START = 9

FILTERED_CHARACTERS = "'!?()[]-$:\""
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

	prediction = int(stage5[0])
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
	return bigramas(words, [0 for i in xrange(TAM_VECS)])

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

def filter_stopwords(tests):
	stop_words = get_stop_words(texts)
	return map(lambda text: filter(lambda word: word not in stop_words, text), texts)

def main():
	texts, predictions = parse() 

	#Por ahora no pienso en sacar las stopwords, pero queda para probar:
	#texts = filter_stopwords(texts)
	vecs = np.array(map(lambda text: word2vec(text), texts))
	vecs -= vecs.mean(axis=0)
	vecs /= vecs.std(axis=0)
	predictions = np.array(predictions)

	prediction.train(vecs, predictions)	

	

if __name__ == "__main__":
	main()
