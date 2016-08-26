import numpy as np

TRAINING_SET = "set/train.csv"
ID_FIELD = 0
PREDICTION_FIELD = 6
SUMMARY_FIELD = 8
TEXT_FIELD_START = 9

FILTERED_CHARACTERS = "'!?()[]-$:\""
SPACED_CHARACTERS = ",/."

TAM_VECS = 100

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
	return unigramas(words, [0 for i in TAM_VECS])

def main():
	texts, predictions = parse()
	#Por ahora no pienso en sacar las stopwords, pero queda para probar
	vecs = [word2vec(text) for text in texts]

	for vec in vecs:
		print vec
		raw_input()


if __name__ == "__main__":
	main()
