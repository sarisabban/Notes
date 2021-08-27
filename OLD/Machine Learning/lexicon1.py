import tensorflow as tf
import numpy as np
import nltk , random , pickle
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter

lemmatizer = WordNetLemmatizer()
lines = 10000000

def creat_lexicon(pos , neg):
	lexicon = list()
	for fi in [pos , neg]:
		with open(fi , 'r') as f:
			contents = f.readlines()
			for l in contents[:lines]:
				all_words = word_tokenize(l.lower())
				lexicon += list(all_words)
	lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
	word_counts = Counter(lexicon)
	l2 = list()
	for w in word_counts:
		if 1000 > word_counts[w] > 50:
			l2.append(w)
	return(l2)
def sample_handeling(sample , lexicon , classification):
	feature_set = list()
	with open(sample , 'r') as f:
		contents = f.readlines()
		for l in contents[:lines]:
			current_words = word_tokenize(l.lower())
			current_words = [lemmatizer.lemmatize(i) for i in current_words]
			features = np.zeros(len(lexicon))
			for word in current_words:
				if word.lower() in lexicon:
					index_value = lexicon.index(word.lower())
					features[index_value] += 1
			features = list(features)
			feature_set.append([features , classification])
	return(feature_set)
def create_feature_sets_and_labels(pos , neg , test_size = 0.1):
	lexicon = creat_lexicon(pos , neg)
	features = list()
	features += sample_handeling('pos.txt' , lexicon , [1 , 0])
	features += sample_handeling('neg.txt' , lexicon , [0 , 1])
	random.shuffle(features)
	features = np.array(features)
	testing_size = int(test_size * len(features))
	train_x = list(features[:,0][:-testing_size])
	train_y = list(features[:,1][:-testing_size])
	test_x = list(features[:,0][-testing_size:])
	test_y = list(features[:,1][-testing_size:])
	return(train_x , train_y , test_x , test_y)
if __name__ == '__main__':
	train_x , train_y , test_x , test_y = create_feature_sets_and_labels('pos.txt' , 'neg.txt')
	with open('sentiment_set.pickle' , 'wb') as f:
		pickle.dump([train_x , train_y , test_x , test_y] , f)
