# import implemented python files
import Config, CultureClassification
from utils import DataLoader, GensimModels, DataPlotter

import random
import numpy as np

if __name__ == '__main__':
	dl = DataLoader.DataLoader()
	cultures, vocab = dl.load_cultures(Config.path_culture)
	gensimLoader = GensimModels.GensimModels()

	

	"""
	Data

	"""

	# Load Data
	X_all, y_all = [], []
	for i in cultures:
		ingredients = cultures[i][0]
		label = cultures[i][1][0]
		X_all.append(ingredients)
		y_all.append(label)


	# Sample Data
	X, y = zip(*random.sample(list(zip(X_all, y_all)), 10000))
	#X, y = X_all, y_all

	X, y = np.array(X), np.array(y)
	print "total examples %s" % len(y)
	print "unique labels", len(set(y))

	"""
	Experiment

	"""

	cultureClassificationWithBaseline = CultureClassification.CultureClassificationWithBaseline()
	# Run Cuisine Classification Baselines
	# "svc_tfidf", "svc", "mult_nb", "mult_nb_tfidf", "bern_nb", "bern_nb_tfidf"
	# 
	cultureClassificationWithBaseline.run_baselines(X,y)

	
	# Load pre-trained vector
	model_loaded = gensimLoader.load_word2vec(path=Config.path_embeddings_ingredients)
	model_vocab = model_loaded.vocab

	count = 0
	for i in vocab:
		if i in model_vocab:
			#print 'vocab with vector', i
			count += 1
		#else:
		#	print i

	print 'total unique number of ingredients:', len(vocab)
	print 'number of embedded ingredients:', count
	print 'number of unembedded ingredients:', len(vocab)-count