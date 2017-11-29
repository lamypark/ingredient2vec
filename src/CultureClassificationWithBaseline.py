# import libraries
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import random
from collections import Counter, defaultdict
from tabulate import tabulate
import time

from gensim.models.word2vec import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# import implemented python files
import Config
from utils import DataLoader, GensimModels, DataPlotter


class CultureClassificationWithBaseline:
	def __init__(self):
		print "\n\n==================================================="
		print "...CultureClassificationWithBaseline initialized"
		print "====================================================\n\n"

	def run_baselines(self, X_train,y_train,X_test,y_test):
		# start with the classics - naive bayes of the multinomial and bernoulli varieties
		# with either pure counts or tfidf features
		mult_nb = Pipeline([("count_vectorizer", CountVectorizer(analyzer=lambda x: x)), ("multinomial nb", MultinomialNB())])
		mult_nb_tfidf = Pipeline([("tfidf_vectorizer", TfidfVectorizer(analyzer=lambda x: x)), ("multinomial nb", MultinomialNB())])
		
		bern_nb = Pipeline([("count_vectorizer", CountVectorizer(analyzer=lambda x: x)), ("bernoulli nb", BernoulliNB())])
		bern_nb_tfidf = Pipeline([("tfidf_vectorizer", TfidfVectorizer(analyzer=lambda x: x)), ("bernoulli nb", BernoulliNB())])
		# SVM - which is supposed to be more or less state of the art 
		# http://www.cs.cornell.edu/people/tj/publications/joachims_98a.pdf
		svc = Pipeline([("count_vectorizer", CountVectorizer(analyzer=lambda x: x)), ("linear svc", SVC(kernel="linear"))])
		svc_tfidf = Pipeline([("tfidf_vectorizer", TfidfVectorizer(analyzer=lambda x: x)), ("linear svc", SVC(kernel="linear"))])

		ridge = Pipeline([("count_vectorizer", CountVectorizer(analyzer=lambda x: x)), ("ridge classifier", RidgeClassifier(tol=1e-2, solver="lsqr"))])
		ridge_tfidf = Pipeline([("tfidf_vectorizer", TfidfVectorizer(analyzer=lambda x: x)), ("ridge classifier", RidgeClassifier(tol=1e-2, solver="lsqr"))])

		perceptron = Pipeline([("count_vectorizer", CountVectorizer(analyzer=lambda x: x)), ("Perceptron", Perceptron(n_iter=50))])
		perceptron_tfidf = Pipeline([("tfidf_vectorizer", TfidfVectorizer(analyzer=lambda x: x)), ("Perceptron", Perceptron(n_iter=50))])

		pass_agg = Pipeline([("count_vectorizer", CountVectorizer(analyzer=lambda x: x)), ("Passive-Aggressive", PassiveAggressiveClassifier(n_iter=50))])
		pass_agg_tfidf = Pipeline([("tfidf_vectorizer", TfidfVectorizer(analyzer=lambda x: x)), ("Passive-Aggressive", PassiveAggressiveClassifier(n_iter=50))])
		
		kNN = Pipeline([("count_vectorizer", CountVectorizer(analyzer=lambda x: x)), ("kNN", KNeighborsClassifier(n_neighbors=10))])
		kNN_tfidf = Pipeline([("tfidf_vectorizer", TfidfVectorizer(analyzer=lambda x: x)), ("kNN", KNeighborsClassifier(n_neighbors=10))])

		rand_for = Pipeline([("count_vectorizer", CountVectorizer(analyzer=lambda x: x)), ("random forest", RandomForestClassifier(n_estimators=100))])
		rand_for_tfidf = Pipeline([("tfidf_vectorizer", TfidfVectorizer(analyzer=lambda x: x)), ("random forest", RandomForestClassifier(n_estimators=100))])

		all_models = [
			("mult_nb", mult_nb),
			("mult_nb_tfidf", mult_nb_tfidf),
			("bern_nb", bern_nb),
			("bern_nb_tfidf", bern_nb_tfidf),
			("svc", svc),
			("svc_tfidf", svc_tfidf),
			("ridge", ridge),
			("ridge_tfidf", ridge_tfidf),
			("perceptron", perceptron),
			("perceptron_tfidf", perceptron_tfidf),
			("pass_agg", pass_agg),
			("pass_agg_tfidf", pass_agg_tfidf),
			("kNN", kNN),
			("kNN_tfidf", kNN_tfidf),
			("rand_for", rand_for),
			("rand_for_tfidf", rand_for_tfidf),
		]
		
		results = []
		for name, model in all_models:
			print "\n...baseline processing:", name
			time_start = time.time()
			results.append((name, accuracy_score(model.fit(X_train, y_train).predict(X_test), y_test)))
			print "Time elapsed: {} seconds".format(time.time()-time_start)
		
		scores = sorted(results, key=lambda (_, x): -x)


		print "\nBaseline Score"
		print tabulate(scores, floatfmt=".4f", headers=("Model", 'Accuracy Score'))
		print

if __name__ == '__main__':
	dl = DataLoader.DataLoader()
	
	gensimLoader = GensimModels.GensimModels()

	

	"""
	Data

	"""

	# Load Data

	print("Loading data...")
	dl = DataLoader.DataLoader()
	id2cult, id2comp, train_cult, train_comp, train_comp_len, test_cult, test_comp, test_comp_len, max_comp_cnt = dl.load_data(Config.path_culture)

	print("Train/Test/Cult/Comp: {:d}/{:d}/{:d}/{:d}".format(len(train_cult), len(test_cult), len(id2cult), len(id2comp)))
	print("==================================================================================")

	X_train = []
	for comp, leng in zip(train_comp, train_comp_len):
		X_train.append(comp[:leng])
	y_train = train_cult

	X_test = []
	for comp, leng in zip(test_comp, test_comp_len):
		X_test.append(comp[:leng])
	y_test = test_cult

	X_train = np.array(X_train)
	y_train = np.array(y_train)

	X_test = np.array(X_test)
	y_test = np.array(y_test)

	"""
	cultures, vocab = dl.load_cultures(Config.path_culture)
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

	for i in X[:10]:
		print i
	#print "total examples %s" % len(y)
	#print "unique labels", len(set(y))
	"""


	"""
	Experiment

	"""

	cultureClassificationWithBaseline = CultureClassificationWithBaseline()
	# Run Cuisine Classification Baselines
	# "svc_tfidf", "svc", "mult_nb", "mult_nb_tfidf", "bern_nb", "bern_nb_tfidf"
	# 
	cultureClassificationWithBaseline.run_baselines(X_train,y_train,X_test,y_test)

	

	"""
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
	"""