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

	def run_baselines(self, X, y):
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
		
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

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

	def benchmark(self, model, X, y, n):
		test_size = 1 - (n / float(len(y)))
		scores = []
		for train, test in StratifiedShuffleSplit(y, n_iter=5, test_size=test_size):
			X_train, X_test = X[train], X[test]
			y_train, y_test = y[train], y[test]
			scores.append(accuracy_score(model.fit(X_train, y_train).predict(X_test), y_test))
		return np.mean(scores)

	def plot_baselines(self):
		train_sizes = [20, 60, 180, 540, 1620, 4860]
		table = []
		for name, model in all_models:
			for n in train_sizes:
				table.append({'model': name, 
						  'accuracy': self.benchmark(model, X, y, n), 
						  'train_size': n})
		df = pd.DataFrame(table)


		plt.figure(figsize=(15, 6))
		fig = sns.pointplot(x='train_size', y='accuracy', hue='model', 
						data=df[df.model.map(lambda x: x in ["svc_tfidf", "svc", "mult_nb", "mult_nb_tfidf", "bern_nb", "bern_nb_tfidf"])])
		
		sns.set_context("notebook", font_scale=1.5)
		fig.set(ylabel="accuracy")
		fig.set(xlabel="labeled training examples")
		fig.set(title="R8 benchmark")
		fig.set(ylabel="accuracy")
		plt.savefig('figure.png')

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