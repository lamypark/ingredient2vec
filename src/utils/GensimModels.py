import time
import gensim
from gensim.models.keyedvectors import KeyedVectors
import multiprocessing

import Config

class GensimModels():
	
	"""
	Train Doc2Vec Model

	"""
	def build_doc2vec(self, corpus, load_pretrained=False, path_pretrained=""):
		print "\nStart to build Doc2Vec Models with Gensim..."

		time_start = time.time()
		cores = multiprocessing.cpu_count()

		#dm/m,d50,n5,w5,mc5,s0.001,t3
		#model = gensim.models.doc2vec.Doc2Vec(size=50, min_count=5, iter=55)

		# PV-DM w/ average
		model = gensim.models.doc2vec.Doc2Vec(size=50, window=Config.FILTERING, min_count=Config.FILTERING, iter=100)
		model.build_vocab(corpus, keep_raw_vocab=False)

		print "Unique Words Count:", len(model.wv.vocab)
		print "Total Documents Count:", model.corpus_count		

		if load_pretrained:
			model.intersect_word2vec_format(path_pretrained, lockf=0.0, binary=True, encoding='utf8', unicode_errors='strict')

		print "\nTraining Started..."
		model.train(corpus, total_examples=model.corpus_count, epochs=model.iter)

		print "Doc2Vec training done!"
		print "Time elapsed: {} seconds".format(time.time()-time_start)

		return model

	def load_word2vec(self, path):
		model = KeyedVectors.load_word2vec_format(path, binary=True)
		return model

	def save_doc2vec(self, model, path):
		model.save_word2vec_format(path, doctag_vec=True, word_vec=True, prefix='*dt_', fvocab=None, binary=True)

	def save_doc2vec_only_doc(self, model, path):
		model.save_word2vec_format(path, doctag_vec=True, word_vec=False, prefix='*dt_', fvocab=None, binary=True)