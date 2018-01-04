import time
import gensim
from gensim.models.keyedvectors import KeyedVectors
import multiprocessing

import Config

class GensimModels():

	"""
	Train Word2Vec Model with Gensim
	"""

	def build_word2vec(self, corpus, load_pretrained=False, path_pretrained=""):
		print "\n\n...Start to build Word2Vec Models with Gensim"

		time_start = time.time()
		cores = multiprocessing.cpu_count()

		model = gensim.models.Word2Vec(corpus, size=Config.WOR_DIM, window=Config.FILTERING, min_count=Config.FILTERING, workers=cores)

		print "Word Embedding Dimension:", Config.WOR_DIM
		print "Word Window & Filtering:", Config.FILTERING

		print "Unique Words Count:", len(model.wv.vocab)

		return model
		
	def save_word2vec(self, model, path):
		print "\n\n...Save Word2Vec with a file name of", path
		model.wv.save_word2vec_format(path, binary=True)

	def load_word2vec(self, path):
		print "\n\n...Load Word2Vec with a file name of", path
		model = KeyedVectors.load_word2vec_format(path, binary=True)
		return model

	"""
	Train Doc2Vec Model with Gensim

	"""
	def build_doc2vec(self, corpus, load_pretrained=False, path_pretrained=""):
		print "\n\n...Start to build Doc2Vec Models with Gensim"

		time_start = time.time()
		cores = multiprocessing.cpu_count()

		#dm/m,d50,n5,w5,mc5,s0.001,t3
		#model = gensim.models.doc2vec.Doc2Vec(size=50, min_count=5, iter=55)

		# PV-DM w/ average
		"""
		class gensim.models.word2vec.Word2Vec
		(sentences=None, size=100, alpha=0.025, window=5, min_count=5, max_vocab_size=None, 
		sample=0.001, seed=1, workers=3, min_alpha=0.0001, sg=0, hs=0, negative=5, cbow_mean=1, 
		hashfxn=<built-in function hash>, iter=5, null_word=0, trim_rule=None, sorted_vocab=1, 
		batch_words=10000, compute_loss=False)
		"""
		model = gensim.models.doc2vec.Doc2Vec(size=Config.DOC_DIM, window=Config.FILTERING, min_count=Config.FILTERING, negative=5,  alpha=0.025, min_alpha=0.001, workers=cores, iter=100, 
												dbow_words=1, dm_mean=0)
		model.build_vocab(corpus)

		#print len(model.wv.vocab)


		if load_pretrained:
			#model_loaded = self.load_word2vec(path_pretrained)
			print "...Update Input Vectors with Pre-Trained Vectors:", path_pretrained
			model.intersect_word2vec_format(path_pretrained, lockf=0.0, binary=True, encoding='utf8', unicode_errors='strict')

		#print len(model)

		#print model.vocab



		print "Document Embedding Dimension:", Config.DOC_DIM
		print "Document Window & Filtering:", Config.FILTERING


		print "Unique Words Count:", len(model.wv.vocab)
		print "Total Documents Count:", model.corpus_count



		print "\n\n...Training Started"
		model.train(corpus, total_examples=model.corpus_count, epochs=model.iter)

		print "Doc2Vec training done!"
		print "Time elapsed: {} seconds".format(time.time()-time_start)

		return model

	def build_doc2vec_compounds(self, corpus, load_pretrained=False, path_pretrained=""):
		print "\n\n...Start to build Doc2Vec Models with Gensim"

		time_start = time.time()
		cores = multiprocessing.cpu_count()

		#dm/m,d50,n5,w5,mc5,s0.001,t3
		#model = gensim.models.doc2vec.Doc2Vec(size=50, min_count=5, iter=55)

		# PV-DM w/ average
		"""
		class gensim.models.word2vec.Word2Vec
		(sentences=None, size=100, alpha=0.025, window=5, min_count=5, max_vocab_size=None, 
		sample=0.001, seed=1, workers=3, min_alpha=0.0001, sg=0, hs=0, negative=5, cbow_mean=1, 
		hashfxn=<built-in function hash>, iter=5, null_word=0, trim_rule=None, sorted_vocab=1, 
		batch_words=10000, compute_loss=False)
		"""
		model = gensim.models.doc2vec.Doc2Vec(size=Config.DOC_DIM, window=Config.FILTERING, min_count=Config.FILTERING, negative=5,  alpha=0.025, min_alpha=0.001, workers=cores, iter=100, 
												dbow_words=1, dm_mean=0)
		model.build_vocab(corpus, keep_raw_vocab=True)

		print "Document Embedding Dimension:", Config.DOC_DIM
		print "Document Window & Filtering:", Config.FILTERING


		print "Unique Words Count:", len(model.wv.vocab)
		print "Total Documents Count:", model.corpus_count

		if load_pretrained:
			model.intersect_word2vec_format(path_pretrained, lockf=0.0, binary=True, encoding='utf8', unicode_errors='strict')

		print "\n\n...Training Started"
		model.train(corpus, total_examples=model.corpus_count, epochs=model.iter)

		print "Doc2Vec training done!"
		print "Time elapsed: {} seconds".format(time.time()-time_start)

		return model

	def save_doc2vec(self, model, path):
		model.save_word2vec_format(path, doctag_vec=True, word_vec=True, prefix='*dt_', fvocab=None, binary=True)

	def save_doc2vec_only_doc(self, model, path):
		model.save_word2vec_format(path, doctag_vec=True, word_vec=False, prefix='', fvocab=None, binary=True)
