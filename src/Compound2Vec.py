# import libraries
import gensim
import requests

# import implemented python files
import Config
from utils import DataLoader, GensimModels, DataPlotter

class Compound2Vec:
	def __init__(self):
		print "Compound2Vec initialized"

	# split compounds to characters (dictionary)
	def build_dict_compound2character(self, dict_compounds):
		dict_compound2character = {}
		for comp_id in dict_compounds:
			compound_name = dict_compounds[comp_id][0]
			char_list = []
			for char in compound_name:
				char_list.append(char)
			dict_compound2character[compound_name] = char_list
		return dict_compound2character

	def build_corpus_compound2character(self, dict_compound2character):
		for comp in dict_compound2character:
			#For training data, add tags
			compound = comp
			characters = dict_compound2character[comp]
			yield gensim.models.doc2vec.TaggedDocument(characters, [compound])

	# split compounds to characters (dictionary)
	def build_dict_compound2inchikey(self, dict_compounds):
		dict_compound2character = {}
		for comp_id in dict_compounds:
			compound_name = dict_compounds[comp_id][0]
			inchikey = dict_compounds[comp_id][2]
			char_list = []
			for char in inchikey:
				char_list.append(char)

			if len(char_list) == 27:
				dict_compound2character[compound_name] = char_list
			else:
				continue
		return dict_compound2character

	def build_corpus_compound2inchikey(self, dict_compound2inchikey):
		for comp in dict_compound2inchikey:
			#For training data, add tags
			compound = comp
			characters = dict_compound2inchikey[comp]
			yield gensim.models.doc2vec.TaggedDocument(characters, [compound])


if __name__ == '__main__':
	dataLoader = DataLoader.DataLoader()
	gensimLoader = GensimModels.GensimModels()
	comp2vec = Compound2Vec()

	"""
	Mode Description

	# mode 1 : Embed Chemical Compounds with Characters
	# mode 2 : Embed Chemical Compounds with FingerPrints
	# mode 999 : Plot Loaded Word2Vec or Doc2vec
	"""

	mode = 1

	if mode == 1:
		"""
		Load Data
		"""
		# load list of compounds to dict
		compounds = dataLoader.load_compounds(Config.path_comp_info)


		"""
		Preproccesing

		"""
		dict_compound2character = comp2vec.build_dict_compound2character(compounds)

		# dict to taggedDocuments
		corpus_compound2character = list(comp2vec.build_corpus_compound2character(dict_compound2character))


		"""
		Build & Save Doc2Vec

		"""

		# build character-level compounds embeddings with doc2vec
		model_comp2vec = gensimLoader.build_doc2vec(corpus_compound2character)

		# save character-level compounds embeddings with doc2vec
		gensimLoader.save_doc2vec_only_doc(model=model_comp2vec, path=Config.path_embeddings_compounds_char)

	elif mode == 2:

		compounds_updated = dataLoader.load_compounds_updated(Config.path_comp_info_updated)

		"""
		Preproccesing

		"""

		dict_compound2inchikey = comp2vec.build_dict_compound2inchikey(compounds_updated)

		build_corpus_compound2inchikey = list(comp2vec.build_corpus_compound2character(dict_compound2inchikey))

		"""
		Build & Save Doc2Vec

		"""

		# build character-level compounds embeddings with doc2vec
		model_comp2vec = gensimLoader.build_doc2vec(build_corpus_compound2inchikey)

		# save character-level compounds embeddings with doc2vec
		gensimLoader.save_doc2vec_only_doc(model=model_comp2vec, path=Config.path_embeddings_compounds_inchi)



	elif mode == 999:

		"""
		Plot Ingredient2Vec

		"""
		model_loaded = gensimLoader.load_word2vec(path=Config.path_embeddings_compounds_char)
		model_tsne = DataPlotter.load_TSNE(model_loaded, dim=2)
		DataPlotter.plot_category(model_loaded, model_tsne, Config.path_plottings_compounds_category_char, withLegends=False)
		#DataPlotter.plot_clustering(model_loaded, model_tsne, Config.path_plottings_ingredients_clustering)

	else:
		print "Please specify the mode you want."
