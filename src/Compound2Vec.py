# import libraries
import gensim

# import implemented python files
import Config
from utils import DataLoader, GensimModels, DataPlotter

class Compound2Vec:
	def __init__(self):
		print "Compound2Vec initialized"

	def build_dict_compound2character(self, dict_compounds):
		dict_compound2character = {}
		for comp_id in dict_compounds:
			compound = dict_compounds[comp_id][0]
			char_list = []
			for char in compound:
				char_list.append(char)
			dict_compound2character[compound] = char_list
		return dict_compound2character

	def build_corpus_compound2character(self, dict_compound2character):
		for comp in dict_compound2character:
			#For training data, add tags
			compound = comp
			characters = dict_compound2character[comp]
			yield gensim.models.doc2vec.TaggedDocument(characters, [compound])


if __name__ == '__main__':
	dataLoader = DataLoader.DataLoader()
	gensimLoader = GensimModels.GensimModels()
	comp2vec = Compound2Vec()

	# load list of compounds to dict
	compounds = dataLoader.load_compounds(Config.path_comp_info)

	"""
	Preproccesing

	"""

	# split compounds to characters (dictionary)
	dict_compound2character = comp2vec.build_dict_compound2character(compounds)
	# dict to taggedDocuments
	corpus_compound2character = list(comp2vec.build_corpus_compound2character(dict_compound2character))


	"""
	Build & Save Doc2Vec 

	"""

	# build character-level compounds embeddings with doc2vec
	model_comp2vec = gensimLoader.build_doc2vec(corpus_compound2character)

	# save character-level compounds embeddings with doc2vec
	gensimLoader.save_doc2vec(model=model_comp2vec, path=Config.path_embeddings_compounds)


	"""
	Plot Doc2Vec 

	"""

	# plot data
	DataPlotter.plot_pipeline(model=model_comp2vec, path=Config.path_results, withLegends=False)






