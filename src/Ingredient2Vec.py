# import libraries
import gensim
import smart_open

# import implemented python files
import Config
from utils import DataLoader, GensimModels, DataPlotter

class Ingredient2Vec:
	def __init__(self):
		print "\nIngredient2Vec initialized"

	# load ingredient-compounds sentence-level parsed file 
	def load_ingredients_sentence(self, fname, tokens_only=False):
		with smart_open.smart_open(fname, encoding="iso-8859-1") as f:
			for i, line in enumerate(f):
				if tokens_only:
					yield gensim.utils.simple_preprocess(line)
				else:
					# For training data, add tags
					line_split = line.split(' ')
					ingredient = line_split[0]
					compounds = ' '.join(line_split[1:])
					yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(compounds), [ingredient])

	def filter_least_compounds(self, corpus, threshold):
		print "\nFiltering ingredeints with the least number() of compounds..."

		list_filtered = []
		for doc_id in range(len(corpus)):
			if len(corpus[doc_id].words) > threshold:
				list_filtered.append(corpus[doc_id])

		print "Before Filtering: ", len(corpus)
		print "After Filtering: ", len(list_filtered)

		return list_filtered

if __name__ == '__main__':
	dataLoader = DataLoader.DataLoader()
	gensimLoader = GensimModels.GensimModels()

	ingr2vec = Ingredient2Vec()

	# load list of compounds to dict
	ingredients = dataLoader.load_compounds(Config.path_ingr_info)
	compounds = dataLoader.load_compounds(Config.path_comp_info)
	relations = dataLoader.load_compounds(Config.path_ingr_comp)
	ingredients_sentence = ingr2vec.load_ingredients_sentence(Config.path_ingr_sentence)

	"""
	Preproccesing

	"""
	corpus_ingredient_compounds = list(ingredients_sentence)
	corpus_ingredient_compounds = ingr2vec.filter_least_compounds(corpus_ingredient_compounds, Config.LEAST_COMPOUNDS)



	"""
	Build & Save Doc2Vec 

	"""

	# build character-level compounds embeddings with doc2vec
	model_ingr2vec = gensimLoader.build_doc2vec(corpus_ingredient_compounds, load_pretrained=True, path_pretrained=Config.path_embeddings_compounds)

	# save character-level compounds embeddings with doc2vec
	gensimLoader.save_doc2vec_only_doc(model=model_ingr2vec, path=Config.path_embeddings_ingredients)

	"""
	Plot Doc2Vec 

	"""

	# plot data
	#DataPlotter.plot_pipeline(model=model_ingr2vec, path=Config.path_results, withLegends=True)

	loaded_model = gensimLoader.load_word2vec(path=Config.path_embeddings_ingredients)






