# import libraries
import gensim
import smart_open
import random
from itertools import combinations

# import implemented python files
import Config
from utils import DataLoader, GensimModels, DataPlotter

class Ingredient2Vec:
	def __init__(self):
		print "\n\n...Ingredient2Vec initialized"

	def num_combinations(self, n, k):
		return math.factorial(n) / (math.factorial(k) * math.factorial(n-k))

	def build_taggedDocument(self, ingredients, compounds, relations, filtering=5, random_sampling=False, num_sampling=0):
		
		compound_length_list = []
		
		print '\n\n...Building Training Set'

		# relations
		for ingr in relations:

			# ingredients
			ingredient_name, ingredient_category = ingredients[ingr][0], ingredients[ingr][1]

			# compounds in a ingredient
			compound_list = []
			for comp in relations[ingr]:
				compound_name, compound_cas = compounds[comp][0], compounds[comp][1]
				compound_list.append(compound_name)

			# filter by number of compounds
			if len(compound_list) > filtering:
				compound_length_list.append(len(compound_list))

				# Random Sampling
				if random_sampling:
					#print ingredient_name, len(compound_list), len(compound_list)/num_sampling*3
					
					#all the combinations
					#sample_list = ["".join(x) for x in combinations(compound_list, 5)]
					#print len(sample_list)

					#sample randomly
					for i in xrange(num_sampling):
						sampled_compounds = random.sample(compound_list, filtering)
						yield gensim.models.doc2vec.TaggedDocument(compound_list, [ingredient_name])
						

				# Just Sampling
				else:
					yield gensim.models.doc2vec.TaggedDocument(compound_list, [ingredient_name])

				
		print 'Sampling %d' % (num_sampling) 
		print 'Filter %d' % (filtering) 
		print 'Number of ingredients : %d' % (len(compound_length_list))
		print 'Average Length of Compounds: %f' % (reduce(lambda x, y: x + y, compound_length_list) / float(len(compound_length_list)))

if __name__ == '__main__':
	dataLoader = DataLoader.DataLoader()
	gensimLoader = GensimModels.GensimModels()

	ingr2vec = Ingredient2Vec()

	# load list of compounds to dict
	ingredients = dataLoader.load_ingredients(Config.path_ingr_info)
	compounds = dataLoader.load_compounds(Config.path_comp_info)
	relations = dataLoader.load_relations(Config.path_ingr_comp)
 

	"""
	Preproccesing

	"""

	# build taggedDocument form of corpus
	corpus_ingredient_compounds = list(ingr2vec.build_taggedDocument(ingredients, compounds, relations, filtering=Config.FILTERING, random_sampling=Config.RANDOM_SAMPLING, num_sampling=Config.NUM_SAMPLING))

	"""
	Build & Save Doc2Vec 

	"""

	# build ingredient embeddings with doc2vec
	model_ingr2vec = gensimLoader.build_doc2vec(corpus_ingredient_compounds, load_pretrained=Config.PRE_TRAIN, path_pretrained=Config.path_embeddings_compounds)

	# save character-level compounds embeddings with doc2vec
	gensimLoader.save_doc2vec_only_doc(model=model_ingr2vec, path=Config.path_embeddings_ingredients)
	model_loaded = gensimLoader.load_word2vec(path=Config.path_embeddings_ingredients)

	"""
	Plot Ingredient2Vec

	"""
	model_tsne = DataPlotter.load_TSNE(model_loaded, dim=2)
	#DataPlotter.plot_category(model_loaded, model_tsne, Config.path_plottings_ingredients_category, withLegends=True)
	DataPlotter.plot_clustering(model_loaded, model_tsne, Config.path_plottings_ingredients_clustering)


