

# import implemented python files
import Config
from utils import DataLoader, GensimModels, DataPlotter


class IngredientAnalysis:
	def __init__(self, word_vectors):
		print "\nIngredientAnalysis initialized"
		self.word_vectors = word_vectors


	def analogy(self):
		list_most_similar_cosmul = self.word_vectors.most_similar(positive=['orange', 'apple_juice'], negative=['apple'])
		

		print "\nIngredient Analogy"
		for dic in list_most_similar_cosmul:
			word = dic[0]
			score = dic[1]

			if score > 0.5 :
				print word, score
			else:
				print "No similar words"




if __name__ == '__main__':
	
	gensimLoader = GensimModels.GensimModels()
	model = gensimLoader.load_word2vec(path=Config.path_embeddings_ingredients)
	vocab = model.vocab

	"""
	Analyze Intredient2Vec

	"""

	# analgoy test
	ingredientAnalyzer = IngredientAnalysis(model)
	ingredientAnalyzer.analogy()



	"""
	Plot Ingredient2Vec

	"""

	# TSNE
	model_TSNE = DataPlotter.load_TSNE(model)


	# plot data with category
	DataPlotter.plot_category(model=model, model_tsne=model_TSNE, path=Config.path_plottings_ingredients_category, withLegends=True)

	# plot data with clustering
	DataPlotter.plot_clustering(model=model, model_tsne=model_TSNE, path=Config.path_plottings_ingredients_clustering)

	