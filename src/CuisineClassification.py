# import libraries


# import implemented python files
import Config
from utils import DataLoader, GensimModels, DataPlotter


class Ingredient2Vec:
	def __init__(self):
		print "\n\n...CuisineClassification initialized"



if __name__ == '__main__':
	dl = DataLoader.DataLoader()
	cuisines = dl.load_cuisine(Config.path_cuisine)

	for i in cuisines:
		print i, cuisines[i]