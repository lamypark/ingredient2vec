import os
import collections
import smart_open

import Config


"""
Load basic ingredients and compounds data from Nature Scientific Report(Ahn, 2011)

"""
class DataLoader:

	# {ingredient_id: [ingredient_name, ingredient_category]}
	def load_ingredients(self, path):
		ingredients = {}
		ingredients_list = []
		with open(path, 'r') as f:
			for line in f:
				if line[0] == '#':
					pass
				else:
					line_split = line.rstrip().split('\t')
					ingredients_id = line_split[0]
					ingredients_list = line_split[1:]
					ingredients[ingredients_id] = ingredients_list
		return ingredients

	# {compound_id: [compound_name, CAS_number]}
	def load_compounds(self, path):
		compounds = {}
		compounds_list = []
		with open(path, 'r') as f:
			for line in f:
				if line[0] == '#':
					pass
				else:
					line_split = line.rstrip().split('\t')
					compounds_id = line_split[0]
					compounds_list = line_split[1:]
					compounds[compounds_id] = compounds_list
		return compounds

	# {ingredient_id: [compound_id1, compound_id2, ...] }
	def load_relations(self, path):
		relations = {}
		with open(path, 'r') as f:
			for line in f:
				if line[0] == '#':
					pass
				else:
					line_split = line.rstrip().split('\t')
					ingredient_id = line_split[0]
					compound_id = line_split[1]
					
					if ingredient_id in relations:
						relations[ingredient_id].append(compound_id)
						
					else:
						relations[ingredient_id] = [compound_id]
						
		return relations

	# Ingredient_to_category
	def ingredient_to_category(self, tag, ingredients):
		for ingr_id in ingredients:
			if ingredients[ingr_id][0] == tag:
				return ingredients[ingr_id][1]
			else: 
				continue
		return


	# Corpus tag to index
	def tag_to_index(tags, corpus):
		for doc_id in range(len(corpus)):
			if tags == corpus[doc_id].tags[0]:
				return doc_id
			else:
				continue
		return
		 

	# Corpus index to tag					
	def index_to_tag(index, corpus):
		return corpus[index].tags


	# Cuisine - Ingredients
	def load_cuisine(self, path):
		cuisines = {}
		ingredient_list = []
		vocab = []

		with open(path, 'r') as f:
			cuisine_id = 0
			for cuisine_id, line in enumerate(f):
				if line[0] == '#':
					pass
				else:
					line_split = line.rstrip().split(',')
					cuisine_label = line_split[0]
					ingredient_list = line_split[1:]
					cuisines[cuisine_id] = [ingredient_list, [cuisine_label]]
					for ingr in ingredient_list:
						vocab.append(ingr)

		return cuisines, set(vocab)



if __name__ == '__main__':
	dl = DataLoader()
	#ingredients = dl.load_ingredients(Config.path_ingr_info)
	#compounds = dl.load_compounds(Config.path_comp_info)
	#relations = dl.load_relations(Config.path_ingr_comp)

	cuisines = dl.load_cuisine(Config.path_cuisine)