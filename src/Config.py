import os


# Building Training Set Paramter
FILTERING = 5					# Ingredient-Compounds Pair / Threshold for Least Compounds
RANDOM_SAMPLING = True			# Whether to Random Sample or not
NUM_SAMPLING = 50				# Number of Random Sampling


# Doc2Vec Model Parameter
PRE_TRAIN = True				# Use pre-trained character-level embeddings


# Data Path
path_data = '..' + os.sep + 'data'
path_ingr_info = path_data + os.sep + 'scientific_report' + os.sep + 'ingr_info.tsv'
path_comp_info = path_data + os.sep + 'scientific_report' + os.sep + 'comp_info.tsv'
path_ingr_comp = path_data + os.sep + 'scientific_report' + os.sep +'ingr_comp.tsv'

path_culture = path_data + os.sep + 'kaggle_and_nature.csv'

# Result Path
path_results = ".." + os.sep + "results"

# Embeddings
path_embeddings_compounds = path_results + os.sep + 'embeddings' + os.sep + 'embeddings_compounds_50.bin'
path_embeddings_ingredients = path_results + os.sep + 'embeddings' + os.sep + 'embeddings_ingredients_f-' + str(FILTERING) + '_rs-' + str(RANDOM_SAMPLING) + '_ns-' + str(NUM_SAMPLING) + '_charemb-' + str(PRE_TRAIN) + '_dim50.bin'

# Plottings
path_plottings_ingredients_category = path_results + os.sep + 'plot_ingredient_embeddings_category_f-' + str(FILTERING) + '_rs-' + str(RANDOM_SAMPLING) + '_ns-' + str(NUM_SAMPLING) + '_charemb-' + str(PRE_TRAIN)
path_plottings_ingredients_clustering = path_results + os.sep + 'plot_ingredient_embeddings_cluster_f-' + str(FILTERING) + '_rs-' + str(RANDOM_SAMPLING) + '_ns-' + str(NUM_SAMPLING) + '_charemb-' + str(PRE_TRAIN)

