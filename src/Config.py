import os

DATE = 1712141252

DATE = 1712141347

# Building Training Set Paramter
FILTERING = 5					# Ingredient-Compounds Pair / Threshold for Least Compounds
RANDOM_SAMPLING = True			# Whether to Random Sample or not
NUM_SAMPLING = 50				# Number of Random Sampling

# Doc2Vec Model Parameter
<<<<<<< HEAD
CHAR_EMB = False                # Use pre-trained character-level embeddings
=======
CHAR_EMB = False				# Use pre-trained character-level embeddings
>>>>>>> c14c8b3c29015e9f529348642ebc24ce93f4218c
DOC_DIM = 10

# Data Path
path_data = '..' + os.sep + 'data'
path_ingr_info = path_data + os.sep + 'scientific_report' + os.sep + 'ingr_info.tsv'
path_comp_info = path_data + os.sep + 'scientific_report' + os.sep + 'comp_info.tsv'
path_ingr_comp = path_data + os.sep + 'scientific_report' + os.sep +'ingr_comp.tsv'

path_culture = path_data + os.sep + 'kaggle_and_nature.csv'

# Result Path
path_results = ".." + os.sep + "results"

<<<<<<< HEAD
# File Name Protocol
file_name = '_f-' + str(FILTERING) + '_rs-' + str(RANDOM_SAMPLING) + '_ns-' + str(NUM_SAMPLING) + '_ifcharemb-' + str(CHAR_EMB) + '_dim-' + str(DOC_DIM) + '_' + str(DATE) + '.bin'

=======

# File Name Protocol
file_name = '_f-' + str(FILTERING) + '_rs-' + str(RANDOM_SAMPLING) + '_ns-' + str(NUM_SAMPLING) + '_ifcharemb-' + str(CHAR_EMB) + '_dim-' + str(DOC_DIM) + '_' + str(DATE) + '.bin'


>>>>>>> c14c8b3c29015e9f529348642ebc24ce93f4218c
# Embeddings
path_embeddings_compounds = path_results + os.sep + 'embeddings' + os.sep + 'embeddings_compounds_50.bin'
path_embeddings_ingredients = path_results + os.sep + 'embeddings' + os.sep + 'embeddings_ingredients' + file_name

# Plottings
path_plottings_ingredients_category = path_results + os.sep + 'plot_ingredient_embeddings_category' + file_name
path_plottings_ingredients_clustering = path_results + os.sep + 'plot_ingredient_embeddings_cluster' + file_name
<<<<<<< HEAD
=======

>>>>>>> c14c8b3c29015e9f529348642ebc24ce93f4218c
