import os

DATE = 1801041938

# Building Training Set Paramter
FILTERING = 5					# Ingredient-Compounds Pair / Threshold for Least Compounds
RANDOM_SAMPLING = False			# Whether to Random Sample or not
NUM_SAMPLING = 50				# Number of Random Sampling

# Word2Vec Model Parameter
WOR_DIM = 50

# Doc2Vec Model Parameter
CHAR_EMB = True				# Use pre-trained character-level embeddings
DOC_DIM = 10

"""
General Paths
"""

# Result Path
path_results = ".." + os.sep + "results"

# Data Path
path_data = '..' + os.sep + 'data'


"""
Paths for Chemical Compounds Embedding
"""

# File Name Protocol
file_name_compounds_embeddings = '_f-' + str(FILTERING) + '_rs-' + str(RANDOM_SAMPLING) + '_ns-' + str(NUM_SAMPLING) + '_ifcharemb-' + str(CHAR_EMB) + '_dim-' + str(DOC_DIM) + '_' + '1801031522' 

path_comp_info = path_data + os.sep + 'scientific_report' + os.sep + 'comp_info.tsv'

# Embeddings for Chemical Compounds
# Chemical Compounds Embedding with Random Initialization
path_embeddings_compounds_rnd = path_results + os.sep + 'embeddings' + os.sep + 'embeddings_compounds_rnd' + file_name_compounds_embeddings + '.bin'

# Chemical Compounds Embedding with Character-level Embedding
path_embeddings_compounds_char = path_results + os.sep + 'embeddings' + os.sep + 'embeddings_compounds_char' + file_name_compounds_embeddings + '.bin'

# Chemical Compounds Embedding with InChiKey Embedding
path_embeddings_compounds_inchi = path_results + os.sep + 'embeddings' + os.sep + 'embeddings_compounds_inchi' + file_name_compounds_embeddings + '.bin'

# Plottings
path_plottings_compounds_category_char = path_results + os.sep + 'plot_compound_embeddings_category_char' + file_name_compounds_embeddings
path_plottings_compounds_category_inchi = path_results + os.sep + 'plot_compound_embeddings_category_inchi' + file_name_compounds_embeddings


path_plottings_compounds_clustering = path_results + os.sep + 'plot_compound_embeddings_cluster' + file_name_compounds_embeddings


"""
Paths for Ingredients Embedding
"""

# File Name Protocol
file_name_ingredient_embeddings = '_f-' + str(FILTERING) + '_rs-' + str(RANDOM_SAMPLING) + '_ns-' + str(NUM_SAMPLING) + '_ifcharemb-' + str(CHAR_EMB) + '_dim-' + str(DOC_DIM) + '_' + str(DATE) 

# Data Path
path_ingr_info = path_data + os.sep + 'scientific_report' + os.sep + 'ingr_info.tsv'
path_comp_info = path_data + os.sep + 'scientific_report' + os.sep + 'comp_info.tsv'
path_comp_info_updated = path_data + os.sep + 'scientific_report' + os.sep + 'comp_info_updated.tsv'
path_ingr_comp = path_data + os.sep + 'scientific_report' + os.sep +'ingr_comp.tsv'

path_culture = path_data + os.sep + 'kaggle_and_nature.csv'

# Embeddings for Ingredients
# Ingredient embedding with chemical compounds
path_embeddings_ingredients_cc_rnd = path_results + os.sep + 'embeddings' + os.sep + 'embeddings_ingredients_cc_rnd' + file_name_ingredient_embeddings + '.bin'
path_embeddings_ingredients_cc_char = path_results + os.sep + 'embeddings' + os.sep + 'embeddings_ingredients_cc_char' + file_name_ingredient_embeddings + '.bin'
path_embeddings_ingredients_cc_inchi = path_results + os.sep + 'embeddings' + os.sep + 'embeddings_ingredients_cc_inchi' + file_name_ingredient_embeddings + '.bin'
# Ingredient embedding with other ingredient contexts
path_embeddings_ingredients_ic = path_results + os.sep + 'embeddings' + os.sep + 'embeddings_ingredients_ic' + file_name_ingredient_embeddings + '.bin'


# Plottings
path_plottings_ingredients_category_cc_rnd = path_results + os.sep + 'plot_ingredient_embeddings_category_cc_rnd' + file_name_ingredient_embeddings
path_plottings_ingredients_category_cc_char = path_results + os.sep + 'plot_ingredient_embeddings_category_cc_char' + file_name_ingredient_embeddings
path_plottings_ingredients_category_cc_inchi = path_results + os.sep + 'plot_ingredient_embeddings_category_cc_inchi' + file_name_ingredient_embeddings
path_plottings_ingredients_clustering = path_results + os.sep + 'plot_ingredient_embeddings_cluster' + file_name_ingredient_embeddings