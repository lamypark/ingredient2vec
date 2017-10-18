import os

# Data Path
path_data = '..' + os.sep + 'data'
path_ingr_info = path_data + os.sep + 'scientific_report' + os.sep + 'ingr_info.tsv'
path_comp_info = path_data + os.sep + 'scientific_report' + os.sep + 'comp_info.tsv'
path_ingr_comp = path_data + os.sep + 'scientific_report' + os.sep +'ingr_comp.tsv'
path_ingr_sentence = path_data + os.sep + 'scientific_report' + os.sep +'ingredient_with_compounds_sentence_level'

# Result Path
path_results = ".." + os.sep + "results"
path_embeddings_compounds = path_results + os.sep + 'embeddings' + os.sep + 'embeddings_compounds_50.bin'
path_embeddings_ingredients = path_results + os.sep + 'embeddings' + os.sep + 'embeddings_ingredients_50.bin'

# Pre-Trained Embedding Path
path_embeddings_compounds = path_results + os.sep + 'embeddings' + os.sep + 'embeddings_compounds_50.bin'
path_embeddings_ingredients = path_results + os.sep + 'embeddings' + os.sep + 'embeddings_ingredients_50.bin'

# Ingredient-Compounds Pair / Threshold for Least Compounds
LEAST_COMPOUNDS = 10