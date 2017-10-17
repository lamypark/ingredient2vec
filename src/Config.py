import os

# Data Path
data_path = '..' + os.sep + 'data'
ingr_info = data_path + os.sep + 'scientific_report' + os.sep + 'ingr_info.tsv'
comp_info = data_path + os.sep + 'scientific_report' + os.sep + 'comp_info.tsv'
ingr_comp = data_path + os.sep + 'scientific_report' + os.sep +'ingr_comp.tsv'

# Result Path
results_path = ".." + os.sep + "results"



# Minibatch Size
BATCH_SIZE = 32
# Gradient clip threshold
GRAD_CLIP = 10
# Learning rate
LEARNING_RATE = 0.0005
# Maximum number of steps in BPTT
GRAD_STEPS = -1
# Number of epochs for training
NUM_EPOCHS = 10
# do validation every VALIDATION_FREQ iterations
VALIDATION_FREQ = 100
# maximum word length for character model
MAX_WORD_LEN = 10


