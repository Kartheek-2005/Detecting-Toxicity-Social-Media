'''
Contains configurations for the project.

This is a template for `configs.py`.
Fill in the values and save the file as `configs.py`.
'''

# Pinecone API key and index name for connection
PINECONE_API_KEY = ''
PINECONE_INDEX = ''

# Path to toxic and benign databases
TOXIC_DB_PATH = 'database/toxic.pkl'
BENIGN_DB_PATH = 'database/benign.pkl'

# Prompt template for creating prompts (must contain `{text}`)
PROMPT_TEMPLATE = '{text}'

# Number of epochs and batch size for training the model
EPOCHS = 10
BATCH_SIZE = 8

# Learning rate for training the model
LEARNING_RATE = 5e-5

# Configurations for ReduceLROnPlateau scheduler
FACTOR = .1
PATIENCE = 2
THRESHOLD = 1e-3
