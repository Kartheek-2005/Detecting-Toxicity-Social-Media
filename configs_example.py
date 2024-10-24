"""
Contains configurations for the project.

This is a template for `configs.py`.
Fill in the values and save the file as `configs.py`.
"""

# Pinecone API key and index name for connection
PINECONE_API_KEY = ""
PINECONE_INDEX = ""

# Namespaces for toxic and benign examples
TOXIC_NAMESPACE = "toxic"
BENIGN_NAMESPACE = "benign"

# Prompt template for creating prompts
PROMPT_TEMPLATE = "Based on the given toxic examples and benign examples, predict if the following text is toxic or benign:{toxic}{benign}{text}"

# Number of epochs and batch size for training the model
EPOCHS = 10
BATCH_SIZE = 8

# Learning rate for training the model
LEARNING_RATE = 1e-5

# Configurations for ReduceLROnPlateau scheduler
FACTOR = 0.1
PATIENCE = 2
THRESHOLD = 1e-3
