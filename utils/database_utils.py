import pickle
from collections.abc import Callable

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer

class DatabaseInterface:
  '''
  Database class to handle all database operations.

  Attributes:
    model_name (str): Name of the pre-trained transformer model used for embedding.
    embed_model (transformers.PreTrainedModel): Pre-trained transformer model for embeddings.
    tokenizer (transformers.PreTrainedTokenizer): Tokenizer for pre-trained transformer model.
    n_features (int): Dimensionality of the embeddings produced by the model.
    texts (list[str]): List of input texts.
    embeddings (numpy.ndarray): Embeddings for the texts.
    nn (sklearn.neighbors.NearestNeighbors): Nearest neighbors model for finding similar embeddings.
    n_neighbors (int): Number of nearest neighbors to return when querying.
  '''

  # Pre-trained model and tokenizer for embedding generation
  model_name = 'sentence-transformers/all-MiniLM-L6-v2'
  embed_model = SentenceTransformer(model_name)
  n_features = embed_model.get_sentence_embedding_dimension()

  def __init__(
    self,
    n_neighbors: int = 2,
    data_path: str = 'data.pkl',
    metric: Callable[[np.ndarray, np.ndarray], float] = np.dot
  ) -> None:
    '''
    Initialize the Database object with optional initial texts and pre-loaded data from a cache.

    :param texts (list[str], optional): Initial list of texts to be added to the database (default is an empty list).
    n_neighbors (int, optional): Number of neighbors for the NearestNeighbors model (default is 2).
    cache_path (str, optional): Path to the cache file that stores the texts and embeddings (default is 'data.pkl').
    '''
    self.n_neighbors = n_neighbors

    # Load cached data if available
    try:
      self.data_fp = open(data_path, 'rb+')
      data = pickle.load(self.data_fp)
      self.texts: list[str] = data['texts']
      self.embeddings: np.ndarray = data['embeddings']
    except:
      self.data_fp = open(data_path, 'wb')
      self.texts = []
      self.embeddings = np.empty((0, Database.n_features))

    # Initialize the NearestNeighbors model
    self.nn = NearestNeighbors(n_neighbors=self.n_neighbors, metric=metric)
    if(len(self.texts) > 0):
      self.nn.fit(self.embeddings)
  
  def __del__(self) -> None:
    '''
    Close the file pointer to the cache file when the object is deleted.
    '''
    self.data_fp.flush()
    self.data_fp.close()

  @staticmethod  
  def embed(texts: list[str]) -> np.ndarray:
    '''
    Embed a list of texts into vector representations using a pre-trained transformer model.

    Args:
      texts (list[str]): List of texts to be embedded.

    Returns:
      np.ndarray: The generated embeddings for each text, with shape (num_texts, embedding_size).
    '''
    if(len(texts) == 0):
      return np.empty((0, Database.n_features))
    embeddings = Database.embed_model.encode(texts, convert_to_numpy=True)
    return embeddings

  def insert(
    self,
    texts: list[str]
  ) -> None:
    '''
    Insert new texts into the database and update the embeddings.

    Args:
      texts (list[str]): A list of texts to insert into the database.

    Returns:
      None
    '''
    # Embed the new texts
    embedding = Database.embed(texts)

    # Update the database
    self.texts.extend(texts)
    self.embeddings = np.vstack((self.embeddings, embedding))

    # Update the NearestNeighbors model
    if(len(self.texts) > 0):
      self.nn.fit(self.embeddings)

    # Save the updated data to the data file
    self.data_fp.truncate(0)
    pickle.dump(
      {'texts': self.texts, 'embeddings': self.embeddings},
      self.data_fp
    )

  def query(
    self,
    q_texts: list[str]
  ) -> list[list[str]]:
    '''
    Query the database for the nearest neighbors of a list of input texts.

    Args:
      q_texts (list[str]): A list of query texts to find the nearest neighbors for.

    Returns:
      np.ndarray: Indices of the nearest neighbors for each query text, with shape (num_queries, n_neighbors).
    '''
    # Embed the query texts
    embeddings = Database.embed(q_texts)

    # Find the nearest neighbors
    _, indices = self.nn.kneighbors(embeddings, n_neighbors=self.n_neighbors)

    # Get texts
    return [self.get_texts(i) for i in indices]

  def get_texts(self, indices: np.ndarray) -> list[str]:
    '''
    Get the texts corresponding to the given indices.

    Args:
      indices (np.ndarray): Indices of the texts to retrieve.

    Returns:
      list[str]: The texts corresponding to the given indices.
    '''
    return [self.texts[i] for i in indices]
  
  def get_embeddings(self, indices: np.ndarray) -> np.ndarray:
    '''
    Get the embeddings corresponding to the given indices.

    Args:
      indices (np.ndarray): Indices of the embeddings to retrieve.

    Returns:
      np.ndarray: The embeddings corresponding to the given indices.
    '''
    return self.embeddings[indices]
  
  def clear_cache(self) -> None:
    '''
    Clear the cache file storing the texts and embeddings.

    Returns:
      None
    '''
    self.data_fp.truncate(0)
