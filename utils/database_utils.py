import os
import pickle
from collections.abc import Callable

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone



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
    data_path: str = 'data.pkl',
    n_neighbors: int = 2,
    metric: Callable[[np.ndarray, np.ndarray], float] = np.dot
  ) -> None:
    '''
    Initialize the Database object with optional initial texts and pre-loaded data from a cache.

    :param texts (list[str], optional): Initial list of texts to be added to the database (default is an empty list).
    n_neighbors (int, optional): Number of neighbors for the NearestNeighbors model (default is 2).
    cache_path (str, optional): Path to the cache file that stores the texts and embeddings (default is 'data.pkl').
    '''
    self.n_neighbors = n_neighbors

    # Load data if available
    try:

      self.data_fp = open(data_path, 'rb+')
      data = pickle.load(self.data_fp)
      self.texts: list[str] = data['texts']
      self.embeddings: np.ndarray = data['embeddings']
    
    # Create a new data file if it doesn't exist
    except:

      # Create the directory if it doesn't exist
      dirs, _ = os.path.split(data_path)
      if dirs:
        os.makedirs(dirs, exist_ok=True)
      
      # Make new data file
      self.data_fp = open(data_path, 'wb')

      # Initialize the database
      self.texts = []
      self.embeddings = np.empty((0, DatabaseInterface.n_features))

    # Initialize the NearestNeighbors model
    self.nn = NearestNeighbors(n_neighbors=self.n_neighbors, metric=metric)
    if(len(self.texts) > 0):
      self.nn.fit(self.embeddings)
  
  def __del__(self) -> None:
    '''
    Flushes and closes the data file when the object is deleted.
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
      return np.empty((0, DatabaseInterface.n_features))
    embeddings = DatabaseInterface.embed_model.encode(texts, convert_to_numpy=True)
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
    embedding = DatabaseInterface.embed(texts)

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
    embeddings = DatabaseInterface.embed(q_texts)

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
  
  def truncate(self) -> None:
    '''
    Truncates the data file.
    '''
    self.data_fp.truncate(0)
  


class PineconeInterface:
  '''
  Class to interact with Pinecone API.

  :param str api_key: Pinecone API key.
  :param str index_name: Name of the index to connect to.
  '''

  def __init__(
    self,
    api_key: str,
    index_name: str
  ) -> None:

    self.index_name = index_name

    # Get Pinecone client
    self.pc = Pinecone(api_key)

    # Connect to index
    self.index = self.pc.Index(index_name)

    # Get number of vectors in each namespace
    self.num_vectors = {
      namespace: info.vector_count
      for namespace, info in self.index.describe_index_stats().namespaces.items()
    }
  
  def embed(
    self,
    texts: list[str],
    start_id: int = 0
  ) -> list[dict[str, str | list[float] | dict[str, str]]]:
    '''
    Embed a list of texts using a Pinecone inference model.
    
    :param list[str] texts: List of texts to embed.
    :param int = 0 start_id: Starting ID for the embeddings.

    :returns embeddings (list[dict]): List of dicts to upsert in Pinecone index.
    '''

    # API call to Pinecone
    response = self.pc.inference.embed(
      'multilingual-e5-large',
      inputs=texts,
      parameters={'input_type': 'passage'}
    )

    # Extract embeddings from response
    embeddings = [
      {
        'id': str(start_id + i),
        'values': embedding.values,
        'metadata': {'text': text}
      } for i, (text, embedding) in enumerate(zip(texts, response))
    ]

    return embeddings
  
  def insert(
    self,
    texts: list[str],
    namespace: str = ''
  ) -> None:
    '''
    Insert a list of texts into the Pinecone index.
    
    :param list[str] texts: List of texts to embed.
    :param str = '' namespace: Namespace to insert the vectors.
    '''

    # Get start id
    start_id = self.num_vectors.get(namespace, 0)

    # Embed the texts
    embeddings = self.embed(texts, start_id)

    # Insert embeddings into Pinecone index
    self.index.upsert(embeddings, namespace)

    # Update number of vectors in the namespace
    self.num_vectors[namespace] = start_id + len(embeddings)

  def query(
    self,
    text: str,
    n: int = 3,
    namespace: str = ''
  ) -> list[str]:
    '''
    Query the Pinecone index and get the top matches.

    :param str text: Query text.
    :param int = 3 n: Number of matches to return.
    :param str = '' namespace: Namespace to query.
  
    :returns list[str]: List of matching texts.
    '''

    # Embed the text
    embedding = self.embed([text])[0]['values']

    # Query the Pinecone index
    response = self.index.query(
      vector=embedding,
      top_k=n,
      namespace=namespace,
      include_metadata=True
    )

    # Get matches
    matches = [match.metadata['text'] for match in response.matches]
    return matches
  
  def truncate(
    self,
    namespace: str = ''
  ) -> None:
    '''
    Delete all vectors from the Pinecone index.

    :param str = '' namespace: Namespace to delete vectors from.
    '''
    self.index.delete(delete_all=True, namespace=namespace)
    self.num_vectors[namespace] = 0
