import numpy as np
from sklearn.neighbors import NearestNeighbors
from transformers import AutoTokenizer, AutoModel
import torch
import pickle
import os

class Database:
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
    embed_model = AutoModel.from_pretrained(model_name) 
    n_features = embed_model.config.hidden_size
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
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
        
        inputs = Database.tokenizer(texts, return_tensors="pt", truncation=True, padding=True)
        
        with torch.no_grad():
            outputs = Database.embed_model(**inputs)
            
        # Average the token embeddings to get one embedding per text
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings.numpy()
        
    @staticmethod
    def metric(x, y):
        return np.dot(x, y)
    
    def __init__(self,n_neighbors: int = 2, cache_path: str = 'data.pkl'):
        '''
        Initialize the Database object with optional initial texts and pre-loaded data from a cache.

        Args:
            texts (list[str], optional): Initial list of texts to be added to the database (default is an empty list).
            n_neighbors (int, optional): Number of neighbors for the NearestNeighbors model (default is 2).
            cache_path (str, optional): Path to the cache file that stores the texts and embeddings (default is 'data.pkl').

        Returns:
            None
        '''
        # Load cached data if available
        data = {'texts': [], 'embeddings': np.empty((0, Database.n_features))}
        if os.path.exists(cache_path):
            data = pickle.load(open(cache_path, 'rb'))
        self.texts = data['texts']
        self.embeddings = data['embeddings']  
        
        # Initialize the NearestNeighbors model
        self.n_neighbors = n_neighbors
        self.nn = NearestNeighbors(n_neighbors=self.n_neighbors, metric=Database.metric)
        if(len(self.texts) > 0):
            self.nn.fit(self.embeddings)

    def insert(self, texts: list[str]):
        '''
        Insert new texts into the database and update the embeddings.

        Args:
            texts (list[str]): A list of texts to insert into the database.

        Returns:
            None
        '''
        embedding = Database.embed(texts)
        self.texts.extend(texts)
        self.embeddings = np.vstack((self.embeddings, embedding))
        if(len(self.texts) > 0):
            self.nn.fit(self.embeddings)
        pickle.dump({'texts': self.texts, 'embeddings': self.embeddings}, open('data.pkl', 'wb'))

    def query(self, q_texts: list[str]) -> np.ndarray:
        '''
        Query the database for the nearest neighbors of a list of input texts.

        Args:
            q_texts (list[str]): A list of query texts to find the nearest neighbors for.

        Returns:
            np.ndarray: Indices of the nearest neighbors for each query text, with shape (num_queries, n_neighbors).
        '''
        embeddings = Database.embed(q_texts)
        _, indices = self.nn.kneighbors(embeddings, n_neighbors=self.n_neighbors)
        return indices
    
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
    
    def clear_cache(self):
        '''
        Clear the cache file storing the texts and embeddings.

        Returns:
            None
        '''
        data = {'texts': [], 'embeddings': np.empty((0, Database.n_features))}
        pickle.dump(data, open('data.pkl', 'wb'))