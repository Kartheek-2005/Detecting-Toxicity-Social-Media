from pinecone import Pinecone


class Pinecone_DB:
  """
  Class to interact with Pinecone API.
  
  ## Parameters
    `api_key`: Pinecone API key.
    `index_name`: Name of the index to connect to.
  """

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

    self.num_vectors = self.index.describe_index_stats().total_vector_count
  
  def embed(
    self,
    texts: list[str]
  ) -> list[dict[str, any]]:
    """
    Embed a list of texts using a Pinecone inference model.
    
    ## Parameters
      `texts`: List of texts to embed.

    ## Returns
      `embeddings`: List of dicts to upsert in Pinecone index.
    """

    # API call to Pinecone
    response = self.pc.inference.embed(
      "multilingual-e5-large",
      inputs = texts,
      parameters = {
        "input_type": "passage"
      }
    )

    # Extract embeddings from response
    embeddings = [{
      "id": str(self.num_vectors + i),
      "values": embedding.values,
      "metadata": {"text": text}
    } for i, (text, embedding) in enumerate(zip(texts, response))
    ]

    return embeddings
  
  def upsert(
    self,
    texts: list[str],
    namespace: str = ""
  ) -> None:
    """
    Insert a list of texts into the Pinecone index.
    
    ## Parameters
      `texts`: List of texts to embed.
    """

    # Embed the texts
    embeddings = self.embed(texts)

    # Insert embeddings into Pinecone index
    self.index.upsert(embeddings, namespace)

    # Update id
    self.num_vectors += len(embeddings)

  def query(
    self,
    text: str,
    n: int = 3,
    namespace: str = ""
  ) -> list[dict]:
    """
    Query the Pinecone index and get the top matches.

    ## Parameters
      `text`: Query text.
      `n`: Number of matches to return.
    
    ## Returns
      `matches`: List of matching texts.
    """

    # Embed the text
    embedding = self.embed([text])[0]["values"]

    # Query the Pinecone index
    response = self.index.query(
      vector = embedding,
      top_k = n,
      namespace = namespace,
      include_values = False,
      include_metadata = True
    )

    # Get matches
    matches = [match.metadata["text"] for match in response.matches]

    return matches
  
  def delete_vectors(
    self,
    namespace: str = ""
  ) -> None:
    """
    Delete all vectors from the Pinecone index.
    """

    self.index.delete(delete_all=True, namespace=namespace)
    self.num_vectors = 0
