from pinecone import Pinecone


class PineconeInterface:
  """
  Class to interact with Pinecone API.

  :param str api_key: Pinecone API key.
  :param str index_name: Name of the index to connect to.
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
    """
    Embed a list of texts using a Pinecone inference model.
    
    :param list[str] texts: List of texts to embed.
    :param int = 0 start_id: Starting ID for the embeddings.

    :returns embeddings (list[dict]): List of dicts to upsert in Pinecone index.
    """

    # API call to Pinecone
    response = self.pc.inference.embed(
      "multilingual-e5-large",
      inputs=texts,
      parameters={"input_type": "passage"}
    )

    # Extract embeddings from response
    embeddings = [{
      "id": str(start_id + i),
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
    
    :param list[str] texts: List of texts to embed.
    :param str = "" namespace: Namespace to insert the vectors.
    """

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
    namespace: str = ""
  ) -> list[str]:
    """
    Query the Pinecone index and get the top matches.

    :param str text: Query text.
    :param int = 3 n: Number of matches to return.
    :param str = "" namespace: Namespace to query.
  
    :returns matches (list[str]): List of matching texts.
    """

    # Embed the text
    embedding = self.embed([text])[0]["values"]

    # Query the Pinecone index
    response = self.index.query(
      vector=embedding,
      top_k=n,
      namespace=namespace,
      include_metadata=True
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

    :param str = "" namespace: Namespace to delete vectors from.
    """

    self.index.delete(delete_all=True, namespace=namespace)
    self.num_vectors[namespace] = 0
