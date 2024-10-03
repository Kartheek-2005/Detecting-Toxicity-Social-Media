from pinecone import Pinecone
from pinecone.data.index import Index


def get_pc_index(
  api_key: str,
  index_name: str
) -> tuple[Pinecone, Index]:
  """
  Get Pinecone client and connect to an index.
  
  ## Parameters
    `api_key`: Pinecone API key.
    `index_name`: Name of the index to connect to.
  
  ## Returns
    `(pc, index)`: Pinecone client and index.
  """

  # Get Pinecone client
  pc = Pinecone(api_key)

  # Connect to index
  index = pc.Index(index_name)

  return pc, index

def embed(
  texts: list[str],
  pc: Pinecone,
  id: int = 0
) -> list[dict[str, any]]:
  """
  Embed a list of texts using a Pinecone inference model.
  
  ## Parameters
    `texts`: List of texts to embed.
    `pc`: Pinecone client.
    `id`: Starting ID for the embeddings.

  ## Returns
    `embeddings`: List of dicts to upsert in Pinecone index.
  """

  # API call to Pinecone
  response = pc.inference.embed(
    "multilingual-e5-large",
    inputs = texts,
    parameters = {
      "input_type": "passage"
    }
  )

  # Extract embeddings from response
  embeddings = [{
    "id": str(id + i),
    "values": embedding.values,
    "metadata": {"text": text}
  } for i, (text, embedding) in enumerate(zip(texts, response))
  ]

  return embeddings

def query(
  text: str,
  index: Index,
  n: int = 3
) -> list[dict]:
  """
  Send a query to a Pinecone index and get the top matches.

  ## Parameters
    `text`: Query text.
    `index`: Pinecone index.
    `n`: Number of matches to return.
  
  ## Returns
    `matches`: List of matching texts.
  """

  # Embed the text
  embedding = embed([text])[0]["values"]

  # Query the Pinecone index
  response = index.query(
    vector = embedding,
    top_k = n,
    include_values = False,
    include_metadata = True
  )

  # Get matches
  matches = [match.metadata["text"] for match in response.matches]

  return matches
