import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from pinecone_utils import PineconeInterface



class Pipeline:

  def __init__(
    self,
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    pc: PineconeInterface,
    toxic_namespace: str,
    benign_namespace: str,
    prompt_template: str
  ) -> None:
    
    self.model = model
    self.tokenizer = tokenizer
    self.pc = pc
    self.toxic_namespace = toxic_namespace
    self.benign_namespace = benign_namespace
    self.prompt_template = prompt_template
  
  def __call__(
    self,
    text: str,
    add_to_pinecone: bool = False
  ) -> float:
    
    # Create prompt
    prompt = self.create_prompt(text)

    # Tokenize prompt
    inputs = self.tokenizer(prompt, return_tensors="pt")

    # Get model output
    outputs = self.model(**inputs)

    # Get probability of toxic class
    probs = F.softmax(outputs.logits, dim=-1)
    toxic_prob = probs[0, 1].item()

    # Add to Pinecone if mentioned
    if add_to_pinecone:
      namespace = self.toxic_namespace if toxic_prob > 0.5 else self.benign_namespace
      self.pc.upsert(text, namespace)

    return toxic_prob
  
  def create_prompt(
    self,
    text: str,
    num_samples: int = 3
  ) -> str:
    
    # Get toxic and benign examples from Pinecone
    toxic_examples = self.pc.query(
      text=text,
      n=num_samples,
      namespace=self.toxic_namespace
    )
    benign_examples = self.pc.query(
      text=text,
      n=num_samples,
      namespace=self.benign_namespace
    )

    # Format examples and text
    text = f"\n\nText:\n{text}"
    toxic_examples = f"\n\nToxic Examples:\n{"\n".join(toxic_examples)}" \
      if toxic_examples else ""
    benign_examples = f"\n\nBenign Examples:\n{"\n".join(benign_examples)}" \
      if benign_examples else ""

    # Create prompt
    prompt = self.prompt_template.format(
      text=text,
      toxic=toxic_examples,
      benign=benign_examples
    )

    return prompt
