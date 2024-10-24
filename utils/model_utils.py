from collections.abc import Iterable
from math import ceil

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from pinecone_utils import PineconeInterface



class Prompter:

  def __init__(
    self,
    tokenizer: AutoTokenizer,
    pc: PineconeInterface | None = None,
    toxic_namespace: str = "",
    benign_namespace: str = "",
    prompt_template: str = "{text}"
  ) -> None:
    
    self.tokenizer = tokenizer
    self.pc = pc
    self.toxic_namespace = toxic_namespace
    self.benign_namespace = benign_namespace
    self.prompt_template = prompt_template

  def create_prompt(
    self,
    text: str,
    num_samples: int = 3
  ) -> str:
    
    # Check if num_samples is valid
    if num_samples <= 0:
      return text
    
    # Check if Pinecone interface is given
    assert self.pc, "Pinecone not connected"
    
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
    ).strip()

    return prompt



class Pipeline(Prompter):

  def __init__(
    self,
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    pc: PineconeInterface | None = "",
    toxic_namespace: str = "",
    benign_namespace: str = "",
    prompt_template: str = "{text}"
  ) -> None:
    
    super().__init__(
      tokenizer,
      pc,
      toxic_namespace,
      benign_namespace,
      prompt_template
    )
    self.model = model
  
  def __call__(
    self,
    text: str,
    add_to_pinecone: bool = False,
    num_samples: int = 3
  ) -> float:
    
    # Create prompt
    prompt = self.create_prompt(text, num_samples)

    # Tokenize prompt
    inputs = self.tokenizer(prompt, return_tensors="pt")

    # Get model output
    outputs = self.model(**inputs)

    # Get probability of toxic class
    probs = F.softmax(outputs.logits, dim=-1)
    toxic_prob = probs[0, 1].item()

    # Add to Pinecone if mentioned
    if add_to_pinecone:
      self.pc.upsert(
        text=text,
        namespace=self.toxic_namespace if toxic_prob > 0.5
        else self.benign_namespace
      )

    return toxic_prob
  


class Trainer(Prompter):

  def __init__(
    self,
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    optimizer: Optimizer,
    scheduler: ReduceLROnPlateau,
    pc: PineconeInterface | None = None,
    toxic_namespace: str = "",
    benign_namespace: str = "",
    prompt_template: str = "{text}"
  ) -> None:
    
    super().__init__(
      tokenizer,
      pc,
      toxic_namespace,
      benign_namespace,
      prompt_template
    )
    self.model = model
    self.optimizer = optimizer
    self.scheduler = scheduler
  
  def train(
    self,
    texts: Iterable[str],
    labels: Iterable[int],
    batch_size: int,
    epochs: int,
    num_samples: int = 0
  ) -> None:
    
    # Create prompts
    prompts = [
      self.create_prompt(text, num_samples)
      for text in texts
    ]

    # Tokenize prompts
    tokenized = self.tokenizer(prompts)["input_ids"]
    tokenized = np.array(tokenized, dtype=object)

    # Dynamically batch texts
    ind = np.argsort([len(ids) for ids in tokenized])
    batches_ind = (
      ind[i : i+batch_size]
      for i in range(0, len(ind), batch_size)
    )
    num_batches = ceil(len(texts) / batch_size)
    labels = torch.tensor(labels, dtype=torch.float32)

    # Train model
    self.model.train()
    for _ in range(epochs):

      # Track total epoch loss
      epoch_loss = 0

      for ind in batches_ind:

        # Get batch
        batch = tokenized[ind]
        inputs = self.tokenizer.pad(batch, return_tensors="pt")
        inputs["labels"] = labels[ind]

        # Get loss
        loss = self.model(**inputs).loss
        epoch_loss += loss.item()

        # Backpropagate
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

      epoch_loss /= num_batches

      # Update learning rate
      self.scheduler.step(epoch_loss)
