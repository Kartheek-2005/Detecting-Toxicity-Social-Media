from time import sleep
from itertools import batched
import subprocess

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from .database_utils import DatabaseInterface



def gpu_usage() -> list[int]:
	"""
	Get the current GPU memory usage.
	"""

	# Get output from nvidia-smi
	result = subprocess.check_output([
		"nvidia-smi",
		"--query-gpu=memory.used",
		"--format=csv,nounits,noheader"
	]).decode("utf-8").strip()

	# Extract memory used by GPUs in MiB
	gpu_memory = [int(mem) for mem in result.split("\n")]

	return gpu_memory


def get_device(threshold: int | float = 500) -> str:
	"""
	Returns a device with memory usage below `threshold`.
	"""

  # Check if CUDA is available
	if torch.cuda.is_available():
		usage = gpu_usage()
		cuda_ind = np.argmin(usage)
		return f"cuda:{cuda_ind}" if usage[cuda_ind] < threshold else "cpu"

  # Check if MPS is available
	if torch.backends.mps.is_available():
		usage = torch.mps.driver_allocated_memory() / 1e6
		return "mps" if usage < threshold else "cpu"

	return "cpu"



class Prompter:

  def __init__(
    self,
    toxic_db: DatabaseInterface,
    benign_db: DatabaseInterface,
    prompt_template: str = "{text}"
  ) -> None:

    self.toxic_db = toxic_db
    self.benign_db = benign_db
    self.prompt_template = prompt_template

  def create_prompts(
    self,
    texts: list[str]
  ) -> list[str]:
    
    # Get toxic and benign examples from Pinecone
    toxic_examples = self.toxic_db.query(texts)
    benign_examples = self.benign_db.query(texts)

    # Create prompts
    prompts = []
    for text, toxic, benign in zip(texts, toxic_examples, benign_examples):

      # Format text and examples
      text = f"\n\nText:\n{text}"
      toxic = f"\n\nToxic Examples:\n{"\n".join(toxic)}" \
        if toxic else ""
      benign = f"\n\nBenign Examples:\n{"\n".join(benign)}" \
        if benign else ""

      # Create prompt
      prompt = self.prompt_template.format(
        text=text,
        toxic=toxic,
        benign=benign
      ).strip()
    
      prompts.append(prompt)

    return prompts



class Pipeline(Prompter):

  def __init__(
    self,
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    toxic_db: DatabaseInterface,
    benign_db: DatabaseInterface,
    prompt_template: str = "{text}",
    device: str = "cpu"
  ) -> None:
    
    super().__init__(toxic_db, benign_db, prompt_template)
    self.model = model.to(device)
    self.tokenizer = tokenizer
    self.device = device
  
  def __call__(
    self,
    texts: list[str],
    add_threshold: float = 0
  ) -> float:
    
    # Create prompts
    prompts = self.create_prompts(texts)

    # Tokenize prompts
    inputs = self.tokenizer(
      prompts,
      return_tensors="pt",
      padding=True,
      truncation=True
    ).to(self.device)

    # Get model output
    with torch.no_grad():
      outputs = self.model(**inputs)

    # Get probability of toxic class
    probs = F.softmax(outputs.logits, dim=-1)
    toxic_probs = probs[:, 1].tolist()

    # Add to database if mentioned
    if 0 < add_threshold < 1:

      # Split toxic and benign texts
      toxic_texts = [text for text, prob in zip(texts, toxic_probs) if prob > add_threshold]
      benign_texts = [text for text, prob in zip(texts, toxic_probs) if prob <= add_threshold]

      # Insert into the databases
      self.toxic_db.insert(toxic_texts)
      self.benign_db.insert(benign_texts)

    return toxic_probs
  


class Trainer(Prompter):

  def __init__(
    self,
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    optimizer: Optimizer,
    scheduler: ReduceLROnPlateau,
    toxic_db: DatabaseInterface,
    benign_db: DatabaseInterface,
    prompt_template: str = "{text}"
  ) -> None:
    
    super().__init__(
      toxic_db,
      benign_db,
      prompt_template
    )

    self.model = model
    self.tokenizer = tokenizer
    self.optimizer = optimizer
    self.scheduler = scheduler

  def train(
    self,
    texts: list[str],
    labels: list[int],
    batch_size: int,
    epochs: int,
    device: str = "cpu"
  ) -> None:
    
    # Create prompts
    # prompts = [self.create_prompt(text, num_samples) for text in texts]
    prompts = self.create_prompts(texts)

    # Tokenize prompts
    tokenized = self.tokenizer(prompts)["input_ids"]
    tokenized = np.array(tokenized, dtype=object)

    # Dynamically batch texts
    inds = np.argsort([len(ids) for ids in tokenized])
    labels = torch.tensor(labels, dtype=torch.long)
    batches = []
    for ind in batched(inds, batch_size):

      # Convert ind to list to sample from numpy array
      ind = list(ind)

      # Get input_ids and labels
      ids = tokenized[ind].tolist()
      batch_labels = labels[ind]

      # Pad input_ids
      inputs = self.tokenizer.pad({"input_ids": ids}, return_tensors="pt")
      inputs["labels"] = batch_labels

      # Add to batches
      batches.append(inputs)

    num_batches = len(batches)

    # Train model
    self.model.train()
    self.model.to(device)
    for epoch in range(epochs):

      # Track total epoch loss
      epoch_loss = 0

      for ind, inputs in enumerate(batches):

        # Move inputs to device
        inputs = inputs.to(device)

        # Get loss
        loss = self.model(**inputs).loss
        epoch_loss += loss.item()

        # Backpropagate
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        print(
          f"Epoch [{epoch + 1}/{epochs}] Batch [{ind + 1}/{num_batches}] Loss [{loss.item(): .4f}]",
          end="\r"
        )

      epoch_loss /= num_batches

      print(f"Epoch [{epoch + 1}/{epochs}] Loss [{epoch_loss: .4f}]")

      # Update learning rate
      self.scheduler.step(epoch_loss)
    
    # Move model back to CPU
    self.model.to("cpu")
