from itertools import batched
import subprocess

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from .pinecone_utils import PineconeInterface



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
    texts: list[str],
    labels: list[int],
    batch_size: int,
    epochs: int,
    num_samples: int = 0,
    device: str = "cpu"
  ) -> None:
    
    # Create prompts
    prompts = [self.create_prompt(text, num_samples) for text in texts]

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
    self.model.eval()
