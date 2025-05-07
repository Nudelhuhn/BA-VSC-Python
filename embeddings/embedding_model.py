from transformers import AutoTokenizer, AutoModel
import torch

class EmbeddingModel:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)  # load tokenizer
        self.model = AutoModel.from_pretrained(model_name)  # load transformer-model

    def get_embedding(self, code_snippet):
        tokens = self.tokenizer(code_snippet, return_tensors="pt", truncation=True, padding=True)   # transform Code into tokens
        with torch.no_grad():   # deactivate training since its not needed here
            outputs = self.model(**tokens)  # **tokens unpacks dictionary input
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()  # return of last transformer-layer
