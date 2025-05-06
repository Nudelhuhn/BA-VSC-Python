from transformers import AutoTokenizer, AutoModel
import torch

class EmbeddingModel:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def get_embedding(self, code_snippet):
        tokens = self.tokenizer(code_snippet, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**tokens)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
