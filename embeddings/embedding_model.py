from transformers import AutoTokenizer, AutoModel   # for automatic selection and loading of a suitable tokenizer based on the name of a pre-trained model
import torch    # used for deep learning operations

class EmbeddingModel:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)  # load tokenizer which converts text to numbers for the model to use it
        self.model = AutoModel.from_pretrained(model_name)  # AutoModel identifies the a pre-trained model by model_name then load it

    def get_embedding(self, code_snippet):
        tokens = self.tokenizer(code_snippet, return_tensors="pt", truncation=True, padding=True)   # transform Code into token-IDs, return_tensors: return result as pyTorch-tensor (tensor: data container, mostly for numbers), truncation=True: cut off to long texts, padding=true: fills shorter texts to the same length
        with torch.no_grad():   # deactivate calculation of gradients (used for train the model, but not needed here)
            outputs = self.model(**tokens)  # load model with given params
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()  # return of last transformer-layer; .last_hidden_state: access to the actual token embeddings, .mean(dim=1): create vector for the whole text, .squeeze(): remove unnecessary batch dimensions, .numpy(): numpy-array for more compatibility
