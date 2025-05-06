import os

class DataLoader:
    def __init__(self, data_path):
        self.data_path = data_path

    def load_code_files(self):
        code_snippets = []
        for file in os.listdir(self.data_path):
            if file.endswith(".java"):
                with open(os.path.join(self.data_path, file), "r", encoding="utf-8") as f:
                    code_snippets.append(f.read())
        return code_snippets
