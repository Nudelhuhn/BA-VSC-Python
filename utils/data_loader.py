import os

class DataLoader:
    def __init__(self, data_path):
        self.data_path = data_path

    def load_code_files(self):
        code_snippets = []
        # os.walk() geht rekursiv durch alle Verzeichnisse und Unterverzeichnisse
        for root, dirs, files in os.walk(self.data_path):
            for file in files:
                if file.endswith(".java"):  # nur Java-Dateien
                    file_path = os.path.join(root, file)
                    print(f"Finde Java-Datei: {file_path}")  # Ausgabe der gefundenen Datei
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:  # errors="ignore" ignores chars which cant be coded in utf-8
                        code_snippets.append(f.read())
        return code_snippets