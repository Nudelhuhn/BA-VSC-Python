import os

class DataLoader:
    def __init__(self, file_name, data_path):
        self.data_path = data_path
        self.file_name = file_name
        self.filenames = []  # Hier speichern wir die Dateinamen
        self.parent_dirs = []  # Hier speichern wir die übergeordneten Ordner

    def load_code_files(self):
        code_snippets = []
        for root, dirs, files in os.walk(self.data_path):   # os.walk() durchsucht rekursiv alle Unterverzeichnisse
            for file in files:
                if file.endswith(self.file_name):  # Nur Java-Dateien
                    file_path = os.path.join(root, file)
                    self.filenames.append(file)  # Dateinamen speichern
                    self.parent_dirs.append(os.path.basename(root))  # Übergeordneten Ordnernamen speichern
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        code_snippets.append(f.read())
        return code_snippets
    
    def get_filenames(self):
        return self.filenames  # Gibt die gespeicherten Dateinamen zurück

    def get_parent_dirs(self):
        return self.parent_dirs  # Gibt die gespeicherten übergeordneten Ordnernamen zurück
