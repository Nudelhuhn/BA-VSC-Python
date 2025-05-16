import os

class DataLoader:
    def __init__(self, file_name, data_path, exclude_files=None):   # =None: optional parameter; (prevent usage of mutable default params e.g. exclude_files=[] -> every instance of the class uses the same list)
        self.data_path = data_path
        self.file_name = file_name
        self.filenames = []  # Hier speichern wir die Dateinamen
        self.exclude_files = exclude_files if exclude_files else []
        self.parent_dirs = []  # Hier speichern wir die übergeordneten Ordner

    def load_code_files(self):
        code_snippets = []
        for root, dirs, files in os.walk(self.data_path):   # os.walk() durchsucht rekursiv alle Unterverzeichnisse
            for file in files:
                if file.endswith(self.file_name) and file not in self.exclude_files:
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
