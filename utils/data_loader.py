import os

class DataLoader:
    def __init__(self, file_name, data_path, exclude_files=None):   # =None: optional parameter
        self.data_path = data_path
        self.file_name = file_name
        self.filenames = [] # save file names for interactive plotting
        self.exclude_files = exclude_files if exclude_files else []
        self.parent_dirs = []   # save directory names for interactive plotting

    def load_code_files(self, concat=False):  # method to load the java files of the given solution set
        if concat:
            return self.load_and_concat_code_files()

        code_snippets = []
        for root, dirs, files in os.walk(self.data_path):   # os.walk(): searches all subdirectories recursively
            for file in files:
                if file.endswith(self.file_name) and file not in self.exclude_files:
                    file_path = os.path.join(root, file)
                    self.filenames.append(file)
                    self.parent_dirs.append(os.path.basename(root))
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        code_snippets.append(f.read())  # read the files and save them
        return code_snippets
    
    def load_and_concat_code_files(self):
        concatenated_solutions = []  # Liste für konkatenierten Code pro Lösung
        for root, dirs, files in os.walk(self.data_path):  # Durchlaufe Ordnerstruktur
            java_files = [f for f in files if f.endswith(self.file_name) and f not in self.exclude_files]  # Relevante Dateien
            if java_files:  # Wenn relevante Dateien vorhanden
                solution_code = ""  # Code-String initialisieren
                for file in java_files:  # Alle Dateien der Lösung
                    file_path = os.path.join(root, file)  # Pfad zur Datei
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        solution_code += f.read() + "\n"  # Inhalt anhängen
                concatenated_solutions.append(solution_code)  # Lösung speichern
                self.filenames.append(", ".join(java_files))  # Dateinamen speichern
                self.parent_dirs.append(os.path.basename(root))  # Überordner speichern
        return concatenated_solutions  # Rückgabe der konkatenierten Lösungen
    
    def get_filenames(self):
        return self.filenames

    def get_parent_dirs(self):
        return self.parent_dirs
