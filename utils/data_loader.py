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
    
    
    def get_scores(self):
        scores = [] # Initialize an empty list to store extracted scores
        for dir_name in self.parent_dirs:   # Iterate over each submission directory name collected earlier
            try:
                score = int(''.join(filter(str.isdigit, dir_name.split(' Punkte')[0].split()[-1])))
                # 1. Split the directory name at ' Punkte' → take the part before it
                # 2. Split that string into words → take the last word (expected to be the score)
                # 3. Keep only digits from that string
                # 4. Convert the result to an integer and assign to score
            except ValueError:
                score = -1  # If conversion fails (no number found), set score to -1 as a fallback
            scores.append(score)    # Add the extracted score to the scores list
        return scores


    def get_filenames(self):
        return self.filenames

    def get_parent_dirs(self):
        return self.parent_dirs
