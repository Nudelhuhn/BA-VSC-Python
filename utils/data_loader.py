import os

class DataLoader:
    def __init__(self, file_name, data_path, exclude_files=None, exclude_folders=None):   # =None: optional parameter
        self.data_path = data_path
        self.file_name = file_name
        self.filenames = [] # save file names for interactive plotting
        self.exclude_files = exclude_files if exclude_files else []
        self.exclude_folders = exclude_folders if exclude_folders else []
        self.parent_dirs = []   # save directory names for interactive plotting


    def load_code_files(self, concat=False):  # method to load the java files of the given solution set
        if concat:
            return self.load_and_concat_code_files()

        code_snippets = []
        for root, dirs, files in os.walk(self.data_path):   # os.walk(): searches all subdirectories recursively
            if any(excl in root for excl in self.exclude_folders):
                continue  # skip folder if it matches exclusion
            for file in files:
                if file.endswith(self.file_name) and file not in self.exclude_files:
                    file_path = os.path.join(root, file)
                    self.filenames.append(file)
                    # Übergeordneten Ordner (Studenten-ID) holen:
                    punktzahl_ordner = os.path.basename(root)
                    student_id = os.path.basename(os.path.dirname(root))  # Elternordner von root

                    self.parent_dirs.append((student_id, punktzahl_ordner))
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        code_snippets.append(f.read())  # read the files and save them
        return code_snippets
    

    def load_and_concat_code_files(self):
        concatenated_solutions = []  # Liste für konkatenierten Code pro Lösung
        for root, dirs, files in os.walk(self.data_path):  # Durchlaufe Ordnerstruktur
            if any(excl in root for excl in self.exclude_folders):
                continue  # skip folder if it matches exclusion
            java_files = [f for f in files if f.endswith(self.file_name) and f not in self.exclude_files]  # Relevante Dateien
            if java_files:  # Wenn relevante Dateien vorhanden
                solution_code = ""  # Code-String initialisieren
                for file in java_files:  # Alle Dateien der Lösung
                    file_path = os.path.join(root, file)  # Pfad zur Datei
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        solution_code += f.read() + "\n"  # Inhalt anhängen
                concatenated_solutions.append(solution_code)  # Lösung speichern
                self.filenames.append(", ".join(java_files))  # Dateinamen speichern
                
                score_dir = os.path.basename(root)
                student_id = os.path.basename(os.path.dirname(root))

                self.parent_dirs.append((student_id, score_dir))
        return concatenated_solutions  # Rückgabe der konkatenierten Lösungen
    
    
    def get_scores(self):
        scores = []
        for parent_tuple in self.parent_dirs:
            punktzahl_ordner = parent_tuple[1]  # Zweites Element im Tupel
            try:
                score = int(''.join(filter(str.isdigit, punktzahl_ordner.split(' Punkte')[0].split()[-1])))
            except ValueError:
                score = -1
            scores.append(score)
        return scores


    def get_filenames(self):
        return self.filenames

    def get_parent_dirs(self):
        return self.parent_dirs