import os

class DataLoader:
    def __init__(self, file_name, data_path, exclude_files=None, exclude_folders=None):   # =None: optional parameter
        self.data_path = data_path
        self.file_name = file_name
        self.filenames = [] # save file names for interactive plotting
        self.exclude_files = exclude_files if exclude_files else []
        self.exclude_folders = exclude_folders if exclude_folders else []
        self.parent_dirs = []   # save directory names for interactive plotting


    def load_code_files(self, concat=False):
        results = []
        for root, dirs, files in os.walk(self.data_path):
            if any(excl in root for excl in self.exclude_folders):
                continue

            java_files = [f for f in files if f.endswith(self.file_name) and f not in self.exclude_files]
            if not java_files:
                continue

            solution_code = ""  # Nur relevant, falls concat=True
            for file in java_files:
                file_path = os.path.join(root, file)
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                    if concat:
                        solution_code += content + "\n"
                    else:
                        results.append(content)

                self.filenames.append(file)

            score_dir = os.path.basename(root)
            student_id = os.path.basename(os.path.dirname(root))
            self.parent_dirs.append((student_id, score_dir))

            if concat:
                results.append(solution_code)

        return results

    
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