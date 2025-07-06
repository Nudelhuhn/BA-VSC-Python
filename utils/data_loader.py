import os

class DataLoader:
    def __init__(self, file_name, data_path, exclude_files=None, exclude_folders=None):
        self.data_path = data_path
        self.file_name = file_name
        self.filenames = [] # save file names for interactive plotting
        self.exclude_files = exclude_files if exclude_files else []
        self.exclude_folders = exclude_folders if exclude_folders else []
        self.parent_dirs = []   # save directory names for interactive plotting


    def load_code_files(self, concat=False):
        results = []
        for root, dirs, files in os.walk(self.data_path):   # walk through all subdirectories
            if any(excl in root for excl in self.exclude_folders):   # skip excluded folders
                continue

            desired_files = [f for f in files if f.endswith(self.file_name) and f not in self.exclude_files]   # find matching files and skip excluded files
            if not desired_files:
                continue

            solution_code = ""
            for file in desired_files:
                file_path = os.path.join(root, file)    # os.path.join: combine data paths
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                    if concat:
                        solution_code += content + "\n" # concatenate file contents
                    else:
                        results.append(content) # otherwise save each content separately

                self.filenames.append(file)

            # extract folder names (parent directory and score folder)
            score_dir = os.path.basename(root)  # os.path.basename: return last part of a path
            student_id = os.path.basename(os.path.dirname(root))
            self.parent_dirs.append((student_id, score_dir))

            if concat:
                results.append(solution_code)

        return results

    
    def get_scores(self):
        scores = []
        for parent_tuple in self.parent_dirs:
            score_dir = parent_tuple[1]
            try:
                score = int(''.join(filter(str.isdigit, score_dir.split(' Punkte')[0].split()[-1]))) # Extract numeric score from folder name by splitting at ' Punkte', taking the last word, filtering digits, joining them and converting to integer
            except ValueError:
                score = -1  # fallback value if parsing fails
            scores.append(score)
        return scores
    

    def get_filenames(self):
        return self.filenames

    def get_parent_dirs(self):
        return self.parent_dirs