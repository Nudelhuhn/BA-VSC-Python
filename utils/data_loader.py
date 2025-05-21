import os

class DataLoader:
    def __init__(self, file_name, data_path, exclude_files=None):   # =None: optional parameter
        self.data_path = data_path
        self.file_name = file_name
        self.filenames = [] # save file names for interactive plotting
        self.exclude_files = exclude_files if exclude_files else []
        self.parent_dirs = []   # save directory names for interactive plotting

    def load_code_files(self):  # method to load the java files of the given solution set
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
    
    def get_filenames(self):
        return self.filenames

    def get_parent_dirs(self):
        return self.parent_dirs
