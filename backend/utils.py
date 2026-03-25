import os

def save_file(file):
    path = os.path.join("uploads", file.filename)
    file.save(path)
    return path