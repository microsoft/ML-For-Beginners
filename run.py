# python -m pip install opencv-python textblob
import os
import subprocess

def find_notebooks(directory):
    notebooks = []
    for root, dirs, files in os.walk(directory):
        dirs[:] = [d for d in dirs if d != "R"]  # Ignore directories named "R"
        if "solution" in root:
            for file in files:
                if file.endswith(".ipynb"):
                    notebooks.append(os.path.join(root, file))
    return notebooks


def execute_notebooks(notebooks):
    for notebook in notebooks:
        try:
            subprocess.run(["jupyter", "execute", notebook], check=True)
            print(f"Executed notebook: {notebook}")
        except subprocess.CalledProcessError as e:
            print(f"Error executing notebook {notebook}: {e}")

if __name__ == "__main__":
    start_directory = "."  # Replace with the path to the directory you want to search
    notebooks = find_notebooks(start_directory)
    execute_notebooks(notebooks)
