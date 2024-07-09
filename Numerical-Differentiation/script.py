import os
import sys

def read_files(directory, extensions):
    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                print(f"Start file: {file}")
                with open(os.path.join(root, file), 'r') as f:
                    content = f.read()
                print(content)
                print(f"End file: {file}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 script.py <extensions> <output_file>")
        sys.exit(1)

    extensions = sys.argv[1].split(',')
    output_file = sys.argv[2]

    with open(output_file, 'w') as out:
        sys.stdout = out
        read_files('.', extensions)
        sys.stdout = sys.__stdout__

    print(f"Output written to {output_file}")
