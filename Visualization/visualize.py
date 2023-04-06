import json
import matplotlib as plt

if __name__ == "__main__":
    file_path = "C:\\Projects\\VisualStudio\\SpikepropSharp\\SpikepropSharp\\bin\\Debug\\net7.0\\23.04.06.10.43.00-val_res.json"
    file = open(file_path)
    data = json.load(file)




    file.close()
