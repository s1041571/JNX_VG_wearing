import os
import json

def save_json(aDict, filedir, filename):
    filename = filename+'.json' if '.json' not in filename else filename
    filepath = os.path.join(filedir, filename)
    with open(filepath, 'w', encoding='utf8') as f:
        json.dump(aDict, f, indent=4, ensure_ascii=False)

def load_json(filedir, filename):
    filename = filename+'.json' if '.json' not in filename else filename
    filepath = os.path.join(filedir, filename)
    with open(filepath, 'r', encoding='utf8') as f:
        return json.load(f)
