import json

def config_reader(json_str):
    config = json.loads(json_str)
    return config