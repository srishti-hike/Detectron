import yaml

with open("./api/apiutils/config/config.yaml", 'r') as ymlfile:
    conf = yaml.load(ymlfile)

def get_config():
    return conf