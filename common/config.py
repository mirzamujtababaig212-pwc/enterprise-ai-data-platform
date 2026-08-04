import yaml
from dotenv import load_dotenv

load_dotenv()


def load_config():
    with open("config/config.yaml") as file:
        return yaml.safe_load(file)
