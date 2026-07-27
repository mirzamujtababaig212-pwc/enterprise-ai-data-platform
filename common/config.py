import yaml
from dotenv import load_dotenv

load_dotenv()


def load_config():
    with open("config/config.yaml", "r") as file:
        return yaml.safe_load(file)
