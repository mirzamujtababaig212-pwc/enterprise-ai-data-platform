import os

from dotenv import load_dotenv

load_dotenv()

class Environment:
	APP_ENV=os.getenv("APP_ENV","DEV")
	LOG_LEVEL=os.getenv("LOG_LEVEL","INFO")
