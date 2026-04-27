from dotenv import load_dotenv
import os

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
NASA_API_KEY = os.getenv("NASA_API_KEY")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL is not set")

if not NASA_API_KEY:
    raise ValueError("NASA_API_KEY is not set")