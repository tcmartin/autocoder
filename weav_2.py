import weaviate
import json
import os
import dotenv

dotenv.load_dotenv()

client = weaviate.Client(
        url = "https://localhost:8080",  # Replace with your endpoint
    additional_headers = {
        "X-Palm-Api-Key": os.getenv("PALM_APIKEY")  # Replace with your inference API key
    }
)
