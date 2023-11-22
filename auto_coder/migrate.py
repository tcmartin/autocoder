import subprocess
import weaviate
import requests
import json

def refresh_token() -> str:
    result = subprocess.run(["gcloud", "auth", "print-access-token"], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error refreshing token: {result.stderr}")
        return None
    return result.stdout.strip()

def re_instantiate_weaviate() -> weaviate.Client:
    token = refresh_token()

    client = weaviate.Client(
      url = "http://localhost:8080",  # Replace with your Weaviate URL
      additional_headers = {
        "X-Palm-Api-Key": token,
      }
    )
    return client

# Run this every ~60 minutes
client = re_instantiate_weaviate()

class_obj = {
    "class": "Memory",
    "vectorizer": "text2vec-palm",  # If set to "none" you must always provide vectors yourself. Could be any other "text2vec-*" also.
    "moduleConfig": {
        "text2vec-palm": {
            "projectId": "zenithai1",
            "vectorizeClassName": True
        },
        "generative-palm": {
            "projectId": "zenithai1"
        }  # Ensure the `generative-openai` module is used for generative queries
    }
}

client.schema.create_class(class_obj)