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

client = re_instantiate_weaviate()

response = (
    client.query
    .get("Question", ["question", "answer", "category"])
    .with_near_text({"concepts": ["biology"]})
    .with_limit(2)
    .do()
)
print(json.dumps(response, indent=4))

response = (
    client.query
    .get("Question", ["question", "answer", "category"])
    .with_near_text({"concepts": ["biology"]})
    .with_where({
        "path": ["category"],
        "operator": "Equal",
        "valueText": "ANIMALS"
    })
    .with_limit(2)
    .do()
)

print(json.dumps(response, indent=4))

response = (
    client.query
    .get("Question", ["question", "answer", "category"])
    .with_near_text({"concepts": ["biology"]})
    .with_generate(single_prompt="Explain {answer} as you might to a five-year-old.")
    .with_limit(2)
    .do()
)

print(json.dumps(response, indent=4))

response = (
    client.query
    .get("Question", ["question", "answer", "category"])
    .with_near_text({"concepts": ["biology"]})
    .with_generate(grouped_task="Write a tweet with emojis about these facts.")
    .with_limit(2)
    .do()
)

print(response["data"]["Get"]["Question"][0]["_additional"]["generate"]["groupedResult"])


