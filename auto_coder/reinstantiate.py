import subprocess
import weaviate

def refresh_token() -> str:
    result = subprocess.run(["gcloud", "auth", "print-access-token"], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error refreshing token: {result.stderr}")
        return None
    return result.stdout.strip()

def re_instantiate_weaviate() -> weaviate.Client:
    token = refresh_token()

    client = weaviate.Client(
      url = "https://some-endpoint.weaviate.network",  # Replace with your Weaviate URL
      additional_headers = {
        "X-Palm-Api-Key": token,
      }
    )
    return client

# Run this every ~60 minutes
client = re_instantiate_weaviate()