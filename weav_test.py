import weaviate
import weaviate.classes as wvc
import os
import dotenv
import requests
import json

dotenv.load_dotenv()
headers = {"X-Palm-Api-Key": os.environ["PALM_APIKEY"]}
client = weaviate.connect_to_local(port=8080, grpc_port=50051, headers=headers)

questions = client.collections.create(
    name="Question_",
    vectorizer_config=wvc.Configure.Vectorizer.text2vec_palm(),
    generative_config=wvc.Configure.Generative.palm()  
)

resp = requests.get('https://raw.githubusercontent.com/weaviate-tutorials/quickstart/main/data/jeopardy_tiny.json')
data = json.loads(resp.text)  # Load data

question_objs = list()
for i, d in enumerate(data):
    question_objs.append({
        "answer": d["Answer"],
        "question": d["Question"],
        "category": d["Category"],
    })

questions = client.collections.get("Question_")
questions.data.insert_many(question_objs)


response = questions.query.near_text(
    query="biology",
    limit=2
)

print(response.objects[0].properties)