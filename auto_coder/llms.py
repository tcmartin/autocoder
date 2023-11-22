import google.generativeai as palm
import os
from dotenv import load_dotenv

load_dotenv()

def get_response(message: str) -> str:
    palm.configure(api_key=os.environ['PALM_APIKEY'])
    response = palm.chat(context="",messages=[message])
    return response.last