import google.generativeai as palm
import os
from dotenv import load_dotenv
import sys

load_dotenv()
sys_message="""
Respond in a json format array. You can’t respond directly to the user, but this system will manage information for you. Additionally, you should be mindful that you only have 7000 tokens.The only way to respond to the user is using the message_user function in the array. 
functions available: save_memory- description- writes to short term memory
params: memory - type=string
pop_memory- description removes from memory
params: memory - type=string
write_memory- writes to long term memory (vector database)
params: memory - type=string
message_user-sends a message to the user
params: message - type=string
heart_beat- prompts you again without waiting for user input
Format: {function:params}
Example: {save_memory: {memory: "This is interesting"}}

message from the user: 
Hi! I’m intereste in math and computer science
"""

palm.configure(api_key=os.environ['API_KEY'])
response = palm.chat(context="",messages=[sys_message])
print(response.last) #  'Hello! What can I help you with?'
#response.reply("Can you tell me a joke?")

