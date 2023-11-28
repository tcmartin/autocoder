import typer
from .bot import bot
#from .llms import get_response
import json
import os
from dotenv import load_dotenv

load_dotenv()

app = typer.Typer()


DEBUG = True
if(os.getenv("OPENAI_API_KEY") is not None):
    use_openai = True
    print("Using OpenAI")
@app.command()
def run():
    print("Running")
    my_bot = bot()

    user_input = input("Enter your command: ")
    my_bot.update_recent_messages("User: "+user_input)
    if use_openai:
        while True:
            # Prepare the system message
            system_message = my_bot.make_system_message_openai()+'\n'+user_input
            if DEBUG:
                print(system_message)
            response = my_bot.get_assistant_response(system_message)
            print(response)
            if not my_bot.heart_beated:
                user_input = input("Enter your command: ")
            
    while True:
        # Prepare the system message
        system_message = my_bot.generate_system_message()
        if DEBUG:
            print(system_message)
        
        # Get response from llms.get_response
        #json_commands = json.loads(bot_response)
        #if DEBUG:
            #print(json_commands)
        
        # Execute the commands and store responses
        responses, commands = my_bot.process_and_interpret_message(system_message)
        for response in responses:
            print(response)
            my_bot.update_recent_messages(response)  # Function to update recent messages
        for command in commands:
            my_bot.update_recent_messages(command)
        # Check for heartbeat
        if not my_bot.heart_beated:
            # Wait for user input
            user_input = input("Enter your command: ")
            my_bot.update_recent_messages("User: "+user_input)  # Update recent messages with user input

            # Exit condition
            if user_input.lower() == 'exit':
                print("Exiting the application.")
                break

        # Reset heartbeat
        my_bot.heart_beated = False
