import typer
from .bot import bot
from .llms import get_response
import json

app = typer.Typer()



@app.command()
def run():
    print("Running")
    my_bot = bot()

    user_input = input("Enter your command: ")
    my_bot.update_recent_messages("User: "+user_input)
    while True:
        # Prepare the system message
        system_message = my_bot.generate_system_message()
        
        # Get response from llms.get_response
        json_commands = json.loads(get_response(system_message))
        
        # Execute the commands and store responses
        responses = my_bot.interpret_json(json_commands)
        for response in responses:
            print(response)
            my_bot.update_recent_messages(response)  # Function to update recent messages

        # Check for heartbeat
        if not my_bot.heart_beat:
            # Wait for user input
            user_input = input("Enter your command: ")
            my_bot.update_recent_messages("User: "+user_input)  # Update recent messages with user input

            # Exit condition
            if user_input.lower() == 'exit':
                print("Exiting the application.")
                break

        # Reset heartbeat
        my_bot.heart_beat = False