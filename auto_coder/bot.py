
import weaviate
import subprocess
import requests
import json
from bs4 import BeautifulSoup
import tiktoken
encoding = tiktoken.get_encoding("cl100k_base")
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
import inspect
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


import os

from dotenv import load_dotenv

load_dotenv()
headers = {"X-PaLM-Api": os.environ["PALM_APIKEY"]}

def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    num_tokens = len(encoding.encode(string))
    return num_tokens
def leftTruncate(text, length):
    encoded = encoding.encode(text)
    num = len(encoded)
    if num > length:
        return encoding.decode(encoded[num - length:])
    else:
        return text
    
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



def include_in_system_message(func):
    func.include_in_system_message = True
    return func

class bot:
    function_descriptions = {}
    def __init__(self, name="", token="", admin_id="", base_message="", file_directory=None,token_threshold=7500):
        self.context = ""
        self.recent_messages = []
        self.name = name
        self.recent_message = ""
        self.memory = []
        self.prev_commands = []
        self.base_message = base_message
        self.heart_beat = False
        self.system_notice = ""
        self.token_count = 0
        self.token_threshold = token_threshold
        self.extract_function_descriptions()
        if file_directory is None:
            self.file_directory = os.getcwd()
        else:
            self.file_directory = file_directory
        self.function_descriptions = {}

    @classmethod
    def register_function_description(cls, func_name, description):
        cls.function_descriptions[func_name] = description
    def extract_function_descriptions(self):
        for name, func in inspect.getmembers(self, predicate=inspect.ismethod):
            if getattr(func, 'include_in_system_message', False) and func.__doc__:
                self.function_descriptions[name] = func.__doc__.strip()
    def get_function_descriptions(self):
        description_text = ""
        for func, desc in self.function_descriptions.items():
            description_text += f"{func} - {desc}\n"
        return description_text
    def generate_system_message(self):
        system_message = self.base_message+"Respond in a json format array. You can’t respond directly to the user, but this system will manage information for you. Additionally, you should be mindful that you only have 7000 tokens.The only way to respond to the user is using the message_user function in the array.\n"+"System Functions:\n"
        system_message += self.get_function_descriptions()
        system_message += f"\nTokens Remaining: {self.token_threshold - self.token_count}"
        system_message += "\nFile Directory Overview:\n" + self.get_directory_overview(self.file_directory)
        system_message = "Recent Messages:\n"
        for msg in self.recent_messages:
            system_message += f"- {msg}\n"
        system_message += """
        Format: {function:params}
        Example: {save_memory: {memory: "This is interesting"}}
        """
        # Calculate the token count
        self.token_count = num_tokens_from_string(system_message)

        # Check if token count exceeds the threshold and truncate if necessary
        if self.token_count > self.token_threshold:
            system_message = leftTruncate(system_message, self.token_threshold)

        # Add token count and remaining tokens at the top of the message
        token_info = f"Token Count: {self.token_count}\nTokens Remaining: {self.token_threshold - self.token_count}\n"
        system_message = token_info + system_message

        # Add system notice if token count exceeded
        if self.token_count > self.token_threshold:
            self.system_notice = "Alert: Token count has exceeded the threshold. Message truncated. Please manage your data."
            system_message = self.system_notice + "\n" + system_message
        return system_message
    def update_recent_messages(self, message):
        """
        Updates the list of recent messages with the new message.
        """
        self.recent_messages.append(message)
        # Keep only the last 5 messages (including the new one)
        self.recent_messages = self.recent_messages[-5:]
    @classmethod
    def get_directory_overview(cls, path):
        """
        Returns a string representation of the file directory overview.
        """
        overview = ""
        for root, dirs, files in os.walk(path):
            level = root.replace(path, '').count(os.sep)
            indent = ' ' * 4 * (level)
            overview += f"{indent}{os.path.basename(root)}/\n"
            subindent = ' ' * 4 * (level + 1)
            for f in files:
                overview += f"{subindent}{f}\n"
        return overview
    def update_token_count(self, added_tokens):
        """
        Updates the token count and checks if the threshold is exceeded.
        Parameters:
        - added_tokens (int): Number of tokens to add to the count.
        """
        self.token_count += added_tokens
        if self.token_count > self.token_threshold:
            self.system_notice = "Alert: Token count has exceeded the threshold. Consider deleting some data."
    @include_in_system_message
    def save_memory(self, memory):
        """
        Writes to short term memory.
        Parameters:
        - memory (str): The memory to be saved.
        """
        self.memory.append(memory)
        
    @include_in_system_message
    def delete_lines(self, file_name, lines_to_delete):
        """
        Deletes specific lines from a file.
        Parameters:
        - file_name (str): The name of the file to be modified.
        - lines_to_delete (list of int): The line numbers to be deleted (0-indexed).
        """
        

        try:
            with open(file_name, 'r') as file:
                lines = file.readlines()

            lines = [line for idx, line in enumerate(lines) if idx not in lines_to_delete]

            with open(file_name, 'w') as file:
                file.writelines(lines)
        except FileNotFoundError:
            return "File not found."
        except Exception as e:
            return f"An error occurred: {e}"
    @include_in_system_message
    def read_file(self, file_name, start_line, num_lines):
        """
        Reads a file and returns a specified range of lines along with their start and end line numbers.
        Parameters:
        - file_name (str): The name of the file to be read.
        - start_line (int): The line to start reading from (0-indexed).
        - num_lines (int): The number of lines to read.
        Returns:
        - dict: A dictionary containing the start line, end line, and the lines read.
        """
        
        try:
            with open(file_name, "r") as f:
                lines = f.readlines()

            # Calculate the end line index
            end_line = start_line + num_lines

            # Extract the requested lines
            requested_lines = lines[start_line:end_line]

            # Return the result along with the start and end line numbers
            return {
                "start_line": start_line,
                "end_line": end_line - 1,  # Adjusting because the range is exclusive at the end
                "lines": requested_lines
            }
        except FileNotFoundError:
            return {"error": "File not found."}
        except Exception as e:
            return {"error": f"An error occurred: {e}"}

    @include_in_system_message
    def write_file(self, file_name, start_line, num_lines, lines):
        """
        Writes to a file starting at a specific line.
        Parameters:
        - file_name (str): The name of the file to be written to.
        - start_line (int): The line to start writing from (0-indexed).
        - num_lines (int): The number of lines to write.
        - lines (list): The lines to be written.
        """
        

        # Read existing file content
        try:
            with open(file_name, 'r') as file:
                existing_lines = file.readlines()
        except FileNotFoundError:
            existing_lines = []

        # Ensure existing_lines has enough entries to modify
        while len(existing_lines) < start_line:
            existing_lines.append('\n')

        # Replace specified lines
        for i in range(num_lines):
            if start_line + i < len(existing_lines):
                existing_lines[start_line + i] = lines[i] + '\n'
            else:
                existing_lines.append(lines[i] + '\n')

        # Write modified content back to file
        with open(file_name, 'w') as file:
            file.writelines(existing_lines)
    @include_in_system_message
    def replace_lines(self, file_name, start_line, end_line, new_lines):
        """
        Replaces lines in a file with new content.
        Parameters:
        - file_name (str): The name of the file to be modified.
        - start_line (int): The first line to be replaced (0-indexed).
        - end_line (int): The last line to be replaced (0-indexed).
        - new_lines (list): The new lines to be inserted.
        """
        

        # Read the existing file content
        try:
            with open(file_name, 'r') as file:
                existing_lines = file.readlines()
        except FileNotFoundError:
            existing_lines = []

        # Modify the lines in the specified range
        # Adjust the range to fit within the existing file size
        start_line = max(0, start_line)
        end_line = min(end_line, len(existing_lines) - 1)

        # Replace the specified range with new lines
        if start_line <= len(existing_lines):
            modified_content = existing_lines[:start_line] + new_lines + existing_lines[end_line + 1:]
        else:
            # If start_line is beyond the end of the file, append new lines
            modified_content = existing_lines + new_lines

        # Write the modified content back to the file
        with open(file_name, 'w') as file:
            file.writelines(modified_content)
    @include_in_system_message
    def search_in_file(self, file_name, search_term):
        """
        Searches for a term in a file and returns lines containing that term along with their line numbers.
        Parameters:
        - file_name (str): The name of the file to be searched.
        - search_term (str): The term to search for in the file.
        """
        

        # Initialize a list to hold the results
        results = []

        try:
            with open(file_name, 'r') as file:
                for line_number, line in enumerate(file, 1):  # Start enumeration at 1 for human-readable line numbers
                    if search_term in line:
                        results.append(f"{line_number}: {line}")
        except FileNotFoundError:
            results.append("File not found.")

        return results
        
    @include_in_system_message
    def pop_memory(self, memory):
        """
        Removes from memory.

        Parameters:
        - memory (int): Index of the memory to be removed.
        """
        
        try:
            self.memory.pop(memory)
        except IndexError:
            return "Memory not found."
        
    @include_in_system_message
    def write_memory(self, memory):
        """
        Writes to long term memory (vector database).

        Parameters:
        - memory (str): The memory to be written.
        """

        
       
        client.batch.configure(batch_size=100)  # Configure batch
        with client.batch as batch:  # Initialize a batch process
            properties = {
                "memory": memory
            }
            batch.add_data_object(
                data_object=properties,
                class_name="Memory"
            )
    @include_in_system_message
    def search_memory(self, memory):
        """
        Searches long term memory (vector database).

        Parameters:
        - memory (str): The memory to be searched for.
        """
        
        
        response = (
            client.query
            .get("Memory", ["memory"])
            .with_near_text({"memory": [memory]})
            .with_limit(2)
            .do()
        )
        print(json.dumps(response, indent=4))
        memories = []
        for i in response["data"]["Get"]["Memory"]:
            memories.append(i["memory"])
        return memories
    @include_in_system_message
    def message_user(self, message):
        """
        Sends a message to the user.

        Parameters:
        - message (str): The message to be sent to the user.
        """
        
        
        print(message)
        self.update_recent_messages("Assistant (You): "+message)
        
    @include_in_system_message
    def heart_beat(self):
        """
        Prompts you again without waiting for user input.
        """
        
        
        self.heart_beat = True
    @include_in_system_message
    def download_and_extract_text(self, url, is_html=False):
        """
        Downloads a webpage or a text file from a given URL and extracts its text.
        Parameters:
        - url (str): The URL of the webpage or text file to download.
        - is_html (bool): Flag to indicate if the URL is an HTML page (default: False).
        Returns:
        - str: The text content of the downloaded webpage or file.
        """
        

        try:
            response = requests.get(url)
            response.raise_for_status()

            if is_html:
                # Use BeautifulSoup to parse HTML and extract text
                soup = BeautifulSoup(response.content, 'html.parser')
                text_content = soup.get_text()
            else:
                # Directly use response text for plain text files
                text_content = response.text

            return text_content
        except requests.RequestException as e:
            return f"An error occurred: {e}"

    def interpret_json(self, json_array):
        try:
            commands = json.loads(json_array)
            responses = []
            for command in commands:
                for function, params in command.items():
                    if hasattr(self, function) and callable(getattr(self, function)):
                        try:
                            response = getattr(self, function)(**params)
                            responses.append(response)
                        except TypeError as e:
                            print(f"Error calling function '{function}': {e}")
                    else:
                        print(f"Function '{function}' not found in bot class.")
            return responses
        except json.JSONDecodeError:
            print("Invalid JSON array format.")


