
import weaviate
import subprocess
import threading
import requests
import json
from bs4 import BeautifulSoup
import tiktoken
encoding = tiktoken.get_encoding("cl100k_base")
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
import inspect
import fnmatch
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
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
import re
import google.generativeai as palm
from langchain.llms import VertexAI
#from langchain.prompts import PromptTemplate
#from langchain.chains import LLMChain
from kor import create_extraction_chain, Object, Text
from kor.nodes import Object, Text, Number
#@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
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

llm = VertexAI()



class bot:
    
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
        self.function_descriptions = {}
        self.token_threshold = token_threshold
        self.extract_function_descriptions()
        print(self.function_descriptions)
        #self.schema = [self.create_schema_object(name, desc) for name, desc in self.function_descriptions.items()]
        #function_attributes = [self.create_schema_object(name, desc) for name, desc in self.function_descriptions.items()]
        #print(function_attributes)
        
        self.schema = self.get_schema()
        self.chain = create_extraction_chain(llm, self.schema, encoder_or_encoder_class="json")
        
        if file_directory is None:
            self.file_directory = os.getcwd()
        else:
            self.file_directory = file_directory
        
    def include_in_system_message(func):
        func.include_in_system_message = True
        return func
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

    def get_bot_output(self, message):
        """
        Returns the bot output for a given user message.
        """
        # Prepare the system message
        x = self.chain.run(message)
        print(x)
        return x['data']
        #palm.configure(api_key=os.environ['PALM_APIKEY'])
        #response = palm.chat(context="",messages=[message])
        #return response.last


    """def parse_description(self, description):
        
        params_pattern = r"Parameters: (.+)"
        match = re.search(params_pattern, description)
        
        if match:
            params_text = match.group(1)
            param_pattern = r"- (\w+) \((\w+)\): ([^.]+)"
            params = re.findall(param_pattern, params_text)

            return [{
                "id": param[0],
                "type": param[1],
                "description": param[2]
            } for param in params]
        else:
            # Handle the case where the pattern is not found
            print(f"Parameters not found in description for {description}")
            return []"""
    def parse_description(self, description):
        """Parse a function description to extract parameters."""
        params_pattern = r"Parameters:\s+(.*)"
        match = re.search(params_pattern, description, re.DOTALL)

        if match:
            params_text = match.group(1)
            param_pattern = r"- (\w+) \((\w+)\): ([^.]+)"
            params = re.findall(param_pattern, params_text.strip())

            return [{
                "id": param[0],
                "type": param[1],
                "description": param[2].strip()
            } for param in params]
        else:
            # Handle the case where the pattern is not found
            print(f"Parameters not found in description for {description}")
            return []
    
    def get_schema(self):
        parameters = Object(
            id="parameters",
            description="Parameters for the function. Functions are available under 'System Functions:'.",
            attributes=[],
            examples=[
                ( "Can you delete lines 1 and 3 from example.txt? And then read in the first five lines? And then do whatever you want next?",
                [{"file_name": "example.txt", "lines_to_delete": [1, 3]},
                {"file_name": "example.txt", "start_line": 0, "num_lines": 5},
                {},]
                )
            ],
            many=False
        )
        schema = Object(
            id="bot_commands",
            description="Generic schema for any bot command",
            attributes=[
                Text(id="function_name", description="Name of the function to call. Available functions:"+self.get_function_descriptions()),
                parameters,
            ],
            examples=[("Can you delete lines 1 and 3 from example.txt? And then download https://example.com?",
                [{
                    "function_name": "delete_lines",
                    "parameters": {"file_name": "example.txt", "lines_to_delete": (1, 3)}
                },
                {
                    "function_name": "download_and_extract_text",
                    "parameters": {"url": "https://example.com"}
                }]),("Can you read in the first five lines of example.txt? And then do whatever you want next?",
                [{"function_name":"heart_beat", "parameters": {}},
                {"function_name": "read_file", "parameters": {"file_name": "example.txt", "start_line": 0, "num_lines": 5}},
                ]),("hello",[{"function_name":"message_user", "parameters": {"message": "Hi, what can I help you with?"}},])
            ],
            many=True
        )
        return schema
    def create_schema_object(self, func_name, description):
        """Create a schema object for a function based on its description."""
        params = self.parse_description(description)
        attributes = []
        for param in params:
            # Assuming all parameters are of type Text for simplicity
            # You might want to adjust this based on actual parameter types
            attributes.append(Text(id=param['id'], description=param['description']))

        return Object(
            id=func_name,
            description=description.split("Parameters:")[0].strip(),
            attributes=attributes,
            many=False  # Adjust this based on your requirements
        )
    def generate_system_message(self):
        system_message = self.base_message+f"""You are now part of an autonomous agent system that is geared primarily towards developing programs. This system will run your commands and return output to you when appropriate. You can view and edit files and even download webpages from the internet through this system. This system will also manage memory for you. Speaking of that, you have a limited context window. of {self.token_threshold} tokens that prevent you from being able to see everything at once. Please manage your memory wisely. Additionally, you should be mindful that you only have {self.token_threshold} tokens. You can respond to the user, but your primary focus should be the tasks they give you.\n"""
        
    
        system_message += "\nFile Directory Overview:\n" + self.get_directory_overview(self.file_directory)
        system_message += "Recent Messages:\n"
        for msg in self.recent_messages:
            system_message += f"- {msg}\n"
        
        # Calculate the token count
        self.token_count = num_tokens_from_string(system_message)

        # Check if token count exceeds the threshold and truncate if necessary
        if self.token_count > self.token_threshold:
            system_message = leftTruncate(system_message, self.token_threshold)

        # Add token count and remaining tokens at the top of the message
        token_info = f"Token Count: {self.token_count+600}\nTokens Remaining: {self.token_threshold - self.token_count -600}\n"
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
        # Keep only the last 10 messages (including the new one)
        self.recent_messages = self.recent_messages[-12:]
    
    @classmethod
    def get_directory_overview(cls, path):
        """
        Returns a string representation of the file directory overview, ignoring files/folders in .botignore.
        """
        def parse_botignore(botignore_path):
            """Parse .botignore and return a list of patterns."""
            patterns = []
            if os.path.exists(botignore_path):
                with open(botignore_path, 'r') as file:
                    for line in file:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            # Append a wildcard for directories to match all contents
                            if not line.endswith('/'):
                                line += '/'
                            patterns.append(line)
                            # Also add the pattern without the trailing slash for exact matches
                            patterns.append(line[:-1])
            return patterns

        def is_ignored(path, ignore_patterns, root):
            """Check if a given path matches any of the ignore patterns."""
            relative_path = os.path.relpath(path, root)
            for pattern in ignore_patterns:
                if fnmatch.fnmatch(relative_path, pattern):
                    return True
                if pattern.endswith('/') and relative_path.startswith(pattern[:-1]):
                    return True
            return False

        botignore_path = os.path.join(path, '.botignore')
        ignore_patterns = parse_botignore(botignore_path)
        print(ignore_patterns)

        overview = ""
        for root, dirs, files in os.walk(path, topdown=True):
            # Filter directories and files based on ignore patterns
            dirs[:] = [d for d in dirs if not is_ignored(os.path.join(root, d), ignore_patterns, path)]
            files = [f for f in files if not is_ignored(os.path.join(root, f), ignore_patterns, path)]

            level = root.replace(path, '').count(os.sep)
            indent = ' ' * 4 * (level)
            overview += f"{indent}{os.path.basename(root)}/\n"
            subindent = ' ' * 4 * (level + 1)
            for f in files:
                overview += f"{subindent}{f}\n"
        return overview
    @include_in_system_message
    def save_memory(self, memory):
        """
        Writes to short term memory.
        Parameters:
        - memory (str): The memory to be saved.
        """
        self.memory.append(memory)
    @include_in_system_message
    def execute_shell_command(self, command, background=False):
        """
        Executes an arbitrary shell command, with an option to run it in the background.
        Parameters:
        - command (str): The shell command to be executed.
        - background (bool): Whether to run the command in the background.
        """

        def run_command(cmd):
            try:
                # Execute the command
                process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

                # Read the output up to 1000 characters
                stdout, stderr = process.communicate()
                stdout = stdout[:1000]
                stderr = stderr[:1000]

                self.command_output = {"stdout": stdout, "stderr": stderr}
            except Exception as e:
                self.command_output = {"error": str(e)}

        if background:
            # Run the command in a separate thread if background is True
            thread = threading.Thread(target=run_command, args=(command,))
            thread.start()
            return "Command is running in the background."
        else:
            # Run the command in the foreground
            run_command(command)
            return self.command_output
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
    def write_file(self, file_name, start_line, text):
        """
        Inserts text into a file starting at a specific line. If the file doesn't
        exist, it's created. If the file doesn't have enough lines, it pads the file 
        with new lines until it reaches the specified start line.
        Parameters:
        - file_name (str): The name of the file to be written to.
        - start_line (int): The line to start inserting text from (0-indexed).
        - text (str): The text to be inserted.
        """

        try:
            # Read the file's existing content or create an empty list if the file doesn't exist
            try:
                with open(file_name, 'r') as file:
                    existing_lines = file.readlines()
            except FileNotFoundError:
                existing_lines = []

            # Ensure existing_lines has enough entries to reach the start_line
            while len(existing_lines) < start_line:
                existing_lines.append('\n')

            # Insert the new text at the specified line
            insertion_point = start_line if start_line < len(existing_lines) else -1
            if insertion_point != -1:
                # If the start_line is within the existing_lines, insert there
                existing_lines.insert(insertion_point, text + '\n')
            else:
                # If the start_line is beyond the end, just append
                existing_lines.append(text + '\n')

            # Write back the modified content
            with open(file_name, 'w') as file:
                file.writelines(existing_lines)

        except Exception as e:
            # Handle other exceptions
            return f"An error occurred: {e}"


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
        Parameters:
        - None
        """
        
        
        self.heart_beat = True
    @include_in_system_message
    def download_and_extract_text(self, url):
        """
        Downloads a webpage or a text file from a given URL, extracts its text, 
        and saves it to a file named 'downloaded_page.txt'. 
        Dynamically determines if the content is HTML.
        Parameters:
        - url (str): The URL of the webpage or text file to download.
        Returns:
        - str: The file path of the saved text content.
        """

        try:
            response = requests.get(url)
            response.raise_for_status()

            # Check if the content type is HTML
            if 'text/html' in response.headers.get('Content-Type', ''):
                # Use BeautifulSoup to parse HTML and extract text
                soup = BeautifulSoup(response.content, 'html.parser')
                text_content = soup.get_text()
            else:
                # Directly use response text for non-HTML content
                text_content = response.text

            # Save to a file
            file_path = "downloaded_page.txt"
            with open(file_path, "w", encoding="utf-8") as file:
                file.write(text_content)

            return file_path
        except requests.RequestException as e:
            return f"An error occurred: {e}"

    def interpret_result(self, data):
        try:
            # Load the JSON array into a Python object
            #data = json.loads(json_array)

            # Check if 'bot_commands' key exists and is a list
            if 'bot_commands' in data and isinstance(data['bot_commands'], list):
                commands = data['bot_commands']
            else:
                print("JSON does not contain 'bot_commands' key or it's not a list.")
                return []

            responses = []
            for command in commands:
                # Check if command contains 'function_name' and 'parameters'
                if 'function_name' in command and 'parameters' in command:
                    function_name = command['function_name']
                    params = command['parameters']

                    if hasattr(self, function_name) and callable(getattr(self, function_name)):
                        try:
                            # Call the function with the provided parameters
                            response = getattr(self, function_name)(**params)
                            responses.append(response)
                        except TypeError as e:
                            print(f"Error calling function '{function_name}': {e}")
                    else:
                        print(f"Function '{function_name}' not found in bot class.")
                else:
                    print("Command does not contain 'function_name' or 'parameters'.")
            self.prev_commands = commands
            return responses, commands
        except json.JSONDecodeError:
            print("Invalid JSON format.")

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def process_and_interpret_message(self, message):
        """
        Processes a user message to get bot output and interprets the result.
        Retries with exponential backoff for a maximum of 6 attempts.
        Raises InterpretationError if the interpretation result is empty.
        """
        # Prepare and get the system message output
        bot_output = self.get_bot_output(message)

        # Interpret the result
        interpretation, commands = self.interpret_result(bot_output)

        # Check if the interpretation is empty and raise an exception if it is
        if not interpretation:
            raise self.InterpretationError("Interpretation resulted in an empty response.")

        return interpretation, commands
