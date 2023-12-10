# AutoCoder

AutoCoder is a cutting-edge programming assistant powered by GPT-4/PaLM. It operates directly within the terminal, responding to natural language commands. This assistant is designed to streamline various programming tasks, making your coding experience more efficient and intuitive.

## Features

- **Search the Internet**: Quickly find coding solutions and references.
- **Download Files**: Automatically handle file downloads as needed.
- **Execute Shell Commands**: Run commands directly from the terminal.
- **File Management**: Read, write, and delete files with ease.
- **Memory Management**: Effectively manage internal memory for optimal performance.
- **Vector Database Storage**: Store information in a Weaviate vector database.
- **Codebase Representation**: Maintain a direct representation of your entire codebase for easy access and manipulation.

## Setup Instructions

1. **Install Requirements**:
   ```bash
   pip install -r requirements.txt
   ```
## Environment Setup

- **For OpenAI users**:
  - Add your OpenAI API key to the `.env` file:
    ```
    OPENAI_API_KEY=your_api_key_here
    ```

- **For Google/PaLM/VertexAI users**:
  - Set up the Weaviate database using Docker:
    ```bash
    docker compose up -d
    ```
  - Add your Google credentials JSON file to the project folder.
  - Update the `.env` file with your Google credentials file path:
    ```
    GCRED_FILE=path_to_your_google_credentials.json
    ```

## Configure `.botignore`

- The `.botignore` file works similarly to `.gitignore`. Create this file in your project root directory and list the files or directories you want AutoCoder to ignore.

## Assistant and Thread IDs

- After creating your assistant, save the assistant ID in the `.env` file:
```bash
ASSISTANT_ID=your_assistant_id_here
```
- Save the thread ID in the `.env` file for persistence:
```bash
THREAD_ID=your_thread_id_here
```
