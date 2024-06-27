import os
import json
import sys
import chardet
import boto3
from datetime import datetime
from langchain.agents import tool
from langchain_community.chat_models import BedrockChat
from crewai import Agent, Task, Crew
# from crewai.telemetry import Telemetry

# Disable CrewAI Anonymous Telemetry
# def noop(*args, **kwargs):
#     pass

# for attr in dir(Telemetry):
#     if callable(getattr(Telemetry, attr)) and not attr.startswith("__"):
#         setattr(Telemetry, attr, noop)

class DateTimeEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, datetime):
            return o.isoformat()
        return super().default(o)

# Config Amazon Bedrock claude3
def initialize_llm():
    """
    Initialize a Large Language Model (LLM) instance using the Bedrock Runtime service.

    Returns:
        BedrockChat: An instance of the BedrockChat class, which represents the initialized LLM.
    """
    model_id = "anthropic.claude-3-sonnet-20240229-v1:0"
    model_kwargs = {
        "max_tokens": 2048,
        "temperature": 0.0,
        "top_k": 250,
        "top_p": 1,
        "stop_sequences": ["\n\nHuman"],
    }

    # Check if AWS_REGION environment variable is set
    aws_region = os.environ.get("AWS_REGION", "us-west-2")

    bedrock_runtime = boto3.client(
        service_name="bedrock-runtime",
        region_name=aws_region,
    )

    bedrock_llm = BedrockChat(
        client=bedrock_runtime,
        model_id=model_id,
        model_kwargs=model_kwargs,
    )

    return bedrock_llm

bedrock_llm = initialize_llm()

def split_data(data, page=1):
    """Tool to split data into pages of 50 items"""
    if isinstance(data, list):
        total_pages = (len(data) + 49) // 50  # Calculate the total number of pages
        if page < 1 or page > total_pages:
            print(f"Invalid page number. Please enter a value between 1 and {total_pages}.")
            return []
        start_index = (page - 1) * 50
        end_index = start_index + 50
        return data[start_index:end_index]
    else:
        print("Input data is not a list.")
        return []

def save_review(app_id, data):
    with open(f"output/{app_id}.json", "w", encoding="utf-8") as file:
        json.dump(data, file, cls=DateTimeEncoder, ensure_ascii=False, indent=4)

@tool
def load_local_file(file_name: str, page: int = 1):
    """Tool to load local file, page must start from 1"""
    try:
        with open(file_name, "rb") as file:
            result = chardet.detect(file.read())
            encoding = result["encoding"]

        file_extension = file_name.split(".")[-1].lower()

        if file_extension == "json":
            with open(file_name, "r", encoding=encoding) as file:
                data = json.load(file)
            return split_data(data, page)

        else:
            print(f"Unsupported file extension: {file_extension}")
            return []

    except Exception as e:
        print(f"Error loading file: {e}")
        return []

review_loader = Agent(
    role='Game Forum Review Research Analyst',
    goal='If user input is a local file, you load and analyze the forum reviews from the JSON file',
    backstory='Specializes in handling and interpreting game forum posts and comments',
    verbose=True,
    tools=[load_local_file],
    allow_delegation=False,
    llm=bedrock_llm,
    memory=True,
)

researcher = Agent(
  role='Expert Game Forum reviews summarizer',
  goal='Find common issues, positive feedback, and suggestions in every user\'s reviews content, keeping the username',
  backstory="""You are a renowned expert summarizer of game forum posts and comments, known for your insightful and engaging summaries. You transform complex discussions into clear, actionable insights.""",
  verbose=True,
  allow_delegation=True,
  llm=bedrock_llm
)

def init_forum_crew(file: str, page: int = 1, output_stream=None):
    """
    This function is the main entry point for the application.
    It takes a file name and a page number as input, and returns the result of processing the forum posts and comments.
    """
    
    if output_stream is not None:
        sys.stdout = output_stream

    review_loading_task = Task(
        description=f"Load and process forum reviews from the local file '/Users/zhuchenqi/git_repo/general-review-llm/upload/{file}', if result is more than 50, process 50 results at a time",
        agent=review_loader,
        expected_output='A refined finalized version of the forum reviews in markdown format'
    )

    content_creation_task = Task(
        description="""Using the insights provided by the research Analyst agent, identify common issues, positive feedback, and suggestions. Develop an engaging summary in markdown format and translate to Chinese.""",
        agent=researcher,
        expected_output='A refined finalized version of the forum reviews summary in markdown format, translated to Chinese.'
    )

    crew = Crew(
        agents=[review_loader, researcher],
        tasks=[review_loading_task, content_creation_task],
        verbose=2, 
    )

    return crew

def interactive_mode():
    """
    This function provides an interactive mode for the user to enter the file name.
    It runs in a loop until the user enters 'q' to quit.
    """
    while True:
        file_name = input("Enter the file name (or 'q' to quit): ")
        if file_name.lower() == 'q':
            break
        page = input("Enter the page number (or 'q' to quit): ")
        try:
            page = int(page)
            if page < 1:
                print("Invalid page number. Please enter a value greater than 0.")
                continue
        except ValueError:
            print("Invalid page number. Please enter a valid integer.")
            continue
        forum_crew = init_forum_crew(file_name, page)
        result = forum_crew.kickoff()
        print(result)

def main(file_name):
    """
    This function calls the kickoff function with the provided file name and a page number of 1.
    It prints the result returned by the kickoff function.
    """
    forum_crew = init_forum_crew(file_name, 1)
    result = forum_crew.kickoff()
    print(result)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Process a local JSON file and get forum reviews summary.')
    parser.add_argument('--file_name', default='forum_posts.json', nargs='?', type=str, help='The name of the JSON file to analyze')
    parser.add_argument('-i', '--interaction', action='store_true', help='Enable interactive mode')
    args = parser.parse_args()
    if args.interaction:
        interactive_mode()
    elif args.file_name:
        main(args.file_name)
    else:
        parser.print_help()
