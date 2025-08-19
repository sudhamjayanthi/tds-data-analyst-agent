import os
import google.generativeai as genai
from pydantic_ai import Agent
from dotenv import load_dotenv
from pydantic_ai.llm import GoogleGemini

# Load environment variables from .env file
load_dotenv()

# Configure the Gemini 2.5 Pro model
try:
    api_key = os.environ["GEMINI_API_KEY"]
    genai.configure(api_key=api_key)
except KeyError:
    print("Error: GEMINI_API_KEY not found.")
    print("Please create a .env file and add your GEMINI_API_KEY.")
    exit()
except Exception as e:
    print(f"An error occurred during model initialization: {e}")
    exit()

llm = GoogleGemini(api_key=api_key)

# Create a Pydantic AI agent
agent = Agent(llm=llm)


@agent.tool_plain
def execute_code(code: str) -> str:
    """
    Executes the provided Python code and returns the output.
    The code should use the 'result' variable to store the output.
    """
    try:
        exec_globals = {}
        exec(code, exec_globals)
        return str(exec_globals.get("result", "Execution completed without a result."))
    except Exception as e:
        return f"Error during execution: {e}"


def main():
    """Main function to run the interactive AI agent."""
    print("AI Agent is ready. Type 'exit' to quit.")
    while True:
        try:
            prompt = input("You: ")
            if prompt.lower() == "exit":
                break

            # Use the agent to process the prompt
            response = agent.run(prompt)
            print(f"AI: {response}")

        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break
        except Exception as e:
            print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
