import os
import time
import uuid
import shutil
import json
import base64
import multiprocessing
import resource
from pathlib import Path
from typing import Optional, List, Dict, Any

# --- Core Dependencies ---
import pandas as pd
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

# --- AI and Environment Dependencies ---
import google.generativeai as genai
from dotenv import load_dotenv
from pydantic_ai import Agent

# ---- 1. Workspace & File Management ---- #

class WorkspaceManager:
    """Manages isolated file system environments for each API request."""
    def __init__(self, base_dir: str = "./agent_workspaces"):
        self.base_dir = Path(base_dir)
        self.graveyard_dir = self.base_dir / "_graveyard"
        self.base_dir.mkdir(exist_ok=True)
        self.graveyard_dir.mkdir(exist_ok=True)

    def create(self) -> str:
        """Creates a new, unique workspace and returns its ID."""
        workspace_id = uuid.uuid4().hex
        workspace_path = self.base_dir / workspace_id
        for subdir in ['inputs', 'outputs', 'temp']:
            (workspace_path / subdir).mkdir(parents=True, exist_ok=True)
        # Write metadata for tracking and cleanup
        meta = {'created_at': time.time(), 'status': 'ACTIVE'}
        (workspace_path / "meta.json").write_text(json.dumps(meta))
        return workspace_id

    def get_path(self, workspace_id: str, *subdirs) -> Path:
        """Returns a safe path within a given workspace."""
        if not workspace_id:
            raise ValueError("Workspace ID cannot be empty.")
        p = self.base_dir / workspace_id
        for subdir in subdirs:
            p = p / subdir
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def cleanup(self, workspace_id: str):
        """Moves a completed workspace to the graveyard for later deletion."""
        workspace_path = self.base_dir / workspace_id
        if workspace_path.exists():
            shutil.move(str(workspace_path), self.graveyard_dir / workspace_id)

workspace_manager = WorkspaceManager()

# ---- 2. Dynamic System Prompt for Gemini ---- #

def get_system_prompt(workspace_id: str, input_files: List[str]) -> str:
    """Generates a system prompt tailored to the specific request and workspace."""
    workspace_path = workspace_manager.get_path(workspace_id)
    input_dir = workspace_path / 'inputs'
    output_dir = workspace_path / 'outputs'
    
    file_list = "\n".join([f"- {f}" for f in input_files])

    return f"""
You are an autonomous DATA ANALYST AGENT. Your task is to analyze data and answer questions based on user-provided files.

## CRITICAL INSTRUCTIONS
1.  **Output Format is KING**: You MUST format your final output *exactly* as requested in the user's `questions.txt`. Do NOT add any extra summaries, text, or natural language. If the user asks for a JSON array, provide ONLY the JSON array.
2.  **File System is Your ONLY Reality**:
    - **Workspace ID**: {workspace_id}
    - **Read ALL inputs from**: {input_dir}
    - **Write ALL outputs to**: {output_dir}
    - **Available Input Files**:
      {file_list}
    - You CANNOT access files outside these directories. All file paths in your tool calls must be absolute paths within this workspace.
3.  **Tool Usage**:
    - Use ONLY the provided tools.
    - Call tools step-by-step to achieve the final answer.
4.  **Error Handling**: If a step fails, try to recover or move on. If you cannot produce a valid final answer, output a validly formatted JSON array with error messages as strings, like: `["Error: Could not process file X", "Error: Calculation failed"]`.

## Workflow
1.  **PLAN**: Understand the questions in `questions.txt`.
2.  **EXECUTE**: Use tools to load data from the `inputs` folder, process it, and generate any required outputs (like plots) in the `outputs` folder.
3.  **ANSWER**: Construct the final response in the exact format requested and output it.
"""

# ---- 3. Workspace-Aware AI Agent Tools ---- #

DATASETS: Dict[str, pd.DataFrame] = {}

# Helper to get safe paths
def safe_path(workspace_id: str, filename: str, subdir: str) -> str:
    return str(workspace_manager.get_path(workspace_id, subdir, filename))

@Agent.tool_plain
def load_data(name: str, source_filename: str, workspace_id: str, fmt: Optional[str] = None) -> str:
    """Loads a data file from the workspace's 'inputs' directory into a pandas DataFrame."""
    try:
        filepath = safe_path(workspace_id, source_filename, 'inputs')
        if not Path(filepath).exists():
            return f"Error: Input file not found at {filepath}"
            
        file_ext = fmt or Path(filepath).suffix.lower().lstrip('.')
        if file_ext == 'csv':
            df = pd.read_csv(filepath)
        elif file_ext in ['xls', 'xlsx']:
            df = pd.read_excel(filepath)
        elif file_ext == 'json':
            df = pd.read_json(filepath)
        else:
            return f"Error: Unsupported file format '{file_ext}'"
        
        DATASETS[name] = df
        return json.dumps({"name": name, "rows": df.shape[0], "cols": df.shape[1]})
    except Exception as e:
        return f"Error loading data: {e}"

@Agent.tool_plain
def scrape_html_table(name: str, url: str, table_index: int = 0) -> str:
    """Scrapes a table from a URL and loads it into a DataFrame."""
    try:
        tables = pd.read_html(url)
        if not tables:
            return "Error: No tables found on page."
        df = tables[table_index]
        DATASETS[name] = df
        return json.dumps({"name": name, "rows": df.shape, "cols": df.shape[1]})
    except Exception as e:
        return f"Error scraping table: {e}"

@Agent.tool_plain
def execute_pandas_code(code: str, dataset_name: str) -> str:
    """Executes a snippet of pandas code on a loaded DataFrame. The DataFrame is available as `df`."""
    if dataset_name not in DATASETS:
        return f"Error: Dataset '{dataset_name}' not found."
    df = DATASETS[dataset_name]
    try:
        # Create a local scope for exec to run in
        local_scope = {'df': df, 'pd': pd, 'np': np}
        exec(code, {}, local_scope)
        # The result of the code should be stored in a variable named 'result'
        result = local_scope.get('result', "Code executed, but no 'result' variable was set.")
        # If the result is a dataframe, update the dataset
        if isinstance(result, pd.DataFrame):
            DATASETS[dataset_name] = result
            return f"DataFrame '{dataset_name}' updated. Shape: {result.shape}"
        return json.dumps(result, default=str)
    except Exception as e:
        return f"Error executing code: {e}"

@Agent.tool_plain
def create_plot(
    name: str,
    plot_type: str,
    x: str,
    y: List[str],
    workspace_id: str,
    title: Optional[str] = None,
    output_filename: Optional[str] = None
) -> str:
    """Creates a plot (line, bar, scatter) and saves it to the workspace 'outputs' directory."""
    try:
        import matplotlib
        matplotlib.use("Agg") # Non-interactive backend
        import matplotlib.pyplot as plt

        if name not in DATASETS:
            return f"Error: Dataset '{name}' not found."
        df = DATASETS[name]

        plt.figure(figsize=(10, 6))
        if plot_type == 'line':
            for col in y:
                plt.plot(df[x], df[col], label=col)
        elif plot_type == 'bar':
            plt.bar(df[x], df[y[0]])
        elif plot_type == 'scatter':
            plt.scatter(df[x], df[y])
        else:
            return f"Error: Invalid plot type '{plot_type}'"

        plt.xlabel(x)
        plt.ylabel(', '.join(y))
        if title: plt.title(title)
        if len(y) > 1 and plot_type == 'line': plt.legend()
        plt.tight_layout()

        output_path = safe_path(workspace_id, output_filename or f"{name}_{plot_type}.png", 'outputs')
        plt.savefig(output_path)
        plt.close()
        return f"Plot saved to {output_path}"
    except Exception as e:
        return f"Error creating plot: {e}"

@Agent.tool_plain
def create_base64_image(filepath: str) -> str:
    """Reads an image file from the workspace and returns it as a base64 data URI."""
    try:
        if not Path(filepath).exists():
            return f"Error: File not found at {filepath}"
        
        with open(filepath, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
        
        ext = Path(filepath).suffix.lower().lstrip('.')
        return f"data:image/{ext};base64,{encoded_string}"
    except Exception as e:
        return f"Error encoding image: {e}"
        
# ---- 4. Agent Execution with Resource Limits ---- #

def agent_process_target(prompt: str, workspace_id: str, input_files: List[str], output_queue: multiprocessing.Queue):
    """The function that runs in a separate, resource-limited process."""
    try:
        # Set memory limits (on UNIX-like systems)
        mem_limit_mb = 2048
        resource.setrlimit(resource.RLIMIT_AS, (mem_limit_mb * 1024 * 1024, mem_limit_mb * 1024 * 1024))
        
        # Configure AI
        load_dotenv()
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment.")
        genai.configure(api_key=api_key)

        # Create agent with dynamic prompt and tools
        agent = Agent(
            model="gemini-2.5-flash",
            system_prompt=get_system_prompt(workspace_id, input_files),
            retries=1
        )
        # Register tools
        agent.add_tool(load_data)
        agent.add_tool(scrape_html_table)
        agent.add_tool(execute_pandas_code)
        agent.add_tool(create_plot)
        agent.add_tool(create_base64_image)

        result = agent.run_sync(prompt)
        output_queue.put(result.output)
    except Exception as e:
        output_queue.put(json.dumps({"error": "Agent process failed", "details": str(e)}))

def run_agent_in_process(prompt: str, workspace_id: str, input_files: List[str]) -> str:
    """Manages the agent's subprocess, enforcing a timeout."""
    ctx = multiprocessing.get_context('fork')
    output_queue = ctx.Queue()
    
    process = ctx.Process(
        target=agent_process_target,
        args=(prompt, workspace_id, input_files, output_queue)
    )
    
    process.start()
    process.join(timeout=295) # 5 minutes timeout, minus a small buffer

    if process.is_alive():
        process.terminate()
        process.join()
        # Return a valid JSON array indicating timeout
        return json.dumps(["Error: Analysis timed out after 5 minutes."])
    
    if output_queue.empty():
        return json.dumps(["Error: Agent process finished with no output."])
    
    return output_queue.get()


# ---- 5. FastAPI Application ---- #

app = FastAPI(title="Data Analyst Agent API")

@app.post("/api/")
async def analyze(request: Request):
    """
    Accepts multipart/form-data with 'questions.txt' and other optional files,
    runs the data analysis agent, and returns the result.
    """
    if "multipart/form-data" not in request.headers.get("content-type", ""):
        raise HTTPException(status_code=400, detail="Invalid content type. Must be multipart/form-data.")

    workspace_id = workspace_manager.create()
    input_files = []
    questions_content = None

    try:
        form = await request.form()
        for name, file in form.items():
            if isinstance(file, UploadFile):
                filename = file.filename
                input_files.append(filename)
                save_path = workspace_manager.get_path(workspace_id, 'inputs', filename)
                
                with open(save_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
                
                if filename == "questions.txt":
                    save_path.seek(0)
                    questions_content = save_path.read().decode('utf-8')

        if not questions_content:
            raise HTTPException(status_code=400, detail="questions.txt is a required file.")

        # Run the agent in a separate, timed process
        agent_output = run_agent_in_process(questions_content, workspace_id, input_files)
        
        # The agent should return a string that is a valid JSON array.
        # We parse it to ensure it's valid JSON and return it with the correct content type.
        try:
            # Try to parse the agent's output as JSON
            json_output = json.loads(agent_output)
            return JSONResponse(content=json_output)
        except json.JSONDecodeError:
            # If it's not valid JSON, it's an error or malformed output
            return JSONResponse(content={"error": "Agent returned non-JSON output", "raw_output": agent_output}, status_code=500)

    except Exception as e:
        # General error handling
        return JSONResponse(content={"error": "An unexpected error occurred.", "details": str(e)}, status_code=500)
    finally:
        # Schedule cleanup
        workspace_manager.cleanup(workspace_id)

@app.get("/")
def root():
    return {"message": "Data Analyst Agent API is running. POST to /api/ to analyze."}


# ---- 6. Main Entrypoint ---- #

if __name__ == "__main__":
    # Use 4 workers to handle simultaneous requests as per project spec
    uvicorn.run("main:app", host="0.0.0.0", port=8000, workers=4)

