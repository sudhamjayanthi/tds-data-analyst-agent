import os
import json
import base64
import io
from typing import List, Optional
import logging
import sys
import pickle
import hashlib
from pathlib import Path

import google.generativeai as genai
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from bs4 import BeautifulSoup
import numpy as np
from scipy import stats
from dotenv import load_dotenv
import duckdb

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Simple terminal colors for readability
class TerminalColors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    PURPLE = "\033[95m"
    VIOLET = "\033[35m"
    RESET = "\033[0m"


def log_model_input(message: str) -> None:
    logger.info(f"{TerminalColors.BLUE}{message}{TerminalColors.RESET}")


def log_model_output(message: str) -> None:
    logger.info(f"{TerminalColors.PURPLE}{message}{TerminalColors.RESET}")


def log_model_code(message: str) -> None:
    logger.info(f"{TerminalColors.VIOLET}{message}{TerminalColors.RESET}")


def log_our_error(message: str) -> None:
    logger.error(f"{TerminalColors.RED}{message}{TerminalColors.RESET}")


def log_model_error(message: str) -> None:
    logger.warning(f"{TerminalColors.YELLOW}{message}{TerminalColors.RESET}")


def log_info(message: str) -> None:
    logger.info(f"{TerminalColors.BLUE}{message}{TerminalColors.RESET}")


# Initialize FastAPI app
app = FastAPI(
    title="Data Analyst Agent", description="API for data analysis using LLMs"
)

# Configure Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is required")

genai.configure(api_key=GEMINI_API_KEY)


class DataStorage:
    """Simple file-based storage for datasets scraped by the model"""

    def __init__(self, storage_dir: str = "data_storage"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        logger.info(f"Initialized data storage at: {self.storage_dir}")

    def generate_key(self, url_or_description: str) -> str:
        """Generate a unique key for a dataset based on URL or description"""
        return hashlib.md5(url_or_description.encode()).hexdigest()[:12]

    def save_dataset(self, key: str, data: pd.DataFrame, metadata: dict = None) -> str:
        """Save a dataset with optional metadata"""
        try:
            file_path = self.storage_dir / f"{key}.pkl"

            # Save the data and metadata together
            storage_data = {
                "dataframe": data,
                "metadata": metadata or {},
                "timestamp": pd.Timestamp.now().isoformat(),
            }

            with open(file_path, "wb") as f:
                pickle.dump(storage_data, f)

            logger.info(f"Dataset saved to {file_path}")
            return str(file_path)
        except Exception as e:
            logger.error(f"Failed to save dataset: {e}")
            raise

    def load_dataset(self, key: str) -> Optional[dict]:
        """Load a dataset by key"""
        try:
            file_path = self.storage_dir / f"{key}.pkl"

            if not file_path.exists():
                logger.info(f"Dataset {key} not found")
                return None

            with open(file_path, "rb") as f:
                storage_data = pickle.load(f)

            logger.info(f"Dataset {key} loaded successfully")
            return storage_data
        except Exception as e:
            logger.error(f"Failed to load dataset {key}: {e}")
            return None

    def list_datasets(self) -> List[str]:
        """List all available dataset keys"""
        try:
            keys = [f.stem for f in self.storage_dir.glob("*.pkl")]
            logger.info(f"Found {len(keys)} stored datasets")
            return keys
        except Exception as e:
            logger.error(f"Failed to list datasets: {e}")
            return []

    def dataset_exists(self, key: str) -> bool:
        """Check if a dataset exists"""
        file_path = self.storage_dir / f"{key}.pkl"
        return file_path.exists()


class DataAnalystAgent:
    def __init__(self):
        self.model = genai.GenerativeModel(
            "gemini-2.5-pro",
        )
        self.mock_mode = os.getenv("MOCK_MODE", "false").lower() == "true"
        self.data_storage = DataStorage()

    async def analyze_data(self, questions: str, files: List[UploadFile] = None) -> str:
        """Main analysis function that processes questions and optional files"""
        logger.info("Starting data analysis...")
        try:
            dataframes = {}
            file_info = []
            if files:
                logger.info(f"Processing {len(files)} uploaded files...")
                for file in files:
                    filename = file.filename
                    logger.info(f"Reading file: {filename}")
                    content = await file.read()
                    if filename.endswith(".csv"):
                        try:
                            df = pd.read_csv(io.BytesIO(content))
                            dataframes[filename] = df
                            file_info.append(
                                {
                                    "filename": filename,
                                    "type": "csv",
                                    "columns": df.columns.tolist(),
                                    "shape": df.shape,
                                }
                            )
                        except Exception as e:
                            logger.error(f"Failed to read {filename} as CSV: {str(e)}")

            async def run_python_code(code: str) -> str:
                """
                Executes Python code in a sandboxed environment for data analysis and visualization.
                The code has access to a 'dataframes' dictionary containing pandas DataFrames of the uploaded files.
                For plotting, use matplotlib/seaborn and save the plot.
                The function will return a dictionary containing any printed output and the base64 encoded plot if generated.
                """
                safe_builtins = {
                    "abs",
                    "all",
                    "any",
                    "ascii",
                    "bin",
                    "bool",
                    "bytearray",
                    "bytes",
                    "callable",
                    "chr",
                    "classmethod",
                    "complex",
                    "delattr",
                    "dict",
                    "dir",
                    "divmod",
                    "enumerate",
                    "filter",
                    "float",
                    "format",
                    "frozenset",
                    "getattr",
                    "globals",
                    "hasattr",
                    "hash",
                    "help",
                    "hex",
                    "id",
                    "int",
                    "isinstance",
                    "issubclass",
                    "iter",
                    "len",
                    "list",
                    "locals",
                    "map",
                    "max",
                    "memoryview",
                    "min",
                    "next",
                    "object",
                    "oct",
                    "ord",
                    "pow",
                    "print",
                    "property",
                    "range",
                    "repr",
                    "reversed",
                    "round",
                    "set",
                    "setattr",
                    "slice",
                    "sorted",
                    "staticmethod",
                    "str",
                    "sum",
                    "super",
                    "tuple",
                    "type",
                    "vars",
                    "zip",
                }
                # Handle __builtins__ properly (can be dict or module)
                if isinstance(__builtins__, dict):
                    safe_builtins_dict = {
                        k: __builtins__[k] for k in safe_builtins if k in __builtins__
                    }
                else:
                    safe_builtins_dict = {
                        k: getattr(__builtins__, k)
                        for k in safe_builtins
                        if hasattr(__builtins__, k)
                    }

                # Add essential builtins for imports
                safe_builtins_dict["__import__"] = __import__
                safe_builtins_dict["__name__"] = "__main__"
                safe_builtins_dict["__file__"] = "<sandbox>"

                safe_globals = {
                    "__builtins__": safe_builtins_dict,
                    "pd": pd,
                    "np": np,
                    "sns": sns,
                    "plt": plt,
                    "stats": stats,
                    "dataframes": dataframes,
                    "io": io,
                    "base64": base64,
                    "duckdb": duckdb,
                    "requests": requests,
                    "BeautifulSoup": BeautifulSoup,
                    "json": json,
                    # Add modules directly to avoid import issues
                    "pandas": pd,
                    "numpy": np,
                    "seaborn": sns,
                    "matplotlib": matplotlib,
                }
                try:
                    logger.info("Starting Python code execution")
                    logger.info(
                        f"Code to execute: {code}"
                    )  # Log first 500 chars

                    output_capture = io.StringIO()
                    original_stdout = sys.stdout
                    sys.stdout = output_capture

                    exec(code, safe_globals, local_scope)

                    sys.stdout = original_stdout
                    printed_output = output_capture.getvalue()
                    logger.info(f"Python execution stdout: {printed_output}")

                    # Check if a plot was created
                    plot_data = None
                    fig_nums = plt.get_fignums()
                    logger.info(
                        f"Number of matplotlib figures created: {len(fig_nums)}"
                    )

                    if fig_nums:
                        buffer = io.BytesIO()
                        plt.savefig(buffer, format="png", dpi=100, bbox_inches="tight")
                        buffer.seek(0)
                        plot_data = base64.b64encode(buffer.getvalue()).decode()
                        logger.info(f"Plot data length: {len(plot_data)} characters")
                        plt.close("all")

                    result = {"stdout": printed_output}
                    if plot_data:
                        result["plot"] = f"data:image/png;base64,{plot_data}"

                    logger.info(f"Python execution result keys: {list(result.keys())}")
                    return json.dumps(result)
                except Exception as e:
                    log_model_error(f"Python execution failed: {str(e)}")
                    import traceback

                    full_traceback = traceback.format_exc()
                    log_model_error(f"Python execution traceback: {full_traceback}")

                    # Provide detailed error information to help the model debug
                    available_libs = sorted(
                        [
                            "pandas (pd)",
                            "numpy (np)",
                            "matplotlib.pyplot (plt)",
                            "seaborn (sns)",
                            "requests",
                            "beautifulsoup4 (BeautifulSoup)",
                            "json",
                            "io",
                            "base64",
                            "duckdb",
                            "lxml (if used by pandas/bs4)",
                            "html5lib (if used by pandas)",
                        ]
                    )
                    error_info = {
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "traceback": full_traceback,
                        # "code_snippet": code[:500] + "..." if len(code) > 500 else code,
                        "available_modules": available_libs,
                        "available_dataframes": list(dataframes.keys())
                        if dataframes
                        else [],
                        "debugging_tips": [
                            "For ImportError/ModuleNotFoundError: Use the pre-imported modules or specify a pure-Python approach; available modules are listed in 'available_modules'",
                            "For NameError: Check variable names and ensure they are defined before use",
                            "For AttributeError: Verify object methods and properties exist",
                            "For KeyError: Check dictionary keys and DataFrame column names",
                            "For SyntaxError: Review code syntax, brackets, quotes, and indentation",
                            "For web scraping: Use requests.get() and BeautifulSoup for parsing HTML",
                            "For data analysis: pandas DataFrames are in the 'dataframes' dictionary",
                        ],
                        "suggestion": "Analyze the error type and traceback, then apply the appropriate fix using only available modules.",
                    }
                    return json.dumps(error_info)

            local_scope = {}

            async def save_dataset(
                url_or_description: str,
                dataframe_var_name: str = "df",
                metadata: dict = None,
            ) -> str:
                """
                Save a scraped dataset for future use. The dataset should be stored in a variable (default 'df').
                Args:
                    url_or_description: URL or description to generate a unique key for the dataset
                    dataframe_var_name: Name of the variable containing the DataFrame (default: 'df')
                    metadata: Optional metadata to store with the dataset
                Returns:
                    JSON string with save result and dataset key
                """
                try:
                    # Get the dataframe from the local scope
                    if dataframe_var_name not in local_scope:
                        return json.dumps(
                            {
                                "error": f"Variable '{dataframe_var_name}' not found. Make sure to scrape and store data in this variable first.",
                                "available_variables": [
                                    k
                                    for k, v in local_scope.items()
                                    if isinstance(v, pd.DataFrame)
                                ],
                            }
                        )

                    df = local_scope[dataframe_var_name]
                    if not isinstance(df, pd.DataFrame):
                        return json.dumps(
                            {
                                "error": f"Variable '{dataframe_var_name}' is not a DataFrame",
                                "variable_type": str(type(df)),
                            }
                        )

                    # Generate key and save
                    key = self.data_storage.generate_key(url_or_description)
                    file_path = self.data_storage.save_dataset(key, df, metadata)

                    return json.dumps(
                        {
                            "success": True,
                            "message": "Dataset saved successfully",
                            "key": key,
                            "file_path": file_path,
                            "shape": df.shape,
                            "columns": df.columns.tolist(),
                        }
                    )

                except Exception as e:
                    return json.dumps({"error": f"Failed to save dataset: {str(e)}"})

            async def load_dataset(key: str, variable_name: str = "loaded_df") -> str:
                """
                Load a previously saved dataset into a variable.
                Args:
                    key: The dataset key (12-character hash)
                    variable_name: Name for the variable to store the loaded DataFrame (default: 'loaded_df')
                Returns:
                    JSON string with load result
                """
                try:
                    storage_data = self.data_storage.load_dataset(key)

                    if storage_data is None:
                        available_keys = self.data_storage.list_datasets()
                        return json.dumps(
                            {
                                "error": f"Dataset with key '{key}' not found",
                                "available_datasets": available_keys,
                            }
                        )

                    # Load the dataframe into the local scope
                    df = storage_data["dataframe"]
                    local_scope[variable_name] = df

                    return json.dumps(
                        {
                            "success": True,
                            "message": f"Dataset loaded into variable '{variable_name}'",
                            "key": key,
                            "shape": df.shape,
                            "columns": df.columns.tolist(),
                            "metadata": storage_data.get("metadata", {}),
                            "saved_at": storage_data.get("timestamp", "unknown"),
                        }
                    )

                except Exception as e:
                    return json.dumps({"error": f"Failed to load dataset: {str(e)}"})

            async def list_saved_datasets() -> str:
                """
                List all saved datasets with their keys and metadata.
                Returns:
                    JSON string with list of available datasets
                """
                try:
                    keys = self.data_storage.list_datasets()
                    datasets_info = []

                    for key in keys:
                        storage_data = self.data_storage.load_dataset(key)
                        if storage_data:
                            datasets_info.append(
                                {
                                    "key": key,
                                    "shape": storage_data["dataframe"].shape,
                                    "columns": storage_data[
                                        "dataframe"
                                    ].columns.tolist(),
                                    "metadata": storage_data.get("metadata", {}),
                                    "saved_at": storage_data.get(
                                        "timestamp", "unknown"
                                    ),
                                }
                            )

                    return json.dumps(
                        {
                            "success": True,
                            "total_datasets": len(datasets_info),
                            "datasets": datasets_info,
                        }
                    )

                except Exception as e:
                    return json.dumps({"error": f"Failed to list datasets: {str(e)}"})

            tools = [run_python_code, save_dataset, load_dataset, list_saved_datasets]
            model_with_tools = genai.GenerativeModel("gemini-2.5-pro", tools=tools)
            chat = model_with_tools.start_chat()

            prompt = f"""
            <system_instruction>
            You are an expert data analyst AI agent with advanced coding, analytical, and data management capabilities. Your mission is to solve complex data analysis tasks through systematic problem-solving, self-correction, and precise execution.

            **CORE IDENTITY & CAPABILITIES:**
            - Act as a senior data scientist with 10+ years of experience
            - Expert in Python, statistics, data visualization, web scraping, database operations, and data persistence
            - Methodical, detail-oriented, and persistent in problem-solving
            - Can debug and fix your own code when errors occur
            - Efficiently manage data by saving and reusing scraped datasets when valuable

            **YOUR TASK:**
            {questions}

            **AVAILABLE DATA SOURCES:**
            {json.dumps(file_info, indent=2)}

            **EXECUTION PROTOCOL (Chain-of-Thought + ReAct):**

            1. **ANALYZE & PLAN:**
               - Break down the request into clear, manageable subtasks
               - Identify required data sources: uploaded files, web scraping targets, database queries
               - Determine necessary libraries, methods, and exact output format
               - Plan execution sequence: data acquisition → cleaning → analysis → visualization → formatting

            2. **AVAILABLE TOOLS:**
               - `run_python_code(code)`: Execute Python code with full library access
               - `save_dataset(url_or_description, dataframe_var_name, metadata)`: Save scraped data for reuse
               - `load_dataset(key, variable_name)`: Load previously saved data
               - `list_saved_datasets()`: View available saved datasets

            3. **CODE EXECUTION GUIDELINES:**
               - Pre-imported libraries: pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns, requests, BeautifulSoup, json, io, base64, duckdb
               - For web scraping: Use requests + BeautifulSoup or pandas.read_html()
               - For databases: Use duckdb for SQL queries on large datasets  
               - For plotting: Generate base64 data URIs under 100KB
               - Save valuable scraped data using `save_dataset` for future reuse

            4. **ERROR HANDLING:**
               When code execution fails:
               - Read error messages and tracebacks carefully
               - Identify the specific issue (imports, syntax, data, logic)
               - Apply targeted fixes
               - Re-execute corrected code
               - Continue until successful completion

            5. **QUALITY ASSURANCE:**
               - Validate results match original questions exactly
               - Ensure output format is precisely as requested
               - Verify calculations and data transformations
               - Confirm plots display correctly with proper specifications

            6. **OUTPUT REQUIREMENTS:**
               - Return ONLY the final formatted answer as specified
               - Use exact JSON structure requested (array or object)
               - No explanatory text, commentary, or markdown formatting
               - All answers must be complete and accurate

            **SUCCESS CRITERIA:**
            ✓ All questions answered accurately
            ✓ Output in exact format requested  
            ✓ All plots properly generated and encoded
            ✓ No errors in final execution
            ✓ Results delivered within time constraints

            Begin by analyzing the task systematically, then implement your solution step by step using clear reasoning.
            </system_instruction>
            """

            log_model_input(f"Sending prompt to Gemini:\n{prompt}")

            response = await chat.send_message_async(prompt)
            log_info(f"Initial response received: {response}")
            log_info(
                f"Response candidates: {len(response.candidates) if response.candidates else 0}"
            )

            if response.candidates:
                log_info(f"First candidate: {response.candidates[0]}")
                if response.candidates[0].content:
                    log_info(
                        f"Content parts: {len(response.candidates[0].content.parts) if response.candidates[0].content.parts else 0}"
                    )

            # Raw response logging for debugging
            try:
                for idx, cand in enumerate(response.candidates or []):
                    log_info(
                        f"Candidate {idx} role: {getattr(cand.content, 'role', 'unknown')}"
                    )
                    for pidx, part in enumerate(cand.content.parts or []):
                        if hasattr(part, "text") and part.text:
                            log_model_output(
                                f"Part {pidx} text (first 400 chars): {part.text[:400]}{'...' if len(part.text) > 400 else ''}"
                            )
                        elif hasattr(part, "function_call") and part.function_call:
                            name = getattr(part.function_call, "name", "unknown")
                            log_info(f"Part {pidx} function_call name: {name}")
                        else:
                            log_info(f"Part {pidx} type: {type(part)}")
            except Exception as dbg_e:
                log_our_error(f"Failed to log raw response: {dbg_e}")

            # Handle function calls with proper error checking and retry limits
            iteration = 0
            max_iterations = 10  # Prevent infinite loops
            consecutive_errors = 0
            max_consecutive_errors = 3  # Stop if too many consecutive errors

            while iteration < max_iterations:
                iteration += 1
                log_info(
                    f"=== Starting function call loop iteration {iteration}/{max_iterations} ==="
                )

                # Check if response has candidates and content
                if not response.candidates or len(response.candidates) == 0:
                    logger.warning("No candidates in response, exiting loop")
                    break

                candidate = response.candidates[0]
                log_info(
                    f"Processing candidate finish_reason: {candidate.finish_reason}"
                )

                if (
                    not candidate.content
                    or not candidate.content.parts
                    or len(candidate.content.parts) == 0
                ):
                    logger.warning("No content parts in candidate, exiting loop")
                    break

                # Find a function call part if present, else gather text parts
                function_part = None
                all_text_parts = []
                for p in candidate.content.parts or []:
                    if (
                        hasattr(p, "function_call")
                        and p.function_call
                        and getattr(p.function_call, "name", None)
                    ):
                        function_part = p
                        break
                    if hasattr(p, "text") and p.text:
                        all_text_parts.append(p.text)

                if function_part is None:
                    # No function call; treat as final text response
                    final_text_inline = "\n".join(all_text_parts).strip()
                    if final_text_inline:
                        log_model_output(
                            f"Final response text (collected from parts): {final_text_inline[:800]}{'...' if len(final_text_inline) > 800 else ''}"
                        )
                        return final_text_inline
                    # As a last resort try response.text
                    try:
                        response_text = response.text
                        log_model_output(
                            f"Final response text: {response_text[:800]}{'...' if len(response_text) > 800 else ''}"
                        )
                        return response_text
                    except Exception as text_error:
                        logger.warning(
                            f"Cannot get response.text in loop: {text_error}"
                        )
                        break

                part = function_part
                log_info(f"Function call found: {part.function_call}")

                function_call = part.function_call
                tool_name = function_call.name
                log_info(f"Model requested to use tool: {tool_name}")

                # Validate tool arguments
                if not hasattr(function_call, "args") or not function_call.args:
                    logger.warning("No arguments provided for function call")
                    tool_args = {}
                else:
                    # Handle different types of args (dict-like or proto message)
                    try:
                        if hasattr(function_call.args, "items"):
                            tool_args = {
                                key: value for key, value in function_call.args.items()
                            }
                        else:
                            # Convert proto message to dict
                            tool_args = dict(function_call.args)
                    except Exception as e:
                        log_our_error(f"Failed to process function call args: {e}")
                        tool_args = {}

                log_info(f"Tool arguments: {json.dumps(tool_args, indent=2)}")

                # Execute the appropriate tool with error handling
                try:
                    if tool_name == "run_python_code":
                        log_info(f"Executing run_python_code with args: {tool_args}")
                        tool_result = await run_python_code(**tool_args)
                        log_info("Tool execution completed successfully")
                        log_info(f"Tool result type: {type(tool_result)}")
                        log_info(f"Tool result: {tool_result}")

                        # Check if the result contains an error
                        try:
                            parsed_result = (
                                json.loads(tool_result)
                                if isinstance(tool_result, str)
                                else tool_result
                            )
                            if (
                                isinstance(parsed_result, dict)
                                and "error" in parsed_result
                            ):
                                consecutive_errors += 1
                                log_model_error(
                                    f"Code execution error detected (consecutive: {consecutive_errors}/{max_consecutive_errors})"
                                )
                                if consecutive_errors >= max_consecutive_errors:
                                    log_our_error(
                                        "Too many consecutive errors, stopping execution"
                                    )
                                    break
                            else:
                                consecutive_errors = 0  # Reset on successful execution
                        except (json.JSONDecodeError, TypeError, KeyError):
                            pass  # If we can't parse the result, continue anyway

                    elif tool_name == "save_dataset":
                        log_info(f"Executing save_dataset with args: {tool_args}")
                        tool_result = await save_dataset(**tool_args)
                        log_info("Save dataset completed successfully")
                        consecutive_errors = 0

                    elif tool_name == "load_dataset":
                        log_info(f"Executing load_dataset with args: {tool_args}")
                        tool_result = await load_dataset(**tool_args)
                        log_info("Load dataset completed successfully")
                        consecutive_errors = 0

                    elif tool_name == "list_saved_datasets":
                        log_info(
                            f"Executing list_saved_datasets with args: {tool_args}"
                        )
                        tool_result = await list_saved_datasets(**tool_args)
                        log_info("List datasets completed successfully")
                        consecutive_errors = 0

                    else:
                        log_our_error(f"Unknown tool requested: {tool_name}")
                        tool_result = json.dumps(
                            {"error": f"Unknown tool: {tool_name}"}
                        )
                        consecutive_errors += 1
                except Exception as tool_error:
                    log_our_error(f"Tool execution failed: {str(tool_error)}")
                    import traceback

                    log_our_error(f"Tool execution traceback: {traceback.format_exc()}")
                    tool_result = json.dumps(
                        {"error": f"Tool execution failed: {str(tool_error)}"}
                    )
                    consecutive_errors += 1

                log_info("Sending tool output back to the model.")

                # Send tool result back to model with error handling
                try:
                    # Try different ways to create function response
                    log_model_code("Attempting to send tool result back to model")
                    log_model_code(f"Tool result type: {type(tool_result)}")
                    log_model_code(f"Tool result content: {tool_result}")

                    # Parse tool result if it's a string
                    try:
                        parsed_result = (
                            json.loads(tool_result)
                            if isinstance(tool_result, str)
                            else tool_result
                        )
                        log_info(f"Parsed result: {parsed_result}")
                    except Exception as parse_error:
                        logger.warning(
                            f"Failed to parse tool result as JSON: {parse_error}"
                        )
                        parsed_result = {"output": tool_result}

                    # Try the google.ai approach first
                    try:
                        function_response = genai.types.FunctionResponse(
                            name=tool_name,
                            response=parsed_result,
                        )
                        logger.info(
                            f"Created function response with genai.types: {function_response}"
                        )
                        response = await chat.send_message_async(function_response)
                        logger.info(
                            f"Tool response received via genai.types: {response}"
                        )
                        logger.info("Continuing to next loop iteration...")
                    except Exception as genai_error:
                        logger.warning(f"genai.types approach failed: {genai_error}")

                        # Try alternative - create content manually
                        try:
                            from google.generativeai.types import content_types

                            function_response = content_types.FunctionResponse(
                                name=tool_name,
                                response=parsed_result,
                            )
                            logger.info(
                                f"Created function response with content_types: {function_response}"
                            )
                            response = await chat.send_message_async(function_response)
                            logger.info(
                                f"Tool response received via content_types: {response}"
                            )
                            logger.info("Continuing to next loop iteration...")
                        except Exception as content_error:
                            logger.warning(
                                f"content_types approach failed: {content_error}"
                            )

                            # Last resort - send as plain text message
                            text_message = f"Function {tool_name} executed successfully. Result: {tool_result}"
                            logger.info(f"Sending as text message: {text_message}")
                            response = await chat.send_message_async(text_message)
                            logger.info(f"Tool response received via text: {response}")
                            logger.info("Continuing to next loop iteration...")

                except Exception as chat_error:
                    logger.error(
                        f"All approaches failed to send tool result back to model: {str(chat_error)}"
                    )
                    break

            # Check if we exited due to max iterations
            if iteration >= max_iterations:
                logger.warning(
                    f"Reached maximum iterations ({max_iterations}), stopping execution"
                )
            elif consecutive_errors >= max_consecutive_errors:
                logger.warning(
                    f"Stopped due to too many consecutive errors ({consecutive_errors})"
                )

            log_info(f"=== Exited function call loop after {iteration} iterations ===")
            log_info("Processing final response from model")
            log_info(f"Final response object: {response}")
            log_info(
                f"Final response candidates: {len(response.candidates) if response.candidates else 0}"
            )

            try:
                # Prefer concatenating text parts explicitly
                if (
                    response.candidates
                    and response.candidates[0].content
                    and response.candidates[0].content.parts
                ):
                    texts = [
                        p.text
                        for p in response.candidates[0].content.parts
                        if hasattr(p, "text") and p.text
                    ]
                    if texts:
                        final_text = "\n".join(texts)
                        log_model_output(
                            f"Final response text (post-loop): {final_text[:800]}{'...' if len(final_text) > 800 else ''}"
                        )
                        return final_text
                final_text = response.text
                log_model_output(
                    f"Final response text (fallback): {final_text[:800]}{'...' if len(final_text) > 800 else ''}"
                )
                return final_text
            except Exception as text_error:
                log_our_error(f"Failed to get response.text: {text_error}")

                # Try to extract text manually from candidates
                try:
                    if response.candidates and len(response.candidates) > 0:
                        candidate = response.candidates[0]
                        if candidate.content and candidate.content.parts:
                            for part in candidate.content.parts:
                                if hasattr(part, "text") and part.text:
                                    log_model_output(
                                        f"Extracted text from part: {part.text[:800]}{'...' if len(part.text) > 800 else ''}"
                                    )
                                    return part.text
                                else:
                                    logger.warning(f"Part has no text: {part}")

                    # If no text found, return error string (no mock)
                    logger.warning("No text found in response")
                    return json.dumps({"error": "no text found in model response"})

                except Exception as extract_error:
                    log_our_error(f"Failed to extract text manually: {extract_error}")
                    return json.dumps(
                        {"error": f"failed to extract text: {extract_error}"}
                    )

        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            error_msg = str(e)
            if "quota" in error_msg.lower() or "429" in error_msg:
                raise HTTPException(
                    status_code=429,
                    detail="API quota exceeded. Please try again later or check your Gemini API billing.",
                )
            raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


# Initialize the agent
agent = DataAnalystAgent()


@app.post("/api/")
async def analyze_data_endpoint(files: List[UploadFile] = File(...)):
    """Main API endpoint for data analysis"""

    try:
        # Find the questions file
        questions_content = None
        other_files = []

        for file in files:
            if file.filename == "questions.txt":
                content = await file.read()
                questions_content = content.decode("utf-8")
                # Reset file pointer
                await file.seek(0)
            else:
                other_files.append(file)

        if not questions_content:
            raise HTTPException(
                status_code=400, detail="questions.txt file is required"
            )

        # Perform analysis
        result = await agent.analyze_data(questions_content, other_files)

        # The result from the agent is now a string, wrap it in a JSON response
        return JSONResponse(content={"response": result})

    except Exception as e:
        logger.error(f"API endpoint failed: {str(e)}")
        # If the exception is an HTTPException, re-raise it
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Data Analyst Agent API is running"}


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
