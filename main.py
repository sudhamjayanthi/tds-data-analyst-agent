import os
import json
import base64
import io
import tempfile
from typing import List, Optional
import asyncio
import logging

import google.generativeai as genai
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from bs4 import BeautifulSoup
import duckdb
import numpy as np
from scipy import stats
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Data Analyst Agent", description="API for data analysis using LLMs"
)

# Configure Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is required")

genai.configure(api_key=GEMINI_API_KEY)


class DataAnalystAgent:
    def __init__(self):
        self.model = genai.GenerativeModel("gemini-1.5-pro")

    async def analyze_data(self, questions: str, files: List[UploadFile] = None) -> str:
        """Main analysis function that processes questions and optional files"""

        try:
            # Parse questions and determine what data sources are needed
            analysis_plan = await self._create_analysis_plan(questions, files)

            # Execute the analysis plan
            results = await self._execute_analysis(analysis_plan, files)

            return results

        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

    async def _create_analysis_plan(
        self, questions: str, files: List[UploadFile] = None
    ):
        """Create an analysis plan based on the questions"""

        file_info = []
        if files:
            for file in files:
                if file.filename != "questions.txt":
                    file_info.append(
                        {
                            "filename": file.filename,
                            "content_type": file.content_type,
                            "size": file.size if hasattr(file, "size") else None,
                        }
                    )

        prompt = f"""
        You are a data analyst agent. Analyze these questions and create a step-by-step plan:

        Questions: {questions}
        
        Available files: {file_info}
        
        Create a JSON plan with these steps:
        1. Data sourcing (if web scraping is needed)
        2. Data loading and preparation
        3. Analysis and calculations
        4. Visualization (if requested)
        
        Return ONLY a JSON object with this structure:
        {{
            "needs_web_scraping": boolean,
            "scraping_urls": [list of URLs if needed],
            "data_processing_steps": [list of steps],
            "calculations_needed": [list of calculations],
            "visualizations_needed": [list of plots with details],
            "output_format": "json_array" or "json_object"
        }}
        """

        response = await self.model.generate_content_async(prompt)

        try:
            plan = json.loads(response.text)
            return plan
        except:
            # Fallback plan if JSON parsing fails
            return {
                "needs_web_scraping": "wikipedia" in questions.lower(),
                "scraping_urls": [],
                "data_processing_steps": ["load_data", "clean_data"],
                "calculations_needed": ["basic_stats"],
                "visualizations_needed": [],
                "output_format": "json_array",
            }

    async def _execute_analysis(self, plan: dict, files: List[UploadFile] = None):
        """Execute the analysis plan"""

        data_sources = {}

        # Step 1: Data sourcing
        if plan.get("needs_web_scraping"):
            for url in plan.get("scraping_urls", []):
                data_sources[url] = await self._scrape_web_data(url)

        # Step 2: Load uploaded files
        if files:
            for file in files:
                if file.filename != "questions.txt":
                    content = await file.read()
                    data_sources[file.filename] = await self._process_file(
                        file.filename, content
                    )

        # Step 3: Perform analysis using Gemini
        analysis_result = await self._perform_gemini_analysis(plan, data_sources)

        return analysis_result

    async def _scrape_web_data(self, url: str):
        """Scrape data from web URLs"""
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, "html.parser")

            # Extract tables if present
            tables = soup.find_all("table")
            scraped_data = {"url": url, "tables": []}

            for i, table in enumerate(tables):
                try:
                    df = pd.read_html(str(table))[0]
                    scraped_data["tables"].append(
                        {
                            "table_index": i,
                            "data": df.to_dict("records")[
                                :100
                            ],  # Limit to first 100 rows
                            "columns": df.columns.tolist(),
                            "shape": df.shape,
                        }
                    )
                except:
                    continue

            return scraped_data

        except Exception as e:
            logger.error(f"Web scraping failed for {url}: {str(e)}")
            return {"error": str(e)}

    async def _process_file(self, filename: str, content: bytes):
        """Process uploaded files"""
        try:
            if filename.endswith(".csv"):
                df = pd.read_csv(io.BytesIO(content))
                return {
                    "type": "csv",
                    "data": df.to_dict("records")[:1000],  # Limit size
                    "columns": df.columns.tolist(),
                    "shape": df.shape,
                    "sample": df.head().to_dict("records"),
                }
            elif filename.endswith(".json"):
                data = json.loads(content.decode("utf-8"))
                return {"type": "json", "data": data}
            elif filename.endswith((".png", ".jpg", ".jpeg")):
                # For images, return basic info
                return {"type": "image", "filename": filename, "size": len(content)}
            else:
                # Try to read as text
                return {
                    "type": "text",
                    "content": content.decode("utf-8")[:5000],  # Limit size
                }
        except Exception as e:
            logger.error(f"File processing failed for {filename}: {str(e)}")
            return {"error": str(e)}

    async def _perform_gemini_analysis(self, plan: dict, data_sources: dict):
        """Use Gemini to perform the actual analysis"""

        # Prepare context for Gemini
        context = {"plan": plan, "data_sources": data_sources}

        prompt = f"""
        You are a data analyst. Perform the following analysis:

        Plan: {json.dumps(plan, indent=2)}
        
        Data Sources: {json.dumps(data_sources, indent=2)[:10000]}  # Limit context size
        
        Instructions:
        1. If web scraping is needed, use the scraped data from Wikipedia tables
        2. Perform all calculations accurately
        3. If visualizations are needed, provide Python code that I can execute
        4. Return results in the requested format
        
        For calculations involving correlations, use proper statistical methods.
        For counting and filtering, be precise with the data.
        
        If a visualization is requested:
        - Provide complete matplotlib/seaborn code
        - Ensure the plot has proper labels and styling
        - Include regression lines if requested
        - Make plots look professional
        
        Return your response as a JSON object with:
        {{
            "calculations": {{"description": "value"}},
            "visualization_code": "python code string or null",
            "final_answer": "the answer in requested format"
        }}
        """

        response = await self.model.generate_content_async(prompt)

        try:
            analysis = json.loads(response.text)
        except:
            # If JSON parsing fails, create a structured response
            analysis = {
                "calculations": {},
                "visualization_code": None,
                "final_answer": response.text,
            }

        # Execute visualization code if provided
        if analysis.get("visualization_code"):
            viz_result = await self._execute_visualization(
                analysis["visualization_code"], data_sources
            )
            if viz_result:
                # Replace or add the visualization to the final answer
                if isinstance(analysis["final_answer"], list):
                    # For array format, replace the last element if it's a plot
                    if len(analysis["final_answer"]) > 0 and "data:image" in str(
                        analysis["final_answer"][-1]
                    ):
                        analysis["final_answer"][-1] = viz_result
                    else:
                        analysis["final_answer"].append(viz_result)
                elif isinstance(analysis["final_answer"], dict):
                    # For object format, find the plot key and replace
                    for key, value in analysis["final_answer"].items():
                        if "plot" in key.lower() or "data:image" in str(value):
                            analysis["final_answer"][key] = viz_result
                            break

        return analysis["final_answer"]

    async def _execute_visualization(self, code: str, data_sources: dict):
        """Execute visualization code safely"""
        try:
            # Create a safe execution environment
            exec_globals = {
                "plt": plt,
                "sns": sns,
                "pd": pd,
                "np": np,
                "data_sources": data_sources,
                "base64": base64,
                "io": io,
            }

            # Execute the code
            exec(code, exec_globals)

            # Save plot to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format="png", dpi=100, bbox_inches="tight")
            buffer.seek(0)
            plot_data = base64.b64encode(buffer.getvalue()).decode()
            plt.close()

            # Check size limit (100KB)
            if len(plot_data) > 100000:
                # Reduce quality and try again
                buffer = io.BytesIO()
                plt.savefig(buffer, format="png", dpi=72, bbox_inches="tight")
                buffer.seek(0)
                plot_data = base64.b64encode(buffer.getvalue()).decode()
                plt.close()

            return f"data:image/png;base64,{plot_data}"

        except Exception as e:
            logger.error(f"Visualization execution failed: {str(e)}")
            return None


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

        return JSONResponse(content=result)

    except Exception as e:
        logger.error(f"API endpoint failed: {str(e)}")
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
