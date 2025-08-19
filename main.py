import os

import google.generativeai as genai
from dotenv import load_dotenv
from pydantic_ai import Agent
import pandas as pd
import numpy as np

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

system_prompt = """
You are a DATA ANALYSIS AGENT specialized in web data scraping, preprocessing, exploratory analysis, statistics, and charting.

## Mission
Given a user request, you will:
1) PLAN the workflow.
2) CALL the appropriate tools step by step.
3) RETURN a concise FINAL ANSWER with any file paths produced (datasets, charts, exports).

## Hard Rules 
- Use ONLY the registered tools listed under TOOL CATALOG. Do NOT invent tool names or arguments.
- Prefer custom code for simpler tasks. Use tools only for complex operations & reliability.
- Each tool call must include exactly the arguments shown in its signature (no extra keys).
- Never mix natural language inside a tool call. Narration must be separate from tool calls.
- If a request is purely explanatory, respond in plain text (do NOT call a tool).
- When producing Python via execute_code, the snippet MUST assign the final value to a variable named `result`.
- All file outputs must be written under ./outputs (create it if needed in code).

## Step-by-step Protocol
Always follow these steps in order:
1) PLAN: Brief plan of steps you will take.
2) SCRAPE or LOAD: Use fetch_html/read_csv/read_json as appropriate.
3) PARSE/BUILD DATA: Use parse_html_table or execute_code to construct a DataFrame and put it in the session store.
4) CLEAN/PREPROCESS: Use transform_dataset for selections, filters, type parsing, missing values, etc.
5) ANALYZE: Use describe_dataset or execute_code for custom stats/tests.
6) CHART: Use plot_dataset to save figures under ./outputs and report the path.
7) EXPORT (optional): Use export_dataset if needed by the user.
8) FINAL ANSWER: Summarize findings and include any output paths.

## Response Format
- Start with: PLAN: <one or two lines>
- Then perform tool calls as needed.
- End with: FINAL ANSWER: <key results, paths, next steps>

## Examples (strictly illustrative — adapt column names as needed)

Example A: Scrape a table, build a dataset, clean, describe, plot
PLAN: fetch → parse → create dataset → clean → describe → chart → final
1) fetch_html(url="https://example.com/table")
2) parse_html_table(html="<HTML_FROM_STEP_1>", selector="table.data", index=0)
3) create_dataset_from_csv(csv_text="<CSV_FROM_STEP_2>", name="web")
4) transform_dataset(name="web", ops_json='{"parse_dates":["date"],"select":["date","value"]}')
5) describe_dataset(name="web")
6) plot_dataset(name="web", kind="line", x="date", y="value", output_path="./outputs/web_line.png")
FINAL ANSWER: Summarize key stats and include ./outputs/web_line.png

Example B: Load CSV direct and filter
PLAN: load CSV → filter → describe → export
1) read_csv(source="https://site.com/data.csv", name="sales")
2) transform_dataset(name="sales", ops_json='{"filter":"region == \\'APAC\\'"}')
3) describe_dataset(name="sales")
4) export_dataset(name="sales", path="sales_apac.csv", format="csv")
FINAL ANSWER: Report shape and ./outputs/sales_apac.csv

If unsure about a step, explain briefly and proceed with the safest next action.
"""

# Create a Pydantic AI agent
agent = Agent(
    model="gemini-2.5-flash",
    system_prompt=system_prompt,  # string from above
    retries=2,  # helps recover from MALFORMED_FUNCTION_CALL
)

# ---- Add above main(), after `agent = Agent(...)` ----
from typing import Optional, List, Dict
import io, os, json

DATASETS: dict[str, "pandas.DataFrame"] = {}

@agent.tool_plain
def load_data(name: str, source: str, fmt: Optional[str] = None) -> str:
    """
    Load a dataset into memory as DATASETS[name].
    - source: local path or URL. Supports CSV/TSV/XLSX/JSON; HTML tables via 'scrape_table' instead.
    - fmt: optional override, one of ["csv","tsv","xlsx","json"].
    Returns: JSON with {"name":..., "rows":..., "cols":..., "columns":[...]}.
    """
    try:
        fmt = (fmt or os.path.splitext(source.lower())[1].lstrip(".") or "").replace(
            "htm", ""
        )
        if fmt in ("", "csv"):
            df = pd.read_csv(source)
        elif fmt == "tsv":
            df = pd.read_csv(source, sep="\t")
        elif fmt in ("xls", "xlsx"):
            df = pd.read_excel(source)
        elif fmt == "json":
            df = pd.read_json(source, orient="records")
        else:
            return f"Error: Unsupported fmt '{fmt}'."
        DATASETS[name] = df
        return json.dumps(
            {
                "name": name,
                "rows": int(df.shape[0]),
                "cols": int(df.shape[1]),
                "columns": list(map(str, df.columns)),
            }
        )
    except Exception as e:
        return f"Error loading data: {e}"


@agent.tool_plain
def scrape_table(
    name: str, url: str, selector: Optional[str] = None, table_index: int = 0
) -> str:
    """
    Scrape an HTML table and store as DATASETS[name].
    - Uses pandas.read_html; if selector is provided, it's used as match/attrs.
    - table_index: which table to pick if multiple.
    Returns JSON summary like load_data.
    """
    try:
        tables = pd.read_html(url, match=selector if selector else None)
        if not tables:
            return "Error: No tables found."
        df = tables[table_index]
        DATASETS[name] = df
        return json.dumps(
            {
                "name": name,
                "rows": int(df.shape[0]),
                "cols": int(df.shape[1]),
                "columns": list(map(str, df.columns)),
            }
        )
    except Exception as e:
        return f"Error scraping table: {e}"


@agent.tool_plain
def profile(name: str, sample_rows: int = 5) -> str:
    """
    Quick profile of DATASETS[name]: shape, dtypes, null counts, head(sample_rows).
    Returns JSON with keys: rows, cols, dtypes, nulls, head.
    """
    try:
        df = DATASETS[name]
        info = {
            "rows": int(df.shape[0]),
            "cols": int(df.shape[1]),
            "dtypes": {c: str(t) for c, t in df.dtypes.items()},
            "nulls": {c: int(df[c].isna().sum()) for c in df.columns},
            "head": df.head(sample_rows).to_dict(orient="records"),
        }
        return json.dumps(info)
    except Exception as e:
        return f"Error profiling: {e}"


@agent.tool_plain
def clean_basic(
    name: str,
    target: Optional[str] = None,
    drop_duplicates: bool = True,
    strip_whitespace: bool = True,
    parse_dates: Optional[List[str]] = None,
) -> str:
    """
    Basic cleanup on DATASETS[name]:
      - drop_duplicates
      - strip leading/trailing whitespace from object columns
      - parse_dates: list of columns to convert to datetime (UTC-naive)
    Stores result to DATASETS[target or name]; returns JSON with rows_before/after.
    """
    try:
        df = DATASETS[name].copy()
        rows_before = int(df.shape[0])
        if drop_duplicates:
            df = df.drop_duplicates()
        if strip_whitespace:
            for c in df.select_dtypes(include=["object"]).columns:
                df[c] = df[c].astype(str).str.strip()
        if parse_dates:
            for c in parse_dates:
                if c in df.columns:
                    df[c] = pd.to_datetime(df[c], errors="coerce")
        out_name = target or name
        DATASETS[out_name] = df
        return json.dumps(
            {
                "name": out_name,
                "rows_before": rows_before,
                "rows_after": int(df.shape[0]),
            }
        )
    except Exception as e:
        return f"Error cleaning: {e}"


@agent.tool_plain
def filter_rows(name: str, query: str, target: Optional[str] = None) -> str:
    """
    Filter DATASETS[name] using pandas.query syntax, e.g., "country == 'IN' and revenue > 1000".
    Stores to DATASETS[target or name]; returns {"rows": ...}.
    """
    try:
        df = DATASETS[name]
        out = df.query(query, engine="python")
        out_name = target or name
        DATASETS[out_name] = out
        return json.dumps({"name": out_name, "rows": int(out.shape[0])})
    except Exception as e:
        return f"Error filtering: {e}"


@agent.tool_plain
def select_columns(name: str, columns: List[str], target: Optional[str] = None) -> str:
    """
    Select columns from DATASETS[name] in the given order.
    Stores to DATASETS[target or name]; returns {"columns": [...]}.
    """
    try:
        df = DATASETS[name][columns].copy()
        out_name = target or name
        DATASETS[out_name] = df
        return json.dumps({"name": out_name, "columns": columns})
    except Exception as e:
        return f"Error selecting columns: {e}"


@agent.tool_plain
def aggregate(
    name: str, by: List[str], agg: Dict[str, str], target: Optional[str] = None
) -> str:
    """
    Group DATASETS[name] by the 'by' columns and aggregate numeric columns per 'agg' dict,
    e.g., agg={"revenue":"sum","quantity":"mean"}.
    Stores to DATASETS[target or name]; returns shape summary.
    """
    try:
        df = DATASETS[name].groupby(by, dropna=False).agg(agg).reset_index()
        out_name = target or name
        DATASETS[out_name] = df
        return json.dumps(
            {"name": out_name, "rows": int(df.shape[0]), "cols": int(df.shape[1])}
        )
    except Exception as e:
        return f"Error aggregating: {e}"


@agent.tool_plain
def join(
    left: str,
    right: str,
    on: List[str],
    how: str = "inner",
    target: Optional[str] = None,
) -> str:
    """
    Join DATASETS[left] with DATASETS[right] on columns 'on'. how in ["inner","left","right","outer"].
    Stores to DATASETS[target or left]; returns shape summary.
    """
    try:
        df = DATASETS[left].merge(DATASETS[right], on=on, how=how)
        out_name = target or left
        DATASETS[out_name] = df
        return json.dumps(
            {"name": out_name, "rows": int(df.shape[0]), "cols": int(df.shape[1])}
        )
    except Exception as e:
        return f"Error joining: {e}"


@agent.tool_plain
def resample_time(
    name: str,
    date_col: str,
    freq: str,
    agg: Dict[str, str],
    target: Optional[str] = None,
) -> str:
    """
    Resample a time series: ensure DATASETS[name][date_col] is datetime, set index, resample by 'freq' (e.g., 'D','W','M'),
    aggregate per 'agg' dict. Stores to DATASETS[target or name].
    """
    try:
        df = DATASETS[name].copy()
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        out = df.set_index(date_col).resample(freq).agg(agg).reset_index()
        out_name = target or name
        DATASETS[out_name] = out
        return json.dumps(
            {"name": out_name, "rows": int(out.shape[0]), "cols": int(out.shape[1])}
        )
    except Exception as e:
        return f"Error resampling: {e}"


@agent.tool_plain
def corr(name: str, columns: Optional[List[str]] = None) -> str:
    """
    Pearson correlation matrix for DATASETS[name]. If columns is provided, use only those.
    Returns a JSON object: {"corr": {col_i: {col_j: value}}}.
    """
    try:
        df = DATASETS[name]
        if columns:
            df = df[columns]
        c = df.corr(numeric_only=True)
        return json.dumps({"corr": json.loads(c.to_json())})
    except Exception as e:
        return f"Error computing correlation: {e}"


@agent.tool_plain
def linreg(name: str, y: str, X: List[str]) -> str:
    """
    Simple OLS using numpy lstsq on DATASETS[name]: y ~ X.
    Returns JSON with coefficients (including intercept) and R2.
    """
    try:
        df = DATASETS[name].dropna(subset=[y] + X)
        Y = df[y].to_numpy()
        Xmat = np.column_stack([np.ones(len(df))] + [df[c].to_numpy() for c in X])
        coef, *_ = np.linalg.lstsq(Xmat, Y, rcond=None)
        yhat = Xmat @ coef
        ss_res = float(((Y - yhat) ** 2).sum())
        ss_tot = float(((Y - Y.mean()) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot else 0.0
        return json.dumps(
            {
                "intercept": coef[0],
                "coefficients": {X[i]: coef[i + 1] for i in range(len(X))},
                "r2": r2,
            }
        )
    except Exception as e:
        return f"Error in linreg: {e}"


@agent.tool_plain
def plot_line(
    name: str,
    x: str,
    y: List[str],
    title: Optional[str] = None,
    outfile: Optional[str] = None,
) -> str:
    """
    Line plot from DATASETS[name]. x is column for X-axis; y is a list of Y columns.
    Saves PNG to outfile or 'chart_line.png'; returns the file path.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        df = DATASETS[name]
        outfile = outfile or "chart_line.png"
        plt.figure()
        for col in y:
            plt.plot(df[x], df[col], label=col)
        plt.legend()
        if title:
            plt.title(title)
        plt.xlabel(x)
        plt.ylabel(", ".join(y))
        plt.tight_layout()
        plt.savefig(outfile, dpi=144)
        plt.close()
        return outfile
    except Exception as e:
        return f"Error plotting line: {e}"


@agent.tool_plain
def plot_bar(
    name: str,
    x: str,
    y: str,
    title: Optional[str] = None,
    outfile: Optional[str] = None,
) -> str:
    """
    Bar chart from DATASETS[name], one Y series vs categorical X.
    Saves PNG to outfile or 'chart_bar.png'; returns the file path.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        df = DATASETS[name]
        outfile = outfile or "chart_bar.png"
        plt.figure()
        plt.bar(df[x], df[y])
        if title:
            plt.title(title)
        plt.xlabel(x)
        plt.ylabel(y)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(outfile, dpi=144)
        plt.close()
        return outfile
    except Exception as e:
        return f"Error plotting bar: {e}"


@agent.tool_plain
def save_csv(name: str, path: str) -> str:
    """
    Save DATASETS[name] to a CSV file at 'path'. Returns the absolute path.
    """
    try:
        DATASETS[name].to_csv(path, index=False)
        return os.path.abspath(path)
    except Exception as e:
        return f"Error saving CSV: {e}"


# --- Network helpers (for 'network' questions) ---
@agent.tool_plain
def load_edges(
    name: str,
    source: str,
    src_col: str = "source",
    dst_col: str = "target",
    sep: str = ",",
) -> str:
    """
    Load an edge list (CSV/TSV) with columns src_col, dst_col into DATASETS[name]; also returns basic counts.
    """
    try:
        df = pd.read_csv(source, sep=sep)
        DATASETS[name] = df
        return json.dumps(
            {
                "name": name,
                "rows": int(df.shape[0]),
                "cols": int(df.shape[1]),
                "columns": list(df.columns),
            }
        )
    except Exception as e:
        return f"Error loading edges: {e}"


@agent.tool_plain
def network_stats(
    name: str, src_col: str = "source", dst_col: str = "target", top_k: int = 10
) -> str:
    """
    Compute simple network stats from DATASETS[name] edge list: node_count, edge_count, degree per node (top_k).
    Returns JSON with degree rankings.
    """
    try:
        df = DATASETS[name]
        deg = pd.concat(
            [df[src_col].rename("node"), df[dst_col].rename("node")]
        ).value_counts()
        stats = {
            "node_count": int(deg.shape[0]),
            "edge_count": int(df.shape[0]),
            "top_degree": [
                {"node": str(n), "degree": int(d)} for n, d in deg.head(top_k).items()
            ],
        }
        return json.dumps(stats)
    except Exception as e:
        return f"Error computing network stats: {e}"


def main():
    """Main function to run the interactive AI agent."""
    print("AI Agent is ready. Type 'exit' to quit.")
    while True:
        try:
            prompt = input("You: ")
            if prompt.lower() == "exit":
                break

            # Use the agent to process the prompt
            response = agent.run_sync(prompt)
            print(f"AI: {response.output}")

        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break
        except Exception as e:
            print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
