# Data Analyst Agent

A powerful API that uses Google's Gemini 2.5 Pro to source, prepare, analyze, and visualize any data. This agent can handle complex data analysis tasks including web scraping, statistical analysis, and data visualization.

## Features

- 🤖 **AI-Powered Analysis**: Uses Gemini 2.5 Pro for intelligent data interpretation
- 📊 **Data Visualization**: Creates charts, plots, and statistical visualizations
- 🌐 **Web Scraping**: Can scrape data from Wikipedia and other sources
- 📁 **Multi-Format Support**: Handles CSV, JSON, images, and text files
- 🗄️ **Database Integration**: Supports DuckDB for large dataset queries
- ⚡ **Fast API**: Built with FastAPI for high performance

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd tds-proj-2

# Install dependencies using uv (recommended)
uv sync

# Or with pip
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Copy the example environment file
cp env.example .env

# Edit .env and add your Gemini API key
GEMINI_API_KEY=your_gemini_api_key_here
```

Get your Gemini API key from: https://aistudio.google.com/app/apikey

### 3. Run the Application

```bash
# Start the server with uv
uv run python main.py

# Or using uvicorn directly
uv run uvicorn main:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

## API Usage

### Endpoint

```
POST /api/
```

### Request Format

Send a multipart form request with:
- `questions.txt`: A text file containing your analysis questions (required)
- Additional files: CSV, JSON, images, or other data files (optional)

### Example Usage

```bash
# Basic analysis with questions only
curl -X POST "http://localhost:8000/api/" \
  -F "questions.txt=@questions.txt"

# Analysis with data files
curl -X POST "http://localhost:8000/api/" \
  -F "questions.txt=@questions.txt" \
  -F "data.csv=@data.csv" \
  -F "image.png=@image.png"
```

### Sample Questions

The agent can handle various types of analysis:

**Web Scraping Analysis:**
```
Scrape the list of highest grossing films from Wikipedia at:
https://en.wikipedia.org/wiki/List_of_highest-grossing_films

Answer these questions:
1. How many $2 bn movies were released before 2000?
2. Which is the earliest film that grossed over $1.5 bn?
3. What's the correlation between Rank and Peak?
4. Draw a scatterplot of Rank and Peak with a dotted red regression line.
```

**Database Analysis:**
```
Query the Indian high court dataset and answer:
1. Which high court disposed the most cases from 2019-2022?
2. What's the regression slope of registration to decision date by year?
3. Plot the delay trends with a regression line.
```

### Response Format

Responses are returned as JSON arrays or objects depending on the request:

```json
[1, "Titanic", 0.485782, "data:image/png;base64,iVBORw0KG..."]
```

or

```json
{
  "question1": "answer1",
  "question2": "answer2",
  "plot": "data:image/png;base64,iVBORw0KG..."
}
```

## Deployment

The application is designed to be easily deployable to various platforms:

### Render (Free Tier)

1. Fork or clone this repository to your GitHub account
2. Go to [Render Dashboard](https://dashboard.render.com)
3. Click "New" → "Web Service"
4. Connect your GitHub repository
5. Use these settings:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn -c gunicorn.conf.py main:app`
   - **Environment**: Python 3
6. Add environment variable: `GEMINI_API_KEY` = your_gemini_api_key
7. Deploy!

### Other Free Options

**PythonAnywhere**: Upload files and configure WSGI
**Heroku**: Use Heroku CLI or GitHub integration
**Google App Engine**: Use gcloud CLI for deployment
**Koyeb**: Connect GitHub repository with automatic deployments

### Railway

1. Connect your GitHub repository to Railway
2. Set the `GEMINI_API_KEY` environment variable
3. Deploy automatically

### Docker

```bash
# Build the image
docker build -t data-analyst-agent .

# Run the container
docker run -p 8000:8000 -e GEMINI_API_KEY=your_key data-analyst-agent
```

## Development

### Project Structure

```
tds-proj-2/
├── main.py              # Main FastAPI application
├── requirements.txt     # Python dependencies
├── README.md           # This file
├── env.example         # Environment variable template
└── LICENSE            # MIT License
```

### Key Components

- **DataAnalystAgent**: Core class handling analysis logic
- **Gemini Integration**: Uses Google's Gemini 2.5 Pro for reasoning
- **Data Processing**: Handles CSV, JSON, and web scraping
- **Visualization**: Creates charts with matplotlib/seaborn
- **Database Support**: DuckDB integration for large datasets

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues and questions:
1. Check the existing issues on GitHub
2. Create a new issue with detailed information
3. Include sample requests and expected outputs

## API Reference

### Health Check

```
GET /
GET /health
```

Returns API status and health information.

### Data Analysis

```
POST /api/
```

**Parameters:**
- `questions.txt` (required): Text file with analysis questions
- Additional files (optional): Data files to analyze

**Response:**
- JSON array or object with analysis results
- Base64-encoded images for visualizations
- Error messages with HTTP status codes

**Timeout:** 3 minutes maximum processing time
