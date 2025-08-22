# ResearchVest AI

*AI-Powered Market Research & Investment Intelligence Platform*

A market research tool powered by LangGraph and Azure OpenAI that helps investors research market trends and scientific backing on concepts in seconds, complete with data visualization.

## Overview

This tool enables investors to quickly gauge academic interest and research trends in emerging technologies and investment concepts. By leveraging the OpenAlex API, Azure OpenAI, and LangGraph's workflow orchestration, it provides comprehensive insights into:

- Volume of academic research on a given topic
- Research trends over time with visual graphs
- Key influential papers and authors
- Top institutions studying the concept
- Strategic investment insights powered by GPT-4
- Interactive data visualizations
## Features

- **Research Data Analysis**
  - Comprehensive paper statistics
  - Citation analysis
  - Temporal trend analysis
  - Institutional contribution metrics
  - Author impact assessment

- **Data Visualization**
  - Research trend timeline charts
  - Top institutions bar charts
  - Leading authors visualization
  - High-resolution PNG exports

- **Investment Insights**
  - Market maturity assessment
  - Growth potential analysis
  - Key player identification
  - Research momentum evaluation
  - Investment recommendations

## Architecture

The application is built using LangGraph and Azure OpenAI, structured as a directed graph with four main nodes:

1. **Query Node**: Interfaces with OpenAlex API to fetch relevant research papers
2. **Analysis Node**: Processes raw data and generates visualizations
3. **Format Node**: Structures the research insights into a readable format
4. **Insights Node**: Generates investment analysis using Azure OpenAI GPT-4

### State Management

The application uses Pydantic models for robust state management:
```python
class State(BaseModel):
    concept: str
    api_response: Optional[Dict[str, Any]] = None
    stats: Optional[ResearchStats] = None
    result: Optional[str] = None
    investment_insights: Optional[str] = None
    charts: Optional[Dict[str, str]] = None
```

## Installation

1. Clone this repository
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

Required dependencies:
- langgraph>=0.0.15
- langchain>=0.1.0
- requests>=2.31.0
- openai>=1.12.0
- matplotlib>=3.7.2
- python-dotenv>=1.0.0

## Configuration

Set up your Azure OpenAI credentials in the environment:
```python
AZURE_OPENAI_ENDPOINT = "your_endpoint"
AZURE_API_KEY = "your_api_key"
OPENAI_API_VERSION = "2024-10-21"
AZURE_DEPLOYMENT_NAME = "your_deployment_name"
```

## Usage

Run the main script:
```bash
python investor_intel_bot.py
```

Then interact with the bot by entering investment concepts you'd like to research, such as:
- "Quantum computing"
- "Green hydrogen"
- "AI in healthcare"

### Example Output

```
📊 Market Research Insights
🔍 Found 1,234 academic papers on this topic

📈 Research Interest:
Year | Number of Papers
-----|----------------
2023 | 45
2022 | 38
...

🏆 Top Cited Papers:
-------------------
• [Paper Title]
  Citations: 1,234 | Year: 2022
  DOI: [DOI Link]
...

🏛️ Leading Institutions:
--------------------
• Stanford University (15 papers)
• MIT (12 papers)
...

👨‍🔬 Top Researchers:
----------------
• John Doe (8 papers)
• Jane Smith (6 papers)
...
```

### Generated Visualizations

The tool automatically generates three visualization charts for each analysis:
- `{concept}_trend_chart.png`: Research publication trends over time
- `{concept}_institutions_chart.png`: Top contributing institutions
- `{concept}_authors_chart.png`: Leading researchers in the field

## How It Works

1. User inputs a concept
2. LangGraph orchestrates the workflow:
   - Queries OpenAlex API for relevant academic papers
   - Analyzes the data to extract trends and insights
   - Generates visual charts using matplotlib
   - Uses Azure OpenAI GPT-4 to generate investment insights
3. Results are presented with:
   - Research volume metrics
   - Temporal trends
   - Key papers and citations
   - Leading institutions and researchers
   - Visual charts
   - Strategic investment analysis

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT
