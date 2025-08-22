from typing import Dict, Any, List, Tuple, Optional
import requests
from datetime import datetime
from dataclasses import dataclass
from collections import Counter
from openai import AzureOpenAI
import matplotlib.pyplot as plt
from io import BytesIO
import base64

from langgraph.graph import Graph, StateGraph
from langchain.tools import Tool
from langchain_core.messages import BaseMessage, HumanMessage
from langchain.tools.render import format_tool_to_openai_function
from pydantic import BaseModel
import operator

# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT = "https://alpha10x-open-ai-probe-sweden.openai.azure.com/"
OPENAI_API_VERSION = "2024-10-21"
AZURE_DEPLOYMENT_NAME = "gpt-4o"
AZURE_API_KEY = "9e002ed781954a55a6cf4865483fc396"

# Initialize Azure OpenAI client
try:
    client = AzureOpenAI(
        api_key=AZURE_API_KEY,
        api_version=OPENAI_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT
    )
except Exception as e:
    raise Exception(f"Failed to initialize Azure OpenAI client: {str(e)}")

@dataclass
class ResearchStats:
    total_papers: int
    top_papers: List[Dict]
    yearly_trends: Dict[str, int]
    top_institutions: List[Tuple[str, int]]
    top_authors: List[Tuple[str, int]]
    charts: Dict[str, str] = None  # Will store base64 encoded chart images

class State(BaseModel):
    concept: str
    api_response: Optional[Dict[str, Any]] = None
    stats: Optional[ResearchStats] = None
    result: Optional[str] = None
    investment_insights: Optional[str] = None
    charts: Optional[Dict[str, str]] = None

def query_openalex(concept: str) -> Dict[str, Any]:
    """Query OpenAlex API for research papers on a given concept."""
    base_url = "https://api.openalex.org/works"
    params = {
        "search": concept,
        "per_page": 50,  # Get more results for better analysis
        "sort": "cited_by_count:desc"
    }
    
    response = requests.get(base_url, params=params)
    if response.status_code != 200:
        raise Exception(f"OpenAlex API request failed with status {response.status_code}")
    
    return response.json()

def analyze_research_data(api_response: Dict[str, Any]) -> ResearchStats:
    """Analyze the research data from OpenAlex."""
    results = api_response.get("results", [])
    total_papers = api_response.get("meta", {}).get("count", 0)
    
    # Get top papers (already sorted by citation count)
    top_papers = [
        {
            "title": paper.get("title", ""),
            "citations": paper.get("cited_by_count", 0),
            "year": paper.get("publication_year", ""),
            "doi": paper.get("doi", "")
        }
        for paper in results[:5]  # Top 5 papers
    ]
    
    # Analyze yearly trends
    years = [paper.get("publication_year") for paper in results if paper.get("publication_year")]
    yearly_trends = Counter(years)
    
    # Extract institutions and authors
    institutions = []
    authors = []
    for paper in results:
        for inst in paper.get("authorships", []):
            if inst.get("institutions"):
                institutions.extend([i.get("display_name") for i in inst.get("institutions", [])])
        for author in paper.get("authorships", []):
            if author.get("author"):
                authors.append(author["author"].get("display_name"))
    
    top_institutions = Counter(institutions).most_common(5)
    top_authors = Counter(authors).most_common(5)
    
    stats = ResearchStats(
        total_papers=total_papers,
        top_papers=top_papers,
        yearly_trends=dict(sorted(yearly_trends.items())),
        top_institutions=top_institutions,
        top_authors=top_authors
    )
    
    # Generate charts
    stats.charts = create_visualization_charts(stats)
    
    return stats

def format_insights(stats: ResearchStats) -> str:
    """Format the research insights into a readable message."""
    message = [
        "📊 Market Research Insights",
        f"\n🔍 Found {stats.total_papers:,} academic papers on this topic",
        
        "\n📈 Research Interest:",
        "Year | Number of Papers",
        "-----|----------------"
    ]
    
    # Add yearly trend
    for year, count in stats.yearly_trends.items():
        message.append(f"{year} | {count}")
    
    message.extend([
        "\n🏆 Top Cited Papers:",
        "-------------------"
    ])
    
    for paper in stats.top_papers:
        message.append(
            f"• {paper['title']}\n"
            f"  Citations: {paper['citations']:,} | Year: {paper['year']}\n"
            f"  DOI: {paper['doi']}"
        )
    
    message.extend([
        "\n🏛️ Leading Institutions:",
        "--------------------"
    ])
    for inst, count in stats.top_institutions:
        message.append(f"• {inst} ({count} papers)")
    
    message.extend([
        "\n👨‍🔬 Top Researchers:",
        "----------------"
    ])
    for author, count in stats.top_authors:
        message.append(f"• {author} ({count} papers)")

    return "\n".join(message)

def generate_investment_insights(research_data: str, concept: str) -> str:
    """Generate investment insights using Azure OpenAI GPT-4."""
    prompt = f"""As an investment analyst, analyze the following research landscape data for {concept} and provide strategic investment insights. 
    Focus on:
    1. Market maturity and growth potential
    2. Key players and institutional involvement
    3. Research momentum and emerging trends
    4. Potential investment opportunities and risks
    5. Recommendations for investors

    Research Data:
    {research_data}

    Provide a concise, actionable analysis for investors considering this space.
    """
    
    try:
        response = client.chat.completions.create(
            model=AZURE_DEPLOYMENT_NAME,  # Using the gpt-4o deployment
            messages=[
                {"role": "system", "content": "You are an expert investment analyst specializing in emerging technology markets."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Detailed error: {str(e)}")  # Add detailed error logging
        raise Exception(f"Azure OpenAI API call failed: {str(e)}")

def create_visualization_charts(stats: ResearchStats) -> Dict[str, str]:
    """Create visualization charts and return them as base64 encoded strings."""
    charts = {}
    
    # Set style
    plt.style.use('default')  # Changed from 'seaborn' to 'default'
    
    # 1. Research Trend Over Time
    plt.figure(figsize=(12, 6))
    years = list(stats.yearly_trends.keys())
    papers = list(stats.yearly_trends.values())
    plt.plot(years, papers, marker='o', linewidth=2, markersize=8)
    plt.title('Research Trend Over Time', fontsize=14, pad=20)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Number of Papers', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    
    # Save to base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', dpi=300)
    buffer.seek(0)
    charts['trend'] = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    
    # 2. Top Institutions Bar Chart
    plt.figure(figsize=(12, 6))
    institutions, counts = zip(*stats.top_institutions)
    plt.barh(range(len(institutions)), counts)
    plt.yticks(range(len(institutions)), institutions)
    plt.title('Top Research Institutions', fontsize=14, pad=20)
    plt.xlabel('Number of Papers', fontsize=12)
    plt.ylabel('Institution', fontsize=12)
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', dpi=300)
    buffer.seek(0)
    charts['institutions'] = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    
    # 3. Top Authors Bar Chart
    plt.figure(figsize=(12, 6))
    authors, author_counts = zip(*stats.top_authors)
    plt.barh(range(len(authors)), author_counts)
    plt.yticks(range(len(authors)), authors)
    plt.title('Top Authors in the Field', fontsize=14, pad=20)
    plt.xlabel('Number of Papers', fontsize=12)
    plt.ylabel('Author', fontsize=12)
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', dpi=300)
    buffer.seek(0)
    charts['authors'] = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    
    return charts

class ResearchBot:
    def __init__(self):
        # Define tools as functions
        self.tools = {
            "query_openalex": query_openalex,
            "analyze_research": analyze_research_data,
            "format_insights": format_insights,
            "generate_investment_insights": generate_investment_insights
        }
        
        # Build the graph
        self.workflow = self.build_graph()
        self.current_state = None  # Add state tracking

    def build_graph(self) -> StateGraph:
        # Create the graph with state schema
        workflow = StateGraph(state_schema=State)

        # Define nodes
        def query_node(state: State) -> Dict:
            api_response = self.tools["query_openalex"](state.concept)
            return {"api_response": api_response}

        def analyze_node(state: State) -> Dict:
            stats = self.tools["analyze_research"](state.api_response)
            return {"stats": stats, "charts": stats.charts}

        def format_node(state: State) -> Dict:
            result = self.tools["format_insights"](state.stats)
            return {"result": result}

        def insights_node(state: State) -> Dict:
            insights = self.tools["generate_investment_insights"](state.result, state.concept)
            return {"investment_insights": insights}

        # Add nodes
        workflow.add_node("query", query_node)
        workflow.add_node("analyze", analyze_node)
        workflow.add_node("format", format_node)
        workflow.add_node("insights", insights_node)

        # Add edges
        workflow.add_edge("query", "analyze")
        workflow.add_edge("analyze", "format")
        workflow.add_edge("format", "insights")

        # Set entry point
        workflow.set_entry_point("query")

        # Set end state
        workflow.set_finish_point("insights")

        return workflow.compile()
    
    def run(self, concept: str) -> Tuple[str, str, Optional[Dict[str, str]]]:
        try:
            # Execute workflow
            result = self.workflow.invoke({"concept": concept})
            self.current_state = result  # Store the current state
            return result["result"], result["investment_insights"], result.get("charts")
        except Exception as e:
            return f"Error: {str(e)}", "", None

def main():
    bot = ResearchBot()
    
    while True:
        concept = input("\nEnter a concept to research (or 'quit' to exit): ")
        if concept.lower() == 'quit':
            break
        
        research_data, investment_insights, charts = bot.run(concept)
        
        print("\n=== RESEARCH DATA ===")
        print(research_data)
        print("\n=== INVESTMENT INSIGHTS ===")
        print(investment_insights)
        
        # Save and display charts
        if charts:
            print("\n=== VISUALIZATION CHARTS ===")
            print("Charts have been generated and saved as:")
            
            # Save charts to files
            for chart_name, chart_data in charts.items():
                filename = f"{concept.replace(' ', '_')}_{chart_name}_chart.png"
                with open(filename, "wb") as f:
                    f.write(base64.b64decode(chart_data))
                print(f"- {filename}")
            
            print("\nYou can find the visualization charts in the current directory.")

if __name__ == "__main__":
    main() 