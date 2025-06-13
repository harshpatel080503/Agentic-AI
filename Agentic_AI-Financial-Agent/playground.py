from phi.agent import Agent
from phi.model.google.gemini import Gemini
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
import google.generativeai as genai
from dotenv import load_dotenv
import os
import phi
from fastapi.middleware.cors import CORSMiddleware
from phi.playground import Playground, serve_playground_app

# Load environment variables from .env files
load_dotenv()

# Configuring Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

phi.api = os.getenv("PHI_API_KEY")

# Web Search Agent
websearch_agent = Agent(
    name = "Web Search Agent",
    role = "Search the web for the information",
    model = Gemini(id="gemini-2.0-flash-exp"),
    tools = [DuckDuckGo()],
    instructions = ["Always include sources"],
    show_tool_calls = True,
    markdown = True
)

# Financial Agent
finance_agent = Agent(
    name = "Finance AI Agent",
    model = Gemini(id="gemini-2.0-flash-exp"),
    tools = [YFinanceTools(stock_price=True,analyst_recommendations=True,stock_fundamentals=True,company_news=True)],
    instructions = ["Use Tables to Display the data"],
    show_tool_calls = True,
    markdown = True
)

app = Playground(agents=[finance_agent,websearch_agent]).get_app()

# Allow CORS for the playground frontend

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    serve_playground_app("playground:app",reload=True)