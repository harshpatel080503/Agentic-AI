from phi.agent import Agent
from phi.model.google.gemini import Gemini
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
import google.generativeai as genai
from dotenv import load_dotenv
import os
load_dotenv()

# Configuring Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

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

multi_ai_agent = Agent(
    team = [websearch_agent, finance_agent],
    model = Gemini(id="gemini-2.0-flash-exp"),
    instructions = ["Always include sources", "Use table to display the data"],
    show_tool_calls = True,
    markdown = True
)

multi_ai_agent.print_response("Summarize analyst recommendation and share the latest news for NVDA",stream = True)