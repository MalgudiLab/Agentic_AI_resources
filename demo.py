import warnings
import logging
import os
import sys

warnings.filterwarnings("ignore")
logging.getLogger("pydantic").setLevel(logging.ERROR)
os.environ["PYTHONWARNINGS"] = "ignore"

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from duckduckgo_search import DDGS

def get_news_robust(topic: str):
    print(f"\n[1/2] Searching for: {topic} news...")
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(f"{topic} latest news", max_results=5))
            if results:
                print(f"--- Found {len(results)} articles ---")
                return "\n".join([f"- {r.get('title')}: {r.get('body')}" for r in results]), [r.get('title') for r in results]
    except Exception as e:
        print(f"Search engine error: {e}")


# 2. Setup Local LLM
llm = ChatOllama(model="llama3.1", temperature=0.1)

prompt = ChatPromptTemplate.from_template("""
Summarize these news snippets into 3 impactful bullet points for today, March 10, 2026.
NEWS DATA: {news_data}
""")

chain = prompt | llm | StrOutputParser()

if __name__ == "__main__":
    topic = "Bengaluru"
    raw_data, sources = get_news_robust(topic)
    
    print("[2/2] Generating summary...")
    summary = chain.invoke({"news_data": raw_data})
    
    print("\n" + "═"*50)
    print(f" 📰 {topic.upper()} DAILY BRIEF - MARCH 10, 2026 ")
    print("═"*50)
    print(summary)
    print("\n📚 SOURCES REFERENCED:")
    for i, s in enumerate(sources, 1): print(f" {i}. {s}")
    print("═"*50)
