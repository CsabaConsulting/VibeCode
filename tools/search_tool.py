from duckduckgo_search import DDGS
from typing import List, Dict, Any, Type, Optional
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool

class SearchInput(BaseModel):
    query: str = Field(description="The search query")
    max_results: int = Field(default=3, description="Maximum number of results to return")

class SearchTool(BaseTool):
    name: str = "search_web"
    description: str = "Search the web for information using DuckDuckGo"
    args_schema: Type[BaseModel] = SearchInput
    
    def _run(self, query: str, max_results: int = 3) -> str:
        """
        Search the web using DuckDuckGo.
        
        Args:
            query: The search query
            max_results: Maximum number of results to return
            
        Returns:
            str: Formatted search results
        """
        try:
            with DDGS() as ddgs:
                results = []
                for result in ddgs.text(query, max_results=max_results):
                    results.append({
                        'title': result.get('title', ''),
                        'link': result.get('href', ''),
                        'snippet': result.get('body', '')
                    })
                
                if not results:
                    return "No results found."
                    
                # Format results as a string
                formatted_results = []
                for i, result in enumerate(results, 1):
                    formatted_results.append(
                        f"{i}. {result['title']}\n"
                        f"   URL: {result['link']}\n"
                        f"   {result['snippet']}\n"
                    )
                
                return "\n".join(formatted_results)
                
        except Exception as e:
            return f"Error performing search: {str(e)}"
