import asyncio
import python_weather
from typing import Dict, Any, Type, Optional
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool

class WeatherInput(BaseModel):
    location: str = Field(description="The location to get weather for (e.g., 'San Francisco, CA')")

class WeatherTool(BaseTool):
    name = "get_weather"
    description = "Get current weather information for a location"
    args_schema: Type[BaseModel] = WeatherInput
    
    async def _aget_weather(self, location: str) -> str:
        """
        Get current weather for a given location.
        
        Args:
            location: The location to get weather for
            
        Returns:
            str: Formatted weather information
        """
        try:
            async with python_weather.Client() as client:
                weather = await client.get(location)
                
                weather_info = (
                    f"Weather in {location}:\n"
                    f"- Temperature: {weather.current.temperature}°C (Feels like: {weather.current.feels_like}°C)\n"
                    f"- Conditions: {weather.current.description}\n"
                    f"- Humidity: {weather.current.humidity}%\n"
                    f"- Wind: {weather.current.wind_speed} km/h, {weather.current.wind_direction}\n"
                )
                
                return weather_info
                
        except Exception as e:
            return f"Error getting weather: {str(e)}"
    
    def _run(self, location: str) -> str:
        """
        Synchronous wrapper for the async weather function.
        """
        return asyncio.run(self._aget_weather(location))
