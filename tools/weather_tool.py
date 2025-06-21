import asyncio
import python_weather
import os
from typing import Dict, Any, Type, Optional
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool

class WeatherInput(BaseModel):
    location: str = Field(description="The location to get weather for (e.g., 'San Francisco, CA')")

class WeatherTool(BaseTool):
    name: str = "get_weather"
    description: str = "Get current weather information for a location"
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
            # Set event loop policy for Windows if needed
            if os.name == 'nt':
                asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
            
            # Initialize the client with metric units (Celsius, km/h, etc.)
            async with python_weather.Client(unit=python_weather.METRIC) as client:
                # Get the weather for the location
                weather = await client.get(location)
                
                # Format the weather information
                weather_info = [
                    f"Weather in {location}:",
                    f"- Current temperature: {weather.temperature}°C"
                ]
                
                # Add forecast for today if available
                if hasattr(weather, 'forecasts') and weather.forecasts:
                    today = weather.forecasts[0]
                    weather_info.extend([
                        f"- Conditions: {today.sky_text}",
                        f"- High: {today.high}°C, Low: {today.low}°C",
                        f"- Precipitation: {today.precip}%"
                    ])
                
                return '\n'.join(weather_info)
                
        except Exception as e:
            return f"Error getting weather: {str(e)}"
    
    def _run(self, location: str) -> str:
        """
        Synchronous wrapper for the async weather function.
        """
        return asyncio.run(self._aget_weather(location))
