from typing import Dict, Type, Optional
import requests
from pydantic import BaseModel, Field, validator
from langchain_core.tools import BaseTool

class CurrencyConversionInput(BaseModel):
    """Input for the currency conversion tool."""
    amount: float = Field(description="The amount to convert")
    from_currency: str = Field(description="The source currency code (e.g., 'USD', 'EUR')")
    to_currency: str = Field(description="The target currency code (e.g., 'EUR', 'JPY')")
    
    @validator('from_currency', 'to_currency')
    def validate_currency_code(cls, v):
        if not v.isalpha() or len(v) != 3:
            raise ValueError('Currency code must be a 3-letter code (e.g., USD, EUR, JPY)')
        return v.upper()

class CurrencyConversionTool(BaseTool):
    """Tool for converting between currencies using real-time exchange rates."""
    name: str = "convert_currency"
    description: str = (
        "Convert an amount from one currency to another. "
        "Input should be a JSON object with 'amount' (number), 'from_currency' (3-letter code), "
        "and 'to_currency' (3-letter code) fields."
    )
    args_schema: Type[BaseModel] = CurrencyConversionInput
    
    def _run(
        self, 
        amount: float, 
        from_currency: str, 
        to_currency: str,
        **kwargs
    ) -> str:
        """Convert currency using the latest exchange rates.
        
        Args:
            amount: The amount to convert
            from_currency: The source currency code (e.g., 'USD', 'EUR')
            to_currency: The target currency code (e.g., 'EUR', 'JPY')
            
        Returns:
            str: The converted amount and exchange rate information
        """
        try:
            # Use the Frankfurter API which doesn't require an API key
            base_url = f"https://api.frankfurter.app/latest"
            params = {
                "from": from_currency.upper(),
                "to": to_currency.upper(),
                "amount": amount
            }
            
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if 'error' in data:
                return f"Error: {data.get('error', 'Failed to fetch exchange rates')}"
            
            # Get the converted amount and rate from the response
            converted_amount = data['rates'].get(to_currency.upper())
            if converted_amount is None:
                return f"Error: Could not find exchange rate for {to_currency}"
            
            # Calculate the actual rate (amount might be different if input was rounded)
            rate = converted_amount / amount
            
            return (
                f"{amount} {from_currency.upper()} = {converted_amount:.2f} {to_currency.upper()}\n"
                f"Exchange rate: 1 {from_currency.upper()} = {rate:.6f} {to_currency.upper()}"
            )
            
        except requests.exceptions.RequestException as e:
            return f"Error fetching exchange rates: {str(e)}"
        except Exception as e:
            return f"Error performing currency conversion: {str(e)}"
