import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Print the API key to verify it's loaded correctly
api_key = os.getenv('LANGCHAIN_API_KEY')
print(f'Loaded API key: {api_key}')
