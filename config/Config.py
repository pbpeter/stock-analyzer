import os

from dotenv import load_dotenv

load_dotenv()


api_key=os.environ.get("OPENAI_API_KEY", None)
base_url=os.environ.get("BASE_URL", None)

assistant_name = os.environ.get("ASSISTANT_NAME", None)
assistant_instruction = os.environ.get("ASSISTANT_INSTRUCTION", None)
assistant_model = os.environ.get("ASSISTANT_MODEL", None)

alphavantage_key = os.environ.get("ALPHAVANTAGE_API_KEY", None)
# def init():
#     global api_key
#     global base_url
#     api_key = api_key
#     base_url = base_url


