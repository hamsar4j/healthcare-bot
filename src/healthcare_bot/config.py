from dotenv import load_dotenv
import os

load_dotenv()


class Settings:
    llm: str = os.getenv("LLM", "")
    api_key: str = os.getenv("OPENAI_API_KEY", "")
    model_provider: str = os.getenv("MODEL_PROVIDER", "together")


settings = Settings()
