from langchain.chat_models import init_chat_model
from healthcare_bot.config import settings


llm = init_chat_model(
    model=settings.llm,
    model_provider=settings.model_provider,
    api_key=settings.api_key,
)
