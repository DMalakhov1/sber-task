"""Named DeepSeek client using the shared OpenAI-compatible transport."""
from task2_bot.bot.openai_client import APIError, OpenAIClient

class DeepSeekClient(OpenAIClient):
    def __init__(self, model="deepseek-flash", transport=None):
        super().__init__(model=model, base_url="https://api.deepseek.com", transport=transport)

__all__ = ["APIError", "OpenAIClient", "DeepSeekClient"]
