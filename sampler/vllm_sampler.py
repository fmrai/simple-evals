import time
from typing import Any
from openai import OpenAI

from eval_types import MessageList, SamplerBase

SYSTEM_MESSAGE = "You are a helpful assistant."


class VLLMSampler(SamplerBase):
    """
    Sample from a VLLM server using the OpenAI client
    """

    def __init__(
        self,
        server_url: str = "http://localhost:8000/v1",
        model: str = None,
        system_message: str | None = SYSTEM_MESSAGE,
        temperature: float = 0.5,
        max_tokens: int = 2048,
    ):
        # Create OpenAI client with custom base URL pointing to VLLM server
        self.client = OpenAI(base_url=server_url)
        self.model = (
            model  # This is optional as the server already has the model loaded
        )
        self.system_message = system_message
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _pack_message(self, role: str, content: Any):
        return {"role": str(role), "content": content}

    def __call__(
        self, message_list: MessageList, max_tokens: int | None = None
    ) -> str | tuple[str, list]:
        if self.system_message:
            message_list = [
                self._pack_message("system", self.system_message)
            ] + message_list
        
        if max_tokens is None:
            max_tokens = self.max_tokens

        trial = 0
        while True:
            try:
                # Use the OpenAI client to make the request
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=message_list,
                    temperature=self.temperature,
                    max_tokens=max_tokens,
                    logprobs=True,
                )
                message = response.choices[0].message
                return message.content

            except Exception as e:
                exception_backoff = 2**trial  # exponential back off
                print(
                    f"Request error, retrying {trial} after {exception_backoff} sec",
                    e,
                )
                time.sleep(exception_backoff)
                trial += 1

                # If we've tried too many times, return empty string
                if trial > 5:
                    raise Exception("Too many retries")
