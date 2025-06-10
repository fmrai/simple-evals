import time
from typing import Any
from openai import OpenAI

from eval_types import MessageList, SamplerBase

SYSTEM_MESSAGE = "You are a helpful assistant."


class VLLMSampler(SamplerBase):
    """
    Sample from a VLLM server using the OpenAI client
    Supports both chat and base (completion) models.
    """

    def __init__(
        self,
        server_url: str = "http://localhost:8000/v1",
        model: str = None,
        system_message: str | None = SYSTEM_MESSAGE,
        temperature: float = 0.5,
        max_tokens: int = 2048,
        mode: str = "chat",  # 'chat' or 'base'
    ):
        # Create OpenAI client with custom base URL pointing to VLLM server
        self.client = OpenAI(base_url=server_url)
        self.model = model  # This is optional as the server already has the model loaded
        self.system_message = system_message
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.mode = mode

    def _pack_message(self, role: str, content: Any):
        return {"role": str(role), "content": content}
    
    def _handle_chat(self, input_data: MessageList, max_tokens: int | None = None) -> str:
        message_list = input_data
        if self.system_message:
            message_list = [
                self._pack_message("system", self.system_message)
            ] + message_list
        response = self.client.chat.completions.create(
            model=self.model,
            messages=message_list,
            temperature=self.temperature,
            max_tokens=max_tokens,
            logprobs=True,
        )
        message = response.choices[0].message
        return message.content
    
    def _handle_base(self, input_data: str, max_tokens: int | None = None) -> str:
        response = self.client.completions.create(
            model=self.model,
            prompt=input_data,
            temperature=self.temperature,
            max_tokens=max_tokens,
            logprobs=True,
        )
        message = response.choices[0].text
        return message

    def __call__(
        self, input_data: MessageList | str, max_tokens: int | None = None
    ) -> str | tuple[str, list]:
        if max_tokens is None:
            max_tokens = self.max_tokens

        trial = 0
        while True:
            try:
                if self.mode == "chat":
                    return self._handle_chat(input_data, max_tokens)
                elif self.mode == "base":
                    return self._handle_base(input_data, max_tokens)
                else:
                    raise ValueError(f"Unknown mode: {self.mode}")

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
