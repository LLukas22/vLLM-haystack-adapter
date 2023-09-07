from haystack.nodes import PromptModelInvocationLayer
from haystack.nodes.prompt.invocation_layer.handlers import TokenStreamingHandler, DefaultTokenStreamingHandler

import sseclient
import os
from typing import Dict, List, Union, Type, Optional, Any, cast
import logging 
import json 
import requests

from tokenizers import Tokenizer
from haystack.errors import OpenAIError
from haystack.utils.openai_utils import (
    openai_request,
    _check_openai_finish_reason,
)


logger = logging.getLogger(__name__)

# based on OpenAIInvocationLayer, with some modifications to the tokenization
class vLLMInvocationLayer(PromptModelInvocationLayer):
    """
    PromptModelInvocationLayer implementation for vLLM's "OpenAI compatible API". Invocations are made using REST API.
    See [vLLM Quickstart](https://vllm.readthedocs.io/en/latest/getting_started/quickstart.html#openai-compatible-server) for more details.

    Note: kwargs other than init parameter names are ignored to enable reflective construction of the class
    as many variants of PromptModelInvocationLayer are possible and they may have different parameters.
    """
   
    def __init__(
        self,
        api_base: str,
        max_length: Optional[int] = 100,
        model_name_or_path: Optional[str] = None,
        tokenizer: Optional[str] = None,
        hf_token:Optional[str] = None,
        maximum_context_length: Optional[int] = None,
        **kwargs,
    ):
        """
        Creates an instance of vLLMInvocationLayer for an hosted vLLM server.

        :param api_base: The base url, used to cummunicate with your vLLM server. E.g. `https://[MY Server]/v1`.
        :param model_name_or_path: The name or path of the underlying model.
        :param max_length: The maximum number of tokens the output text can have.
        :param tokenizer: Optional tokenizer to load from the hub.
        :param hf_token: Optional huggingface token to use for authentication with the hf_hub.
        :param kwargs: Additional keyword arguments passed to the underlying model. Due to reflective construction of
        all PromptModelInvocationLayer instances, this instance of vLLMInvocationLayer might receive some unrelated
        kwargs. Only the kwargs relevant to vLLMInvocationLayer are considered. The list of OpenAI-relevant
        kwargs includes: suffix, temperature, top_p, presence_penalty, frequency_penalty, best_of, n, max_tokens,
        stop, echo, and logprobs. For more details about these kwargs, see OpenAI
        [documentation](https://platform.openai.com/docs/api-reference/completions/create).
        """

        if not isinstance(api_base, str) or len(api_base) == 0:
            raise OpenAIError(
                f"api_base `{api_base}` must be a valid OpenAI-like URL to your vLLM server."
            )
        
        if not isinstance(model_name_or_path, str) or len(model_name_or_path) == 0:
            #Get the hosted model
            model_name_or_path = vLLMInvocationLayer.get_supported_models(api_base)[0]

        super().__init__(model_name_or_path)
        
        self.api_key = "EMPTY" # this is a dummy value
        self.api_base = api_base

        # 16 is the default length for answers from OpenAI shown in the docs
        # here, https://platform.openai.com/docs/api-reference/completions/create.
        # max_length must be set otherwise OpenAIInvocationLayer._ensure_token_limit will fail.
        self.max_length = max_length or 16

        # Due to reflective construction of all invocation layers we might receive some
        # unknown kwargs, so we need to take only the relevant.
        # For more details refer to OpenAI documentation
        self.model_input_kwargs = {
            key: kwargs[key]
            for key in [
                "suffix",
                "max_tokens",
                "temperature",
                "top_p",
                "n",
                "logprobs",
                "echo",
                "stop",
                "presence_penalty",
                "frequency_penalty",
                "best_of",
                "logit_bias",
                "stream",
                "stream_handler",
                "moderate_content",
            ]
            if key in kwargs
        }

        self.tokenizer = Tokenizer.from_pretrained(tokenizer or model_name_or_path,auth_token=hf_token)

        #Infer the context length of the model
        if maximum_context_length:
            self.max_tokens_limit = maximum_context_length
        else:
            # Just default to 2048
            self.max_tokens_limit = 2048

    @property
    def url(self) -> str:
        return f"{self.api_base}/completions"

    @property
    def headers(self) -> Dict[str, str]:
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        return headers

    def invoke(self, *args, **kwargs):
        """
        Invokes a prompt on the model. Based on the model, it takes in a prompt (or either a prompt or a list of messages)
        and returns a list of responses using a REST invocation.

        :return: The responses are being returned.

        Note: Only kwargs relevant to OpenAI are passed to OpenAI rest API. Others kwargs are ignored.
        For more details, see OpenAI [documentation](https://platform.openai.com/docs/api-reference/completions/create).
        """
        prompt = kwargs.get("prompt")
        # either stream is True (will use default handler) or stream_handler is provided
        kwargs_with_defaults = self.model_input_kwargs
        if kwargs:
            # we use keyword stop_words but OpenAI uses stop
            if "stop_words" in kwargs:
                kwargs["stop"] = kwargs.pop("stop_words")
            if "top_k" in kwargs:
                top_k = kwargs.pop("top_k")
                kwargs["n"] = top_k
                kwargs["best_of"] = top_k
            kwargs_with_defaults.update(kwargs)
        stream = (
            kwargs_with_defaults.get("stream", False) or kwargs_with_defaults.get("stream_handler", None) is not None
        )
        base_payload = {  # payload common to all OpenAI models
            "model": self.model_name_or_path,
            "max_tokens": kwargs_with_defaults.get("max_tokens", self.max_length),
            "temperature": kwargs_with_defaults.get("temperature", 0.7),
            "top_p": kwargs_with_defaults.get("top_p", 1),
            "n": kwargs_with_defaults.get("n", 1),
            "stream": stream,
            "stop": kwargs_with_defaults.get("stop", None),
            "presence_penalty": kwargs_with_defaults.get("presence_penalty", 0),
            "frequency_penalty": kwargs_with_defaults.get("frequency_penalty", 0),
        }
        responses = self._execute_openai_request(
            prompt=prompt, base_payload=base_payload, kwargs_with_defaults=kwargs_with_defaults, stream=stream
        )
        return responses

    def _execute_openai_request(self, prompt: str, base_payload: Dict, kwargs_with_defaults: Dict, stream: bool):
        if not prompt:
            raise ValueError(
                f"No prompt provided. Model {self.model_name_or_path} requires prompt."
                f"Make sure to provide prompt in kwargs."
            )
        extra_payload = {
            "prompt": prompt,
            "suffix": kwargs_with_defaults.get("suffix", None),
            "logprobs": kwargs_with_defaults.get("logprobs", None),
            "echo": kwargs_with_defaults.get("echo", False),
            "best_of": kwargs_with_defaults.get("best_of", 1),
        }
        payload = {**base_payload, **extra_payload}
        if not stream:
            res = openai_request(url=self.url, headers=self.headers, payload=payload)
            _check_openai_finish_reason(result=res, payload=payload)
            responses = [ans["text"].strip() for ans in res["choices"]]
            return responses
        else:
            response = openai_request(
                url=self.url, headers=self.headers, payload=payload, read_response=False, stream=True
            )
            handler: TokenStreamingHandler = kwargs_with_defaults.pop("stream_handler", DefaultTokenStreamingHandler())
            return self._process_streaming_response(response=response, stream_handler=handler)

    def _process_streaming_response(self, response, stream_handler: TokenStreamingHandler):
        client = sseclient.SSEClient(response)
        tokens: List[str] = []
        try:
            for event in client.events():
                if event.data != TokenStreamingHandler.DONE_MARKER:
                    event_data = json.loads(event.data)
                    token: str = self._extract_token(event_data)
                    if token:
                        tokens.append(stream_handler(token, event_data=event_data["choices"]))
        finally:
            client.close()
        return ["".join(tokens)]  # return a list of strings just like non-streaming

    def _extract_token(self, event_data: Dict[str, Any]):
        return event_data["choices"][0]["text"]

    def _ensure_token_limit(self, prompt: Union[str, List[Dict[str, str]]]) -> Union[str, List[Dict[str, str]]]:
        """Ensure that the length of the prompt and answer is within the max tokens limit of the model.
        If needed, truncate the prompt text so that it fits within the limit.

        :param prompt: Prompt text to be sent to the generative model.
        """
        encoded_prompt = list(self.tokenizer.encode(cast(str, prompt)).ids)
        n_prompt_tokens = len(encoded_prompt)
        n_answer_tokens = self.max_length
        if (n_prompt_tokens + n_answer_tokens) <= self.max_tokens_limit:
            return prompt

        logger.warning(
            "The prompt has been truncated from %s tokens to %s tokens so that the prompt length and "
            "answer length (%s tokens) fit within the max token limit (%s tokens). "
            "Reduce the length of the prompt to prevent it from being cut off.",
            n_prompt_tokens,
            self.max_tokens_limit - n_answer_tokens,
            n_answer_tokens,
            self.max_tokens_limit,
        )

        decoded_string = self.tokenizer.decode(encoded_prompt[: self.max_tokens_limit - n_answer_tokens])
        return decoded_string

    @classmethod
    def supports(cls, model_name_or_path: str, **kwargs) -> bool:
        url = kwargs.get("api_base")
        valid_model = any(m for m in cls.get_supported_models(url) if m in model_name_or_path) or model_name_or_path == None or model_name_or_path == ""
        return valid_model
    
    @classmethod
    def get_supported_models(cls, url:str)->List[str]:
        try:
            result = requests.request("GET", f"{url}/models")
            if result.status_code != 200:
                raise ValueError(f"Could not get models from `{url}/models`!")
            json_response = json.loads(result.text)
            return [info["id"] for info in json_response["data"]]
        except Exception as e:
            raise ValueError(f"Could not get models from `{url}/models`!") from e
        