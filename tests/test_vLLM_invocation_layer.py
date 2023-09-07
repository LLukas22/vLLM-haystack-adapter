from haystack.nodes import PromptNode, PromptModel, PromptTemplate
from src.vllm_haystack import vLLMInvocationLayer

API = "" #Define your API URL here
LLAMA_TOKENIZER = "hf-internal-testing/llama-tokenizer"

def test_vLLM_can_be_costructed():
    vllm = vLLMInvocationLayer(API,tokenizer=LLAMA_TOKENIZER)
    assert isinstance(vllm, vLLMInvocationLayer)
    assert vllm.model_name_or_path is not None
    assert vllm.tokenizer is not None

def test_vLLM_can_be_used_in_prompt_model():
    model = PromptModel(model_name_or_path="", invocation_layer_class=vLLMInvocationLayer, max_length=256, api_key="EMPTY", model_kwargs={
    "api_base" : API,
    "tokenizer": LLAMA_TOKENIZER,
    "maximum_context_length": 2048,
    })
    assert model is not None
    result = model.invoke("What is the capital of France?")
    assert result is not None and len(result) > 0


def test_vLLM_can_be_used_in_prompt_node():
    model = PromptModel(model_name_or_path="", invocation_layer_class=vLLMInvocationLayer, max_length=256, api_key="EMPTY", model_kwargs={
    "api_base" : API,
    "tokenizer": LLAMA_TOKENIZER,
    "maximum_context_length": 2048,
    })
    assert model is not None
    prompt_node = PromptNode(model_name_or_path=model, top_k=1, max_length=256)
    result = prompt_node.prompt("What is the capital of France?")
    assert result is not None and len(result) > 0


def test_vLLM_prompt_is_truncated():
    model = PromptModel(model_name_or_path="", invocation_layer_class=vLLMInvocationLayer, max_length=64, api_key="EMPTY", model_kwargs={
    "api_base" : API,
    "tokenizer": LLAMA_TOKENIZER,
    "maximum_context_length": 128,
    })
    assert model is not None
    prompt_node = PromptNode(model_name_or_path=model, top_k=1, max_length=64)
    result = prompt_node.prompt("What is the capital of France?" * 100)
    assert result is not None and len(result) > 0


def test_vLLM_can_stream():
    model = PromptModel(model_name_or_path="", invocation_layer_class=vLLMInvocationLayer, max_length=64, api_key="EMPTY", model_kwargs={
    "api_base" : API,
    "tokenizer": LLAMA_TOKENIZER,
    "maximum_context_length": 2048,
    "stream":True
    })
    assert model is not None
    prompt_node = PromptNode(model_name_or_path=model, top_k=1, max_length=64)
    result = prompt_node.prompt("What is the capital of France?" * 5)
    assert result is not None and len(result) > 0
    






