from haystack.nodes import PromptNode, PromptModel, PromptTemplate
from src.vllm_haystack import vLLMLocalInvocationLayer

LLAMA_TOKENIZER = "hf-internal-testing/llama-tokenizer"
LLAMA_MODEL = "decapoda-research/llama-7b-hf"

def test_vLLM_can_be_costructed():
    vllm = vLLMLocalInvocationLayer(LLAMA_MODEL,tokenizer=LLAMA_TOKENIZER)
    assert isinstance(vllm, vLLMLocalInvocationLayer)
    assert vllm.model_name_or_path is not None
    assert vllm.tokenizer is not None
    

def test_vLLM_can_be_used_in_prompt_model():
    model = PromptModel(model_name_or_path=LLAMA_MODEL, invocation_layer_class=vLLMLocalInvocationLayer, max_length=256, model_kwargs={
    "tokenizer": LLAMA_TOKENIZER,
    "maximum_context_length": 2048,
    })
    assert model is not None
    result = model.invoke("What is the capital of France?")
    assert result is not None and len(result) > 0
    
def test_vLLM_can_be_used_in_prompt_node():
    model = PromptModel(model_name_or_path=LLAMA_MODEL, invocation_layer_class=vLLMLocalInvocationLayer, max_length=256, model_kwargs={
    "tokenizer": LLAMA_TOKENIZER,
    "maximum_context_length": 2048,
    })
    assert model is not None
    prompt_node = PromptNode(model_name_or_path=model, top_k=1, max_length=256)
    result = prompt_node.prompt("What is the capital of France?")
    assert result is not None and len(result) > 0


def test_vLLM_prompt_is_truncated():
    model = PromptModel(model_name_or_path=LLAMA_MODEL, invocation_layer_class=vLLMLocalInvocationLayer, max_length=64, model_kwargs={
    "tokenizer": LLAMA_TOKENIZER,
    "maximum_context_length": 128,
    })
    assert model is not None
    prompt_node = PromptNode(model_name_or_path=model, top_k=1, max_length=64)
    result = prompt_node.prompt("What is the capital of France?" * 100)
    assert result is not None and len(result) > 0
    
    
def test_memory_can_be_limited():
    model = PromptModel(model_name_or_path=LLAMA_MODEL, invocation_layer_class=vLLMLocalInvocationLayer, max_length=64, model_kwargs={
    "tokenizer": LLAMA_TOKENIZER,
    "maximum_context_length": 128,
    "gpu_memory_utilization" : 0.5
    })
    assert model is not None
    prompt_node = PromptNode(model_name_or_path=model, top_k=1, max_length=64)
    result = prompt_node.prompt("What is the capital of France?" * 100)
    assert result is not None and len(result) > 0