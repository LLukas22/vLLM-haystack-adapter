# vLLM-haystack-adapter


Simply connect your haystack pipeline to a selfhosted [vLLM-API](https://github.com/vllm-project/vllm) server. 

<p align="center">
    <img alt="vLLM" src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/source/assets/logos/vllm-logo-text-light.png" width="45%" style="vertical-align: middle;">
    <a href="https://www.deepset.ai/haystack/">
        <img src="https://raw.githubusercontent.com/deepset-ai/haystack/main/docs/img/haystack_logo_colored.png" alt="Haystack" width="45%" style="vertical-align: middle;">
    </a>
</p>

## Installation
Install the wrapper via pip:  `pip install vllm-haystack`

## Usage
To utilize the wrapper the `vLLMInvocationLayer` has to be used. 

Here is a simple example of how a `PromptNode` can be created with the wrapper.

```python
from haystack.nodes import PromptNode, PromptModel
from vllm_haystack import vLLMInvocationLayer


model = PromptModel(model_name_or_path="", invocation_layer_class=vLLMInvocationLayer, max_length=256, api_key="EMPTY", model_kwargs={
        "api_base" : API, # Replace this with your API-URL
        "maximum_context_length": 2048,
    })

prompt_node = PromptNode(model_name_or_path=model, top_k=1, max_length=256)
```
For more configuration examples, take a look at the unit-tests.

## Hosting a vLLM Server

To create an *OpenAI-Compatible Server* via vLLM you can follow the steps in the 
Quickstart section of their [documenetation](https://vllm.readthedocs.io/en/latest/getting_started/quickstart.html#openai-compatible-server).
