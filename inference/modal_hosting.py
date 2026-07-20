from fastapi import FastAPI, Body, Response
import modal
from modal import App, Image, asgi_app, enter, method
from typing import Union, Annotated, Dict, Optional
from pydantic import BaseModel

app = App("SabiYarn")

image = Image.debian_slim().pip_install(
    "transformers", "torch", "huggingface-hub==0.23.4", "fastapi", "pydantic"
)

repo_name = "BeardedMonster/SabiYarn-125M"

GPU_CONFIG = "T4"

device = "cuda"


with image.imports():
    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        GenerationConfig,
    )  # , TextIteratorStreamer
    from huggingface_hub import snapshot_download


@app.cls(
    gpu=GPU_CONFIG,
    scaledown_window=60 * 10,
    timeout=60 * 60,
    image=image,
)
@modal.concurrent(max_inputs=1000)
class TextGeneration:
    @enter()
    def load_model(self):
        snapshot_download(repo_id=repo_name, cache_dir="/cache")
        self.tokenizer = AutoTokenizer.from_pretrained(
            repo_name, cache_dir="/cache", trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            repo_name, cache_dir="/cache", trust_remote_code=True
        ).to(device)

    @method()
    def generate(self, prompt, config={}):
        generation_config = GenerationConfig(
            max_length=config.get("max_length", 100),
            num_beams=config.get("num_beams", 5),
            do_sample=config.get("do_sample", True),
            temperature=config.get("temperature", 0.9),
            top_k=config.get("top_k", 50),
            top_p=config.get("top_p", 0.95),
            repetition_penalty=config.get("repetition_penalty", 2.0),
            length_penalty=config.get("length_penalty", 1.7),
            early_stopping=config.get("early_stopping", True),
        )
        max_new_tokens = config.get("max_new_tokens", 50)
        input_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
        output = self.model.generate(
            input_ids,
            generation_config=generation_config,
            max_new_tokens=max_new_tokens,
        )
        input_len = len(input_ids[0])
        generated_text = self.tokenizer.decode(
            output[0][input_len:], skip_special_tokens=True
        )
        generated_text = generated_text.split("|end_of_text|")[0]
        # print(generated_text)
        return generated_text


web_app = FastAPI()


class Request(BaseModel):
    prompt: str
    config: Optional[Dict]


@web_app.post("/predict")
async def predict(request: Request):
    output = TextGeneration().generate.remote(request.prompt, request.config)
    return Response(content=output)


@app.function(image=image)
@asgi_app()
def fastapi_app():
    return web_app


# if __name__ == '__main__':
#     import requests
#     from transformers import GenerationConfig

#     generation_config = {
#     "max_length":100,
#     "num_beams":5,
#     "do_sample":True,
#     "temperature":0.9,
#     "top_k":50,
#     "top_p":0.95,
#     "repetition_penalty":2.0,
#     "length_penalty":1.7,
#     "early_stopping":True
# }

#     text = requests.post('https://pauljeffrey--sabiyarn-fastapi-app.modal.run/predict', data={"prompt": "<prompt> who are you? <response>", "config": generation_config})
#     print(text)
