import torch
from fastapi import FastAPI, Request
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import uvicorn
import os

app = FastAPI()

model_id = "Qwen/Qwen3-VL-4B-Instruct"

model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    device_map="auto",
    trust_remote_code=True
)
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    data = await request.json()
    messages = data["messages"]

    do_sample = os.environ.get('greedy', 'false').lower() == 'false'
    top_p = float(os.environ.get('top_p', 0.8))
    top_k = int(os.environ.get('top_k', 20))
    temperature = float(os.environ.get('temperature', 0.7))
    repetition_penalty = float(os.environ.get('repetition_penalty', 1.0))
    presence_penalty = float(os.environ.get('presence_penalty', 1.5))
    max_new_tokens = int(os.environ.get('out_seq_length', 16384))
    
    inputs = processor.apply_chat_template(
        messages, 
        tokenize=True, 
        add_generation_prompt=True, 
        return_dict=True, 
        return_tensors="pt"
    ).to(model.device)

    generated_ids = model.generate(
        **inputs, 
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
        repetition_penalty=repetition_penalty
    )
    output_text = processor.batch_decode(generated_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]

    prompt_tokens = inputs.input_ids.shape[1]
    completion_tokens = generated_ids.shape[1] - prompt_tokens
    total_tokens = prompt_tokens + completion_tokens

    response_json = {
        "id": "chatcmpl-qwen",
        "object": "chat.completion",
        "created": 123456789,
        "model": "qwen3-vl",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": output_text},
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": int(prompt_tokens or 0),
            "completion_tokens": int(completion_tokens or 0),
            "total_tokens": int(total_tokens or 0)
        },
        "cost": 0.0
    }

    return response_json
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)