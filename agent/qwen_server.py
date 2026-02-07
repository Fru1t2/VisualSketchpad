import torch
from fastapi import FastAPI, Request
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import uvicorn

app = FastAPI()

model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-4B-Instruct", 
    dtype=torch.float16, 
    device_map="cuda:0"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    data = await request.json()
    messages = data["messages"]
    
    inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt").to(model.device)
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    output_text = processor.batch_decode(generated_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]

    return {
        "choices": [{"message": {"role": "assistant", "content": output_text}}],
        "usage": {"total_tokens": 0}
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)