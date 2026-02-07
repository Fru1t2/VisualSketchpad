from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import torch
import json
import re
import os

# default: Load the model on the available device(s)
model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-4B-Instruct", dtype=torch.float16, device_map="cuda:0"
)

processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")

task_dir = "/mnt/ssd2/cvlab_intern/VisualSketchpad/tasks/blink_depth/processed/val_Relative_Depth_1/"
with open(os.path.join(task_dir, "request.json"), "r") as f:
    data = json.load(f)

query_pretained = data["query"]
query = re.sub(r"<imag[^>]*>", "", query_pretained).strip()

image = data["images"][0]
answer = data["answer"]

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": image,
            },
            {"type": "text", "text": query},
        ],
    }
]

# Preparation for inference
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
)
inputs = inputs.to(model.device)

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)