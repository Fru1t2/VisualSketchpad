import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
import json
from PIL import Image
from dataclasses import dataclass
from typing import List, Dict, Any
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import Dataset
from types import SimpleNamespace

DEFAULT_IMAGE_ROOT = "/mnt/ssd2/cvlab_intern/VisualSketchpad/agent/fine_tuning/"
model_id = "Qwen/Qwen3-VL-4B-Instruct"


try:
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel
except ImportError:
    from transformers import Qwen3VLForConditionalGeneration
    print("Import Error")

def patched_get_image_features(self, pixel_values, image_grid_thw, **kwargs):
    
    image_outputs = self.visual(pixel_values, grid_thw=image_grid_thw)
    image_features = image_outputs[0] 
    
    if hasattr(self.visual, "merger"):
        image_features = self.visual.merger(image_features)

    grid_cpu = image_grid_thw.cpu()
    prod = grid_cpu.prod(-1)
    merge_size = self.visual.spatial_merge_size
    split_sizes = (prod // (merge_size**2)).tolist()
    
    split_tensors = image_features.split(split_sizes, dim=0)
    
    ds_features = getattr(image_outputs, "deepstack_features", None)
    
    return SimpleNamespace(
        pooler_output=split_tensors,
        deepstack_features=ds_features,
        hidden_states=getattr(image_outputs, "hidden_states", None),
        attentions=getattr(image_outputs, "attentions", None)
    )
Qwen3VLModel.get_image_features = patched_get_image_features

model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="sdpa"
)
model.config.use_cache = False

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    task_type=TaskType.CAUSAL_LM,
    lora_dropout=0.1,
    bias="none",
)

model = get_peft_model(model, peft_config)

processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

class VisualExpertDataset(Dataset):
    def __init__(self, json_path, processor, image_root=DEFAULT_IMAGE_ROOT, max_length=4096):
        with open(json_path, "r", encoding="utf-8") as f:
            try:
                self.data = json.load(f)
            except:
                f.seek(0)
                self.data = [json.loads(line) for line in f]
        
        self.image_root = image_root
        self.processor = processor
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = os.path.join(self.image_root, item["image"])
        image = Image.open(img_path).convert("RGB")

        # image = image.resize((448, 448))

        user_text = item["conversation"][0]["value"]
        assistant_text = item["conversation"][1]["value"]

        prompt_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": user_text},
                ],
            }
        ]

        full_messages = prompt_messages + [
            {"role": "assistant", "content": [{"type": "text", "text": assistant_text}]}
        ]

        prompt_text = self.processor.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
        prompt_inputs = self.processor(text=[prompt_text], images=[image], return_tensors="pt")
        prompt_len = prompt_inputs["input_ids"].shape[1]

        full_text = self.processor.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=False)
        full_text += self.processor.tokenizer.eos_token 
        
        inputs = self.processor(text=[full_text], images=[image], return_tensors="pt")
        
        input_ids = inputs["input_ids"][0]
        labels = input_ids.clone()
        labels[:prompt_len] = -100

        return {
            "input_ids": input_ids,
            "labels": labels,
            "pixel_values": inputs["pixel_values"].squeeze(0),      # [N_patches, Hidden]
            "image_grid_thw": inputs["image_grid_thw"].squeeze(),  # [3] (T, H, W)
            "mm_token_type_ids": inputs["mm_token_type_ids"][0] if "mm_token_type_ids" in inputs else None
        }

@dataclass
class MultimodalDataCollator:
    processor: Any

    def __call__(self, instances: List[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [instance["input_ids"] for instance in instances]
        labels = [instance["labels"] for instance in instances]
        
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.processor.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100
        )
        
        pixel_values = torch.cat([instance["pixel_values"] for instance in instances], dim=0)
        image_grid_thw = torch.stack([instance["image_grid_thw"] for instance in instances], dim=0).long()
        
        batch = {
            "input_ids": input_ids,
            "labels": labels,
            "pixel_values": pixel_values.to(dtype=torch.bfloat16),
            "image_grid_thw": image_grid_thw,
            "attention_mask": input_ids.ne(self.processor.tokenizer.pad_token_id),
        }
        
        if instances[0].get("mm_token_type_ids") is not None:
            batch["mm_token_type_ids"] = torch.nn.utils.rnn.pad_sequence(
                [instance["mm_token_type_ids"] for instance in instances], batch_first=True, padding_value=0
            )
        

        return batch

train_dataset = VisualExpertDataset(
    json_path="vsr_lora_train.jsonl",
    processor=processor,
    max_length=1024
)

training_args = TrainingArguments(
    output_dir="./qwen3-vl-lora-v4",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    bf16=False,
    fp16=True,
    learning_rate=5e-5, # 원래 2e-4로 했었음
    weight_decay=0.01,
    num_train_epochs=2,
    logging_steps=10,
    save_steps=100,
    gradient_checkpointing=True,
    remove_unused_columns=False,
    report_to="none"
)

data_collator = MultimodalDataCollator(processor=processor)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator, 
)


trainer.train()

trainer.save_model("./qwen3-vl-lora-v4-final") 
processor.save_pretrained("./qwen3-vl-lora-v4-final")