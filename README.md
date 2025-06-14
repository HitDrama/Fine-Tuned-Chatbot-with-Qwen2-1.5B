# Fine-Tuned Chatbot with Qwen2-1.5B

A conversational AI chatbot fine-tuned from **Qwen2-1.5B** (by Qwen Team, Alibaba Cloud) using the `tatsu-lab/alpaca` dataset. Designed for natural, human-like dialogue.

---

## 🔧 Fine-Tuning Configuration
### 📦 Model Weights
- Due to model size, weights are hosted on Google Drive:  
- 👉 [Download Qwen2-1.5B Fine-Tuned Chatbot](https://drive.google.com/drive/folders/1nP0fparpJdVbfXxKPuNEqyQIczvZACj5?usp=sharing)
---

### 🧠 Model Loading with Quantization (4-bit)
```python
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quant_config,
    device_map="auto",
)
```
### 📚 Dataset
```python
dataset = load_dataset("tatsu-lab/alpaca", split="train[:2000]")

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding='max_length', max_length=256)

tokenizer_dataset = dataset.map(preprocess_function, batch_size=True, remove_columns=dataset.column_names)
```
---
### 🔁 LoRA Fine-Tuning Setup
``` python
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
```
---
### 🧪 Demo

![Demo](https://raw.githubusercontent.com/HitDrama/Fine-Tuned-Chatbot-with-Qwen2-1.5B/main/static/qwen.gif)


---
### 👤 Author
Đặng Tố Nhân Developer / Fine-Tuning Engineer
