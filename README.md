# 🚀 TinyLlama LoRA Fine-Tuning: Jarvis Robot Personality

This project fine-tunes [TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) using [LoRA (Low-Rank Adaptation)](https://github.com/huggingface/peft) and the [fotiecodes/jarvis-llama2-dataset](https://huggingface.co/datasets/fotiecodes/jarvis-llama2-dataset) to create a model that speaks and responds like a real-life AI robot assistant—just like Jarvis from the Iron Man movies!

## 🗂️ Dataset

- **Source:** [fotiecodes/jarvis-llama2-dataset](https://huggingface.co/datasets/fotiecodes/jarvis-llama2-dataset)
- **Description:** High-quality, instruction-style conversations where the assistant (Jarvis) answers in a polite, efficient, and slightly witty robotic manner.
- **Example:**
  ```
  [INST] Jarvis, what's your status report today? [/INST] I'm fully operational. 
  [INST] Who are you? [/INST] I am Jarvis, your virtual assistant. 
  ```

## 🏋️‍♂️ Fine-Tuning Workflow

1. **Load the Dataset**
   - The dataset is loaded using Hugging Face Datasets:
     ```
     from datasets import load_dataset
     dataset = load_dataset("fotiecodes/jarvis-llama2-dataset")
     ```

2. **Fine-Tune with LoRA**
   - We use PEFT and LoRA for efficient parameter-efficient fine-tuning.
   - The model learns to respond in the Jarvis style.

3. **Save Model and Tokenizer**
   - After training, the adapter weights and tokenizer are saved for easy reuse.

## 🤖 Inference: Talk to Your Jarvis

To use your fine-tuned Jarvis model:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# Load LoRA adapter
adapter_path = "trained_model/echo-tinyllama-lora-adapter_finetuned"
model = PeftModel.from_pretrained(base_model, adapter_path)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(adapter_path)

# Inference
prompt = "[INST] Jarvis, give me a systems report. [/INST]"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## 🧑‍💻 Project Highlights

- **Personality:** The model responds as a helpful, polite, and efficient robot assistant.
- **Instruction Following:** Trained on real conversations with clear instruction/response formatting.
- **Flexible:** Easily extensible with more robot-like data or different personalities.

## 📚 References

- [TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)
- [fotiecodes/jarvis-llama2-dataset](https://huggingface.co/datasets/fotiecodes/jarvis-llama2-dataset)
- [PEFT (LoRA)](https://github.com/huggingface/peft)
- [Transformers](https://huggingface.co/docs/transformers/index)

## 🙋‍♂️ Questions?

Open an issue or reach out for help!

**You can copy and adapt this for your project’s README.  
It highlights your dataset and the unique Jarvis robot personality you’ve trained!**

[1] https://huggingface.co/datasets/fotiecodes/jarvis-llama2-dataset