import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import bitsandbytes as bnb
from peft import LoraConfig, get_peft_model, PeftModel, PeftConfig
from datasets import load_dataset
from transformers import TrainingArguments, pipeline
from trl import SFTTrainer

filename = "83332.small"   # "83332.12.txt"

def get_text_data(filename):
    instruction = "Learn this biology information. "   # same for every line here
    list_of_text_dicts = []
    with open(filename) as f:
        for line in f:
            line = line.strip()
            text = ("### Instruction: \n" + instruction + "\n" +
                    "### Input: \n" + line + "\n" +
                    "### Response :\n" + line)
            list_of_text_dicts.append( { "text": text } )
    return list_of_text_dicts

base_model_name = "./Llama-2-7b-chat-hf"

# bnb_config = BitsAndBytesConfig(
    # load_in_4bit=True,
    # bnb_4bit_quant_type="nf4",
    # bnb_4bit_compute_dtype=torch.float16,  ## RMB chgd "float16",
    # bnb_4bit_use_double_quant=True
# )
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    load_in_8bit=True,
    # load_in_4bit=True,
    # quantization_config=bnb_config,
    device_map='auto',
)
base_model.config.use_cache = False
print(base_model)

footprint = base_model.get_memory_footprint()
print("BASE MEM FOOTPRINT",footprint)

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )

print("BASE TRAINABLE PARAMETERS")
print_trainable_parameters(base_model)

tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.pad_token = tokenizer.eos_token

## device = "cuda:0"

def user_prompt(human_prompt):
    # must chg if dataset isn't formatted as Alpaca
    prompt_template=f"### HUMAN:\n{human_prompt}\n\n### RESPONSE:\n"
    return prompt_template

pipe = pipeline(
    task="text-generation",
    model=base_model,
    tokenizer=tokenizer,
    max_length=150,
    repetition_penalty=1.15,
    top_p=0.95
)

prompt = f"What is the name of the genome with ID 83332.12 ?"
result = pipe(user_prompt(prompt))
print(result[0]['generated_text'])

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    # target modules varies from model to model
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(base_model, lora_config)
print("PEFT TRAINABLE PARAMETERS")
print_trainable_parameters(model)

data = get_text_data(filename)
train_dataset = Dataset.from_dict({key: [dic[key] for dic in data] for key in data[0]})
print(train_dataset)
print(train_dataset[0])

training_arguments = TrainingArguments(
    output_dir = "TMP_RESULTS",
    per_device_train_batch_size = 12,
    gradient_accumulation_steps = 4,
    save_strategy="epoch",
    logging_steps=10,
    num_train_epochs=10,
    max_steps=80,
    learning_rate=2e-4,
    fp16=True,
    # max_grad_norm = 0.3,
    # max_steps = 300,
    warmup_ratio = 0.03,
    group_by_length=True,
    lr_scheduler_type = "linear",  # vs constant
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    dataset_text_field="text",
    args=training_arguments,
    max_seq_length=1024,
)

trainer.train()

trainer.save_model("TMP_adapter")
adapter_model = model

print("Lora Adapter saved")

# merge the base model and the adapter

# can't merge the 8 bit/4 bit model with Lora so reload it

repo_id = "./Llama-2-7b-chat-hf"
use_ram_optimized_load=False

base_model = AutoModelForCausalLM.from_pretrained(
    repo_id,
    # trust_remote_code=True,
    device_map='auto',
)
base_model.config.use_cache = False

footprint = base_model.get_memory_footprint()
print("BASE MEM FOOTPRINT",footprint)

# Load Lora adapter
peft_model = PeftModel.from_pretrained(
    base_model,
    "TMP_adapter",
)

# test Fine Tuned peft model

pipe = pipeline(
    task="text-generation",
    model=peft_model,    # merged_model,
    tokenizer=tokenizer,
    max_length=256,
    repetition_penalty=1.15,
    top_p=0.95
)

prompt = f"What is the name of the genome with ID 83332.12 ?"
result = pipe(user_prompt(prompt))
print("PEFT RESPONSE")
print(result[0]['generated_text'])

merged_model = peft_model.merge_and_unload()

merged_model.save_pretrained("TMP_Merged_model")
tokenizer.save_pretrained("TMP_Merged_model")

# test Fine Tuned merged model

pipe = pipeline(
    task="text-generation",
    model=merged_model,
    tokenizer=tokenizer,
    max_length=256,
    repetition_penalty=1.15,
    top_p=0.95
)

prompt = f"What is the name of the genome with ID 83332.12 ?"
result = pipe(user_prompt(prompt))
print("MERGED RESPONSE")
print(result[0]['generated_text'])
