# llamaDos

LlamaDos is a model oriented to have conversations in Spanish. It results from a finetuning of the Llama2-7b model by Meta using various optimization techniques such as LoRa, quantization, gradient accumulation and much more.

This has allowed the training to be performed on a single consumer graph (RTX 3090). More specifically, more than 250,000 conversational data were used and the training took approximately 140 hours.

Available on Hugging Face: https://huggingface.co/garrachonr/llamaDos

The training has been performed following the original data structure of the Llama2 paper, so it is recommended to follow the same structure for inference and re-training:

```python
<s>[INST] <<SYS>>
{{ You are a helpful, respectful and honest conversational assistant. Have a conversation with the user in a natural way.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. }}
<</SYS>>

{{ user_msg_1 }} [/INST] {{ model_answer_1 }} </s><s>[INST] {{ user_msg_2 }} [/INST] {{ model_answer_1 }} </s>
```

# Inference

In order to use this model:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
base_model = AutoModelForCausalLM.from_pretrained(
    "garrachonr/llamaDos",
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map=device_map,
)
tokenizer = AutoTokenizer.from_pretrained("garrachonr/llamaDos", trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Run text generation pipeline with llamaDos
system_prompt = "You are a helpful, respectful and honest conversational assistant. Have a conversation with the user in a natural way.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature."
prompt1 = "Acabo de adoptar un perro"
prompt2 = "Muy buena decisión, te gustan los perros?"
prompt3 = "Si, cuando era pequeño tenía uno y ahora he podido adoptar otro"
text = "<s>[INST] <<SYS>> {} <</SYS>> {} [/INST] {} </s><s>[INST] {} [/INST]".format(system_prompt, prompt1, prompt2, prompt3)
pipe = pipeline(task="text-generation", model=base_model, tokenizer=tokenizer, max_length=200)
result = pipe(text)
print(result[0]['generated_text'])
```

# Finetuning

To recreate the obtained results you can follow this command:

```python
python trl/examples/scripts/sft_trainer.py \
    --model_name meta-llama/Llama-2-7b-chat-hf \
    --dataset_name garrachonr/DB-Spanish-Llama2 \
    --load_in_4bit \
    --use_peft \
    --batch_size 12 \
    --gradient_accumulation_steps 32
```

This work is funded by the Comunidad de Madrid through the call Research Grants for Young Investigators from Universidad Politécnica de Madrid (GENIUS:APOYO-JOVENES-21-TAXTYC-32-K61X37), and supported by the following projects: European Commission through Project ASTOUND (101071191–-HORIZON-EIC-2021-PATHFINDERCHALLENGES-01) and BEWORD (PID2021-126061OB-C43) funded by
MCIN/AEI/10.13039/501100011033 and, as appropriate, by “ERDF A way of making Europe”, by the
“European Union”.

We also want to give thanks to MS Azure services (especially to Irving Kwong) for their sponsorship to translate into Spanish all dialogue databases.
