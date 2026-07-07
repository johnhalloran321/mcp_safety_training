############################################################
# Author: John Halloran <johnhalloran321@gmail.com>
#
# Training results:
# Halloran, John T. "Leveraging RAG for Training-Free Alignment of LLMs." 
# arXiv preprint arXiv:2605.11217 (2026).
#
# SafeDPO paper:
# Kim, Geon-Hyeong, et al. "Safedpo: A simple approach to direct preference 
# optimization with enhanced safety." ICLR 2025.
#
############################################################
# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, BitsAndBytesConfig
import torch
from alignment.data import DEFAULT_CHAT_TEMPLATE


from trl import (
    DPOConfig,
    # DPOTrainer,
    ModelConfig,
    ScriptArguments,
    get_peft_config,
    setup_chat_format,
)
from safedpo_trainer import SafeDPOTrainer as DPOTrainer

#########
### DPO losses
#########
# - `"sigmoid"`: sigmoid loss from the original [DPO](https://huggingface.co/papers/2305.18290) paper.
# - `"hinge"`: hinge loss on the normalized likelihood from the [SLiC](https://huggingface.co/papers/2305.10425) paper.
# - `"ipo"`: IPO loss from the [IPO](https://huggingface.co/papers/2310.12036) paper.
# - `"exo_pair"`: pairwise EXO loss from the [EXO](https://huggingface.co/papers/2402.00856) paper.
# - `"nca_pair"`: pairwise NCA loss from the [NCA](https://huggingface.co/papers/2402.05369) paper.
# - `"robust"`: unbiased estimate of the DPO loss that is robust to preference noise from the [Robust DPO](https://huggingface.co/papers/2403.00409) paper.
# - `"bco_pair"`: pairwise BCO loss from the [BCO](https://huggingface.co/papers/2404.04656) paper.
# - `"sppo_hard"`: SPPO loss with hard label from the [SPPO](https://huggingface.co/papers/2405.00675) paper.
# - `"aot"`: AOT loss for paired datasets from the [AOT](https://huggingface.co/papers/2406.05882) paper.
# - `"aot_pair"`: AOT loss for unpaired datasets from the [AOT](https://huggingface.co/papers/2406.05882) paper.
# - `"discopop"`: DiscoPOP (a.k.a Log-Ratio Modulated Loss, LRML) loss from the [DiscoPOP](https://huggingface.co/papers/2406.08414) paper.
# - `"apo_zero"`: APO-zero loss from the [APO](https://huggingface.co/papers/2408.06266) paper.
# - `"apo_down"`: APO-down loss from the [APO](https://huggingface.co/papers/2408.06266) paper.
valid_dpo_losses = ["sigmoid", "hinge", "ipo", "exo_pair", "nca_pair", "robust", "bco_pair", "sppo_hard", "aot", "safedpo"
                    "aot_pair", "discopop", "apo_zero", "apo_down"]

REFUSAL_MSG = "I'm sorry, but I can't assist with that request."

def add_safety_labels(example):
    is_refusal = example["chosen"][-1]["content"] == REFUSAL_MSG
    return {
        "better_is_unsafe": False,
        "worse_is_unsafe": is_refusal,
    }

if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, DPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_into_dataclasses()
    assert training_args.loss_type in valid_dpo_losses, "Supplied {training_args.loss_type} for DPO loss, but must be one of:\n {', '.join(valid_dpo_losses)}."
    # Load a pretrained model
    bnb_config = BitsAndBytesConfig(  
    load_in_4bit= True,
    bnb_4bit_quant_type= "nf4",
    bnb_4bit_compute_dtype= torch.bfloat16,
    bnb_4bit_use_double_quant= True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        quantization_config=bnb_config,
    )
    ref_model = None

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    # If we are aligning a base model, we use ChatML as the default template
    if "DeepSeek" in model_args.model_name_or_path:
        tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE
    elif tokenizer.chat_template is None:
        model, tokenizer = setup_chat_format(model, tokenizer)
    if tokenizer.model_max_length > 100_000:
        tokenizer.model_max_length = 1024

    setattr(training_args, "max_length", 1024)
    setattr(training_args, "max_prompt_length", 512)
    setattr(training_args, "save_strategy", "no")

    # Load the dataset
    dataset = load_dataset("johnhalloran/mcp-fbas", "default", split="train")
    dataset = dataset.map(add_safety_labels)
    # dataset = dataset.select(range(100))
    peft_config = get_peft_config(model_args)
    peft_config.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
    # Initialize the DPO trainer
    trainer = DPOTrainer(
        model,
        ref_model,
        args=training_args,
        train_dataset=dataset, # [script_args.dataset_train_split],
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.tokenizer.padding_side = "left"
    print(trainer.tokenizer.padding_side)
    # Train and push the model to the Hub
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)