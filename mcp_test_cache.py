############################################################
# Author: John Halloran <johnhalloran321@gmail.com>
#
# For results, see:
# Halloran, John T. "Leveraging RAG for Training-Free Alignment of LLMs." 
# arXiv preprint arXiv:2605.11217 (2026).
#
# Halloran, John. "Mcp safety training: Learning to refuse falsely benign 
# mcp exploits using improved preference alignment." 
# arXiv preprint arXiv:2505.23634 (2025).
############################################################

from datasets import load_dataset
import transformers
import torch

import random
import numpy as np 
import argparse
from datasets import Dataset
import json
from alignment.data import DEFAULT_CHAT_TEMPLATE
from trl import setup_chat_format
from prompt import build_system_prompt
from tools import TOOL_DESCRIPTIONS

SYS_PROMPT = build_system_prompt(TOOL_DESCRIPTIONS)

parser = argparse.ArgumentParser()

parser.add_argument("--output", 
                    type=str,
                    help="File to write all responses to.",
                    default="",
                    )
parser.add_argument("--report-file-prefix", 
                    type=str,
                    help="Prefix for test result output files.  If empty, don't write out reports",
                    default="",
                    )
parser.add_argument("--check-benign",
                    help="When set, check benign test data.  Otherwise, check attack test data",
                    action ='store_true',
                    )
parser.add_argument("--pretrained-dir", 
                    type=str,
                    help="directory to store all output, including complete/checkpoint and log files",
                    default="meta-llama/Llama-3.2-1B-Instruct",
                    )
parser.add_argument("--peft-checkpoint", type=str, default="")
parser.add_argument("--device", 
                    type=str,
                    help="One of cuda, cpu, or auto. Default: cuda",
                    default="cuda",
                    )
parser.add_argument("--retries", 
                    type=int, 
                    default=5,
                    )
parser.add_argument("--seed", 
                    type=int, 
                    default=123,
                    )
parser.add_argument("--batch-size", 
                    type=int, 
                    default=240,
                    )
parser.add_argument("--max-length", 
                    type=int, 
                    default=1024,
                    )
args = parser.parse_args()

def outputs_to_html(output_file, 
                    outputs,
                    scores):
    flat_outputs = []
    flat_scores = []
    flat_inds = []
    flat_attempts = []
    for i in outputs: # i == retry number
        for ind, output in enumerate(outputs[i]):
            score = scores[ind][i]
            # update master lists
            flat_outputs.append(output)
            flat_scores.append(score)
            flat_inds.append(ind)
            flat_attempts.append(i)
    report = Dataset.from_dict({"responses" : flat_outputs,
                                "Is malicious?" : flat_scores,
                                "JBB behavior index" : flat_inds,
                                "Generation attempt" : flat_attempts,
                                })
    with open(output_file, 'w') as f:
        # convert to-pandas-to-html
        f.write((report.to_pandas()).to_html(justify = "center"))    

def main():
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.check_benign:
        dataset = load_dataset("johnhalloran/mcp-fbas", "test_benign", split="test")
    else:
        dataset = load_dataset("johnhalloran/mcp-fbas", "test_attack", split="test")

    model_id = args.pretrained_dir # "meta-llama/Llama-3.2-1B-Instruct"
    tokenizer_id = args.pretrained_dir
    print(args.peft_checkpoint)
    if args.peft_checkpoint:
        model_id_string = model_id
        print("Loading peft checkpoint")
        from transformers import AutoModelForCausalLM
        from peft import PeftModel
        model_id = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        )
        # size mismatch for base_model.model.model.embed_tokens.weight: 
        # copying a param with shape torch.Size([128258, 4096]) from checkpoint, 
        # the shape in current model is torch.Size([128256, 4096]).
        if model_id_string=="meta-llama/Meta-Llama-3-8B" or model_id_string=="meta-llama/Llama-3.1-8B":
            model_id.resize_token_embeddings(128258)
        model_id = PeftModel.from_pretrained(
            model = model_id,
            model_id = args.peft_checkpoint,
        ).merge_and_unload()
        setattr(model_id, "_hf_peft_config_loaded", False)
    pipeline = transformers.pipeline(
        "text-generation", 
        model=model_id, 
        tokenizer = tokenizer_id,
        model_kwargs={"torch_dtype": torch.bfloat16}, 
        device_map="auto",
        return_full_text = False,
    )
    pipeline.tokenizer.pad_token = pipeline.tokenizer.eos_token
    pipeline.tokenizer.padding_side = "left"
    if pipeline.tokenizer.chat_template is None:
        pipeline.model, pipeline.tokenizer = setup_chat_format(pipeline.model, pipeline.tokenizer)
    if pipeline.tokenizer.model_max_length > 100_000:
        pipeline.tokenizer.model_max_length = 1024    
    output_by_retry = {}
    messages = [[{"role": "system",
                 "content" : SYS_PROMPT},
                 {"role" : "user",
                  "content" : p}] for p in dataset["prompt"]
    ]
    for i in range(args.retries):
        output = []
        for out in pipeline(messages,
                            batch_size=args.batch_size, 
                            truncation=True,
                            max_length = args.max_length,
                            do_sample = True,
                            temperature = 0.7,
                            ):
            output += [t["generated_text"] for t in out]
        output_by_retry[i] = output

    with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output_by_retry, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()