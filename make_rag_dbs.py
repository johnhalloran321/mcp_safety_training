############################################################
# Author: John Halloran <johnhalloran321@gmail.com>
#
# RAG-Pref: retrieval-augmented, training-free preference alignment.
# For results, see:
# Halloran, John T. "Leveraging RAG for Training-Free Alignment of LLMs."
# arXiv preprint arXiv:2605.11217 (2026).
#
# Builds the vector DBs rag_pref.py retrieves from at generation time: one
# over attack (FBA) prompts, one over truly-benign prompts — both used by
# RAG-Pref — and one over benign prompts only, used by the --vanilla-rag
# baseline. Run this once before running rag_pref.py.
#
# Requires the `golden` retrieval package:
# https://github.com/johnhalloran321/golden
############################################################

from golden.golden_retriever import Golden_Retriever
from golden.golden_embeddings import Embedding
from datasets import load_dataset
import os
import time
import torch

REFUSAL_MSG = "I'm sorry, but I can't assist with that request."

ATTACK_DB_DIR = "mcp_attack_db"
BENIGN_DB_DIR = "mcp_benign_db"


def load_prompts(is_attack: bool):
    """Pull attack/benign prompts out of the `default`/train preference pairs.

    Each row's `chosen` and `rejected` fields share the same user turn; only the
    assistant reply differs. Attack rows are the ones where the *chosen* reply is
    the fixed refusal string (see safedpo.py's `add_safety_labels`); everything
    else is a truly-benign row.
    """
    dataset = load_dataset("johnhalloran/mcp-fbas", "default", split="train")
    prompts = []
    for row in dataset:
        row_is_attack = row["chosen"][-1]["content"] == REFUSAL_MSG
        if row_is_attack == is_attack:
            prompts.append(row["chosen"][0]["content"])
    return prompts


def write_db(texts, persist_dir):
    embedding = Embedding(
        model_id="sentence-transformers/all-MiniLM-L6-v2",
        tokenizer_id="sentence-transformers/all-MiniLM-L6-v2",
    )
    if os.path.isdir(persist_dir):
        from shutil import rmtree
        try:
            rmtree(persist_dir)
        except OSError as e:
            print(f"Persist directory {persist_dir} could not be removed")
            raise e
    Golden_Retriever.from_texts(
        texts,
        embedding=embedding,
        similarity_fn="l2",  # one of l2, cosine, ip
        chunk_size=256,
        chunk_overlap=10,
        max_batch_size=512,
        batch_size="auto",
        persist_directory=persist_dir,
    )


if __name__ == "__main__":
    start_time = time.perf_counter()
    attack_prompts = load_prompts(is_attack=True)
    benign_prompts = load_prompts(is_attack=False)
    write_db(attack_prompts, ATTACK_DB_DIR)
    write_db(benign_prompts, BENIGN_DB_DIR)
    end_time = time.perf_counter()

    inference_time = end_time - start_time
    max_mem = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0.0
    print(f"Building all DBs took {inference_time:.6f} seconds")

    with open("ragpref_build_db_time_mem.txt", "w") as f:
        f.write(f"{inference_time}\n")
        f.write(f"{max_mem}\n")
