from typing import List
import json
import submitit
import braceexpand
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def expand_paths(sharded_filepaths: str):
    # Replace '(' and '[' with '{'
    brace_keys_open = ['(', '[', '<', '_OP_']
    for bkey in brace_keys_open:
        if bkey in sharded_filepaths:
            sharded_filepaths = sharded_filepaths.replace(bkey, "{")

    # Replace ')' and ']' with '}'
    brace_keys_close = [')', ']', '>', '_CL_']
    for bkey in brace_keys_close:
        if bkey in sharded_filepaths:
            sharded_filepaths = sharded_filepaths.replace(bkey, "}")

    # Brace expand, set escape=False for Windows compatibility
    sharded_filepaths = list(braceexpand.braceexpand(sharded_filepaths, escape=False))
    return sharded_filepaths


def remove_special_tokens(input: str, tokenizer):
    tokens_to_remove = [
        tokenizer.pad_token,
        tokenizer.eos_token,
        "<end_of_turn>\n",
        "<end_of_turn>",
    ]
    for tok in tokens_to_remove:
        input = input.replace(tok, "")
    return input


def generate_data(
    model_name: str,    # hf model name or path
    manifests: List[str],   # list of paths to manifests
    out_dir: str,   # root dir to write the output
    user_prompt: str,
    max_new_tokens: int,
    batch_size: int,
    path_prefix: str,   # this will be replaced with out_dir
):
    """Generates question and answer using transcripts, and save in SQA format."""

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="cuda",
        torch_dtype=torch.bfloat16,
    )

    for manifest in manifests:
        with open(manifest, 'r') as f:
            samples = [json.loads(line) for line in f]
        assert manifest.startswith(path_prefix), manifest
        out_file = Path(manifest.replace(path_prefix, out_dir))
        out_file.parent.mkdir(parents=True, exist_ok=True)

        fout = open(out_file, 'w')

        for idx in range(0, len(samples), batch_size):
            batch_samples = samples[idx : idx + batch_size]
            batch_prompts = [
                tokenizer.apply_chat_template(
                    [
                        {"role": "user", "content": f"{user_prompt}{x['answer']}"},
                    ],
                    tokenize=False,
                    add_generation_prompt=True
                ) for x in batch_samples
            ]

            try:
                inputs = tokenizer(batch_prompts, add_special_tokens=False, return_tensors="pt", padding=True).to(model.device)
                outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
                results = tokenizer.batch_decode(outputs)   # list of string containing special tokens
            except:
                continue

            for cur_sample, cur_prompt, cur_result in zip(batch_samples, batch_prompts, results):
                cur_result = cur_result.replace(tokenizer.pad_token, "")
                assert cur_result.startswith(cur_prompt), cur_result
                model_response = remove_special_tokens(cur_result.removeprefix(cur_prompt), tokenizer)
                try:
                    model_response_dict = json.loads(model_response)
                except:
                    print(model_response)
                    continue

                if ("question" in model_response_dict 
                    and "answer" in model_response_dict 
                    and model_response_dict['question'].lower() != "none" 
                    and model_response_dict['answer'].lower() != "none"
                ):
                    output = {
                        "audio_filepath": cur_sample["audio_filepath"],
                        "duration": cur_sample["duration"],
                        "shard_id": cur_sample["shard_id"],
                        "target_lang": cur_sample["target_lang"],
                        "original_text": cur_sample['answer'],
                        "question": model_response_dict['question'],
                        "answer": model_response_dict['answer'],
                    }
                    fout.write(json.dumps(output) + "\n")

        fout.close()


if __name__ == "__main__":
    model_name = "google/gemma-2-27b-it"
    batch_size = 8
    user_prompt = """I will provide you with several sentences. Please generate **one** question that is closely related to the content of these sentences, along with a corresponding answer. Ensure that your answer is **accurate** and clearly stated. Write your output in a single line in json format:
{"question": "xxx", "answer": "xxx"}
If the question and answer contain a double quote, insert backslash before it to ensure the output can be loaded by python library `json.loads()`. Do not add unnecessary backslash for symbols like dollar $, ampersand &, etc.
However, if the sentences are meaningless, please return **none** in those fields.

Here are the sentences:
"""
    max_new_tokens = 512
    path_prefix = "/path/to/your/data"
    out_dir = "/path/to/your/output"
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    grouped_manifests_and_maxfiles = [
        (
            [
                "/path/to/your/sharded_manifests/manifest__OP_0..511_CL_.json",
            ], 16
        ),
    ]

    for all_manifests, files_per_job in grouped_manifests_and_maxfiles:
        expanded_manifests = []
        for p in all_manifests:
            expanded_manifests.extend(expand_paths(p))

        print(f"Total manifest files: {len(expanded_manifests)}, files per job: {files_per_job}")

        log_folder = "slurm_logs/%j"
        executor = submitit.SlurmExecutor(folder=log_folder)
        executor.update_parameters(
            job_name="data-generation",
            time="24:00:00",
            mem="200G",
            nodes=1,
            cpus_per_task=16,
            gpus_per_node=1,
            ntasks_per_node=1,
            partition="your_partition",
            array_parallelism=256,
            account="your_account",
        )

        jobs = []
        with executor.batch():
            for idx in range(0, len(expanded_manifests), files_per_job):
                job = executor.submit(
                    generate_data,
                    model_name,
                    expanded_manifests[idx:idx+files_per_job],
                    out_dir,
                    user_prompt,
                    max_new_tokens,
                    batch_size,
                    path_prefix
                )
                jobs.append(job)

    # for job in jobs:
    #     print(job.result())
