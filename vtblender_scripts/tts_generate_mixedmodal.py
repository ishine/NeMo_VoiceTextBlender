import json
import random
import numpy as np
import submitit
import torch
import nltk
import soundfile as sf
import librosa
from pathlib import Path

from nemo.collections.asr.parts.preprocessing.features import clean_spectrogram_batch, normalize_batch
from nemo.collections.tts.models import FastPitchModel, SpectrogramEnhancerModel
from nemo.core.classes import typecheck
from nemo.collections.tts.models import HifiGanModel
from tts_normalization_utils import get_normalizer, normalize


AUDIO_LOCATOR = "<unused0>"


def generate_audios(manifest, indices, outdir, shard_id):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    wavdir = outdir / "audios"
    wavdir.mkdir(parents=True, exist_ok=True)
    filedir = outdir / "manifests"
    filedir.mkdir(parents=True, exist_ok=True)

    normalize_type = "per_feature"
    do_normalize = True
    do_lowercase = False
    use_enhancer = False  # TODO: fix it. It has a bug!
    sample_rate = 44100
    new_sr = 16000  # save on 16kHz

    tts_model = FastPitchModel.from_pretrained("tts_en_fastpitch_multispeaker")
    tts_model.eval().cuda()

    vocoder = HifiGanModel.from_pretrained(model_name="tts_en_hifitts_hifigan_ft_fastpitch")
    vocoder.eval().cuda()
    n_speakers = tts_model.cfg.n_speakers

    if use_enhancer:
        enhancer_model = SpectrogramEnhancerModel.from_pretrained(
            model_name="tts_en_spectrogram_enhancer_for_asr_finetuning"
        )
        enhancer_model.eval().cuda()
    else:
        enhancer_model = None

    if do_normalize:
        normalizer = get_normalizer()
    else:
        normalizer = None


    def generate_spec(
        tts_model, vocoder, text, normalizer=None, do_lowercase=False, enhancer_model=None, output_file_path=None
    ):
        original_len = len(text)
        if normalizer:
            text = normalize(text=text, normalizer=normalizer, do_lowercase=do_lowercase)

        src_ids = tts_model.parse(text, normalize=True)  # alternative tts_model.vocab.encode(text)

        with torch.no_grad():
            speaker_id = random.randint(0, n_speakers - 1)
            speaker_id = torch.tensor([speaker_id]).to(src_ids.device)
            spectrogram = tts_model.generate_spectrogram(tokens=src_ids, speaker=speaker_id)
            audio = vocoder.convert_spectrogram_to_audio(spec=spectrogram)
            duration = audio.shape[1] / sample_rate

            sf.write(
                output_file_path,
                librosa.resample(np.ravel(audio.to('cpu').numpy()), orig_sr=sample_rate, target_sr=new_sr),
                new_sr,
                format='WAV'
            )
        return audio, duration


    with open(manifest, 'r') as fin, open(filedir / f"manifest_{shard_id}.jsonl", 'w') as fout:
        for idx, line in enumerate(fin):
            if idx in indices:
                sample = json.loads(line)

                # tokenize user input into sentences
                user_sents = nltk.sent_tokenize(sample["instruction"])
                n_samples_to_gen = min(len(user_sents), 2)

                for suffix in range(n_samples_to_gen):
                    begin_pos = random.randint(0, len(user_sents) - 1)
                    end_pos = random.randint(begin_pos + 1, len(user_sents))

                    cur_user = user_sents[:begin_pos] + [AUDIO_LOCATOR] + user_sents[end_pos:]
                    cur_user = " ".join(cur_user)

                    wav_path = wavdir / f"{sample['uuid']}_{suffix}.wav"

                    try:
                        audio, duration = generate_spec(
                            tts_model,
                            vocoder,
                            text=" ".join(user_sents[begin_pos:end_pos]),
                            normalizer=normalizer,
                            do_lowercase=do_lowercase,
                            enhancer_model=enhancer_model,
                            output_file_path=wav_path,
                        )
                        new_entry = {
                            "id": f"{sample['uuid']}_{suffix}",
                            "audio_filepath": str(wav_path.relative_to(outdir)),
                            "duration": duration,
                            "system": "",
                            "mask": "User",
                            "dataset": "tts-sft",
                            "conversations": [
                                {"from": "User", "value": cur_user, "canonical_form": "", "label": None},
                                {"from": "Assistant", "value": sample['response'], "canonical_form": "", "label": None}
                            ],
                            "answer": sample['response'],
                            "target_lang": "en",
                        }
                        fout.write(json.dumps(new_entry) + "\n")

                    except:
                        print(f"Failed: {sample}")


if __name__ == "__main__":
    ## Step 1: Filter HF data
    from datasets import load_dataset
    import json
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        "google/gemma-2b-it",
    )

    dataset = "Magpie-Align/Magpie-Gemma2-Pro-200K-Filtered"
    split = "train"
    # skip "Math", "Coding & Debugging"
    categories_to_keep = [
        "Brainstorming", "Planning", "Advice seeking", "Role playing", "Reasoning", "Data analysis", "Editing", "Creative writing", "Information seeking"
    ]
    max_instruction_tokens = 128
    max_response_tokens = 1024

    outdir = "/path/to/your/output"
    outdir = Path(outdir)
    outdir.mkdir(exist_ok=True, parents=True)

    ds = load_dataset(dataset, split=split)

    with open(outdir / "hf_data_filtered.jsonl", 'w') as fout:
        cnt = 0
        for sample in ds:
            if sample['task_category'] in categories_to_keep and sample['language'] in ['EN'] and len(tokenizer(sample['instruction']).input_ids) < max_instruction_tokens and len(tokenizer(sample['response']).input_ids) < max_response_tokens:
                fout.write(json.dumps(sample) + "\n")
                cnt += 1
    print(f"Remained: {cnt}")


    ## Step 2: TTS
    input_file = "/path/to/your/hf_data_filtered.jsonl"
    num_samples = 172171    # modify this according to your input file

    samples_per_job = 10000
    outdir = Path(input_file).parent

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
        array_parallelism=128,
        account="your_account",
    )

    jobs = []
    with executor.batch():
        for shard_id, idx in enumerate(range(0, num_samples, samples_per_job)):
            job = executor.submit(
                generate_audios,
                input_file,
                list(range(idx, idx + samples_per_job)),
                outdir,
                shard_id
            )
            jobs.append(job)

    # merge all manifests
    tot_dur = 0
    with open(outdir / "train_manifest.jsonl", 'w') as fout:
        for file in (outdir / "manifests").glob("manifest_*.jsonl"):
            with open(file, 'r') as fin:
                for line in fin:
                    entry = json.loads(line)
                    tot_dur += entry['duration']
                    fout.write(json.dumps(entry) + "\n")

    print(f"Total duration: {tot_dur} seconds = {tot_dur / 3600} hours")
