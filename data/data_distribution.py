import modal
from modal import App, Image, Secret, Volume
from pathlib import Path
import structlog


sabiyarn = Image.debian_slim(python_version="3.12").pip_install(
    "transformers[torch]==4.41.2",
    "triton",
    "bitsandbytes",
    "datasets",
    "wandb",
    "matplotlib",
    "structlog",
    "PyYAML",
    "simple-parsing==0.0.3rc1",
    "sentencepiece",
    "scikit-learn",
    force_build=False,
)

VOL_MOUNT_PATH = Path("/vol")
stub = App(name="sabiyarn-data-distribution", image=sabiyarn)
output_vol = Volume.from_name("sabiyarn_data_dist", create_if_missing=True)

restart_tracker_dict = modal.Dict.from_name(
    "sabiyarn-data-dist", create_if_missing=True
)


def track_restarts(restart_tracker: modal.Dict) -> int:
    if not restart_tracker.contains("count"):
        preemption_count = 0
        print(f"Starting first time. {preemption_count=}")
        restart_tracker["count"] = preemption_count
    else:
        preemption_count = restart_tracker.get("count") + 1
        print(f"Restarting after pre-emption. {preemption_count=}")
        restart_tracker["count"] = preemption_count
    return preemption_count


def prepare_env(vol=output_vol):
    from datasets import load_dataset, concatenate_datasets
    from huggingface_hub import login
    from huggingface_hub import HfApi
    from datasets import DatasetDict
    import re
    import unicodedata
    import pandas as pd
    from tqdm import tqdm
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from transformers import AutoModelForCausalLM
    from transformers import GenerationConfig
    import pandas as pd
    from tqdm import tqdm
    import os, json, re
    import matplotlib.pyplot as plt
    import torch
    import gc

    read_token = os.environ.get("HF_API_KEY") or os.environ.get("HF_TOKEN")
    write_token = os.environ.get("HF_WRITE_TOKEN") or read_token
    clone_repo_token = os.environ.get("HF_CLONE_TOKEN") or read_token

    login(token=clone_repo_token)
    api = HfApi()

    OUTPUT_PATH = "/vol/dataset_distribution.json"
    PROCESSED_LOG = "/vol/processed_files.log"
    LANG_UNKNOWN_LOG = "/vol/unidentified_languages.log"

    tokenizer = AutoTokenizer.from_pretrained(
        "BeardedMonster/SabiYarn-125M", token=read_token
    )
    repo_name = "BeardedMonster/Sabiyarn_language_detection"
    model = AutoModelForCausalLM.from_pretrained(
        repo_name, trust_remote_code=True, token=read_token
    )

    abb = {
        "yor": "yoruba",
        "urhobo": "urhobo",
        "ibo": "igbo",
        "fuv": "fulfulde",
        "fulani": "fulah",
        "ful": "fulah",
        "fula": "fulah",
        "efik": "efik",
        "eng": "english",
        "igbo": "igbo",
        "yoruba": "yoruba",
        "pcm": "pidgin",
        "fulfulde": "fulfulde",
        "english": "english",
        "pidgin": "pidgin",
        "hausa": "hausa",
        "efi": "efik",
    }

    # Language special tags
    LANG_TAGS = {
        "english": "<eng>",
        "yoruba": "<yor>",
        "hausa": "<hau>",
        "igbo": "<ibo>",
        "fuv": "<fuv>",
        "ful": "<ful>",
        "fula": "<ful>",
        "fulani": "<ful>",
        "urhobo": "<urh>",
        "fulah": "<ful>",
        "fulfulde": "<fuv>",
        "efik": "<efik>",
        "pidgin": "<pcm>",
        "fulfulde_": "<ff>",
        "yor": "<yor>",
        "ibo": "<ibo>",
        "fuv": "<fuv>",
        "eng": "<eng>",
        "urh": "<urh>",
        "pcm": "<pcm>",
        "efi": "<efi>",
        "hau": "<hau>",
        "ff": "<fuv>",
        "unknown": "<unk>",
    }

    LANG_TAGS_TO_NAMES = {v: k for k, v in LANG_TAGS.items()}

    languages = [
        "english",
        "yoruba",
        "hausa",
        "igbo",
        "urhobo",
        "fulah",
        "fulfulde",
        "efik",
        "pidgin",
        "unknown",
    ]

    # 'instruction', 'diacritics', 'noise', 'title', 'sentiment', 'monlingual','prompt', 'qa','ner', 'translation','summary', 'noise', 'scraped', 'tabular','language_identification',

    TASK_TAGS = {
        "<translate>": "translation",
        "<classify>": "classification",
        "<sentiment>": "sentiment classification",
        "<topic>": "topic classification",
        "<prompt>": "instruction following/reasoning",
        "<diacritize>": "text diacritization",
        "<clean>": "text cleaning",
        "<summarize>": "text summarization",
        "<NER>": "named entity recognition",
        "<headline>": "headline generation",
        "<qa>": "question answering",
        "<title>": "headline generation ",
        "<lang_ID>": "language identification",
        "instruction": "instruction following/reasoning",
        "diacritics": "text diacritization",
        "noise": "text cleaning",
        "title": "headline generation",
        "sentiment": "sentiment classification",
        "monolingual": "monolingual",
        "prompt": "instruction following/reasoning",
        "ner": "named entity recognition",
        "translation": "translation",
        "summary": "text summarization",
        "scraped": "monolingual",
        "tabular": "monolingual",
        "language_identification": "language_identification",
        "Enendu igboanugo": "monolingual",
    }

    # multilingual_data = [
    #     'prompting_new_-0.parquet',
    # 'prompting_new_-2.parquet',
    # 'reasoning_new_instruction2__.parquet',
    #     'tabular_data.parquet',
    #     'multi_reasoning_new_instruction2__.parquet',
    # 'multilingual_translation.parquet',
    # 'new_monolingual__-0.parquet',
    # 'new_monolingual__-1.parquet',
    # 'new_monolingual__-2.parquet',
    # 'new_monolingual__-3.parquet',
    # 'new_monolingual__-4.parquet',
    # 'new_monolingual__-5.parquet',
    # 'new_monolingual__-6.parquet',
    # 'new_monolingual__-8.parquet',
    # 'new_multilingual_translation2-0.parquet',
    # 'new_multilingual_translation2-1.parquet',
    # 'new_multilingual_translation2-2.parquet',
    # 'new_multilingual_translation2-3.parquet',
    # 'new_multilingual_translation2-4.parquet',
    # 'new_multilingual_translation2-6.parquet',
    # 'new_prompting.parquet',
    # 'new_summary-0.parquet',
    # 'new_summary-2.parquet',
    #     'introduction_new_instruction2__.parquet',
    #     'cohere_new_multilingual_instruct2-0.parquet',
    # 'cohere_new_multilingual_instruct2-1.parquet',
    # 'cohere_new_multilingual_instruct2-2.parquet',
    # 'cohere_new_multilingual_instruct2-3.parquet',
    # 'cohere_new_multilingual_instruct2-4.parquet',
    # 'cohere_new_multilingual_instruct2-5.parquet',
    # 'cohere_new_multilingual_instruct2-6.parquet',
    # 'cohere_new_multilingual_instruct2-8.parquet',
    #     "language_identification.parquet"
    # ]

    diacritics = [
        "diacritics_-0.parquet",
        "diacritics_-1.parquet",
        "diacritics_-10.parquet",
        "diacritics_-11.parquet",
        "diacritics_-12.parquet",
        "diacritics_-14.parquet",
        "diacritics_-2.parquet",
        "diacritics_-3.parquet",
        "diacritics_-4.parquet",
        "diacritics_-5.parquet",
        "diacritics_-6.parquet",
        "diacritics_-7.parquet",
        "diacritics_-8.parquet",
        "diacritics_-9.parquet",
    ]

    noise_data = [
        "language_noise-0.parquet",
        "language_noise-1.parquet",
        "language_noise-10.parquet",
        "language_noise-11.parquet",
        "language_noise-12.parquet",
        "language_noise-13.parquet",
        "language_noise-14.parquet",
        "language_noise-15.parquet",
        "language_noise-16.parquet",
        "language_noise-17.parquet",
        "language_noise-18.parquet",
        "language_noise-19.parquet",
        "language_noise-2.parquet",
        "language_noise-20.parquet",
        "language_noise-21.parquet",
        "language_noise-22.parquet",
        "language_noise-23.parquet",
        "language_noise-24.parquet",
        "language_noise-25.parquet",
        "language_noise-26.parquet",
        "language_noise-27.parquet",
        "language_noise-28.parquet",
        "language_noise-29.parquet",
        "language_noise-3.parquet",
        "language_noise-30.parquet",
        "language_noise-31.parquet",
        "language_noise-32.parquet",
        "language_noise-33.parquet",
        "language_noise-34.parquet",
        "language_noise-35.parquet",
        "language_noise-36.parquet",
        "language_noise-37.parquet",
        "language_noise-38.parquet",
        "language_noise-39.parquet",
        "language_noise-4.parquet",
        "language_noise-40.parquet",
        "language_noise-41.parquet",
        "language_noise-42.parquet",
        "language_noise-43.parquet",
        "language_noise-44.parquet",
        "language_noise-45.parquet",
        "language_noise-46.parquet",
        "language_noise-47.parquet",
        "language_noise-48.parquet",
        "language_noise-49.parquet",
        "language_noise-5.parquet",
        "language_noise-50.parquet",
        "language_noise-52.parquet",
        "language_noise-6.parquet",
        "language_noise-7.parquet",
        "language_noise-8.parquet",
        "language_noise-9.parquet",
    ]

    # others = ["hausa_new_monolingual-3.parquet","ibo_new_monolingual-3.parquet",
    #                 "igbo_monolingual1-1.parquet", "igbo_monolingual1.parquet", "igbo_monolingual-2.parquet", "igbo_monolingual-3.parquet", "igbo_monolingual-0.parquet", "igbo_monolingual-1.parquet",  "english_new_monolingual-4.parquet",
    #                 "newly_processed_english-0.parquet","newly_processed_english-1.parquet","newly_processed_english-2.parquet","newly_processed_english-3.parquet",
    #             "newly_processed_english-4.parquet","newly_processed_english-5.parquet", "newly_processed_english-6.parquet","yoruba_monolingual (1)-0.parquet",
    #             "newly_processed_english-8.parquet", "newly_processed_english-7.parquet", "yoruba_new_monolingual-1.parquet","yoruba_new_qa-3.parquet"]

    language_files = diacritics + noise_data

    LANG_TAGS_TO_NAMES = {v: k for k, v in LANG_TAGS.items()}

    languages = [
        "english",
        "yoruba",
        "hausa",
        "igbo",
        "urhobo",
        "fulah",
        "fulfulde",
        "efik",
        "pidgin",
        "unknown",
    ]
    TASK_TAGS = {
        "<translate>": "translation",
        "<classify>": "classification",
        "<sentiment>": "sentiment classification",
        "<topic>": "topic classification",
        "<prompt>": "instruction following/reasoning",
        "<diacritize>": "text diacritization",
        "<clean>": "text cleaning",
        "<summarize>": "text summarization",
        "<NER>": "named entity recognition",
        "<headline>": "headline generation",
        "<qa>": "question answering",
        "<title>": "headline generation ",
        "<lang_ID>": "language identification",
        "instruction": "instruction following/reasoning",
        "diacritics": "text diacritization",
        "noise": "text cleaning",
        "title": "headline generation",
        "sentiment": "sentiment classification",
        "monolingual": "monolingual",
        "prompt": "instruction following/reasoning",
        "ner": "named entity recognition",
        "translation": "translation",
        "summary": "text summarization",
        "scraped": "monolingual",
        "tabular": "monolingual",
        "language_identification": "language_identification",
        "Enendu igboanugo": "monolingual",
    }

    generation_config = GenerationConfig(
        max_length=100,  # Maximum length of the generated sequence
        num_beams=10,  # Number of beams for beam search
        do_sample=False,  # Whether to use sampling instead of greedy decoding
        temperature=0.9,  # Sampling temperature
        top_k=50,  # Top-K sampling
        top_p=0.95,  # Top-P (nucleus) sampling
        repetition_penalty=2.0,  # Repetition penalty to reduce repetitive outputs
        length_penalty=1.7,  # Length penalty to favor longer sequences
        early_stopping=True,  # Stop early when all beams have finished
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    def clean_text(text):

        if text is None:
            return ""  # Return empty string if text is None

        # Handle dictionaries, lists, and tuples
        if isinstance(text, dict):
            text = " ".join([f"{k}. {v}" for k, v in text.items()])
        elif isinstance(text, (list, tuple)):
            text = " ".join(map(str, text))  #

        # Normalize Unicode characters
        text = unicodedata.normalize("NFKC", str(text))

        # Remove HTML tags using BeautifulSoup, use 'lxml' parser if available
        try:
            text = BeautifulSoup(text, "lxml").get_text()  # Try using 'lxml' parser
        except:
            text = BeautifulSoup(
                text, "html.parser"
            ).get_text()  # Fallback to 'html.parser'

        # Remove URLs (http, https, ftp, etc.)
        # text = re.sub(r'http[s]?://\S+|www\.\S+', '', text)

        # Remove email addresses
        # text = re.sub(r'\S+@\S+\.\S+', '', text)

        # Replace or remove unreadable characters
        # text = re.sub(r'[\u0000-\u001F\u007F-\u009F]|INST|\u200b|#VALUE!|nan', '', text)  # Control characters

        # Replace or remove unreadable characters
        text = re.sub(
            r"\u200b", "", text
        )  # Control characters [\u0000-\u001F\u007F-\u009F]|INST|

        # text = re.sub('\[INST\]','<prompt> ', text)
        # text = re.sub('\[/INST\]','<response>: ', text)
        # text = re.sub('<<SYS>>|<</SYS>>|<s>|</s>','', text)

        # Remove excessive repetitions of non-numeric characters (limit to 3)
        # text = re.sub(r'([^0-9])\1{3,}', r'\1\1\1', text)

        # Remove excessive repetitions of words (limit to 3 repetitions)
        # text = re.sub(r'(\b\w+\b)(?:\s+\1){3,}', r'\1 \1 \1', text)

        # Remove special or unwanted symbols (except essential punctuation and linguistic symbols)
        text = re.sub(r"[^\w\s,.!?;:\-\u00C0-\u024F\u1E00-\u1EFF\n]", "", text)

        # Normalize and collapse multiple spaces into a single space
        # text = re.sub(r'[\r\f]+', ' ', text).strip()
        # text = re.sub(r' {5,}', '    ', text)

        # Remove excessive newlines (limit to 2 consecutive newlines)
        # text = re.sub(r'\n{3,}', '\n\n', text)

        return text

    def extract_file_info(file_name):
        # Try get language from name
        lang = next((abb[k] for k in LANG_TAGS if k in file_name.lower()), None)
        task = next(
            (
                v
                for k, v in TASK_TAGS.items()
                if k.replace("<", "").replace(">", "") in file_name.lower()
            ),
            None,
        )
        return lang, task

    def count_tokens(texts):
        tokens = tokenizer(texts, return_length=True, truncation=False, padding=False)
        return sum(tokens["length"])

    def count_tokens_in_chunks(texts, chunk_size, reduce_length=True):
        if reduce_length:
            total_count = 0
        else:
            total_count = []

        for each in texts:
            if not reduce_length:
                c = 0
            for j in range(0, len(each), chunk_size):
                chunk = each[j : j + chunk_size]
                tokenized = tokenizer(
                    chunk, return_length=True, truncation=False, padding=False
                )
                lengths = (
                    tokenized["length"]
                    if "length" in tokenized
                    else [len(enc) for enc in tokenized["input_ids"]]
                )
                c += sum(lengths)

            if reduce_length:
                total_count += c
            else:
                total_count.append(c)

        return total_count

    def count_tokens(
        texts,
        batch_size=100,
        chunk_size=50000,
        chunk_each_sample=False,
        reduce_length=True,
    ):
        """
        Count the total number of tokens in a list of texts or a single large text using a memory-efficient approach.
        """
        if reduce_length:
            total_tokens = 0
        else:
            total_tokens = []

        if isinstance(texts, str):
            texts = [texts]  # wrap single string in list

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            if chunk_each_sample:
                if reduce_length:
                    total_tokens += count_tokens_in_chunks(
                        batch, chunk_size, reduce_length
                    )
                else:
                    total_tokens.extend(
                        count_tokens_in_chunks(batch, chunk_size, reduce_length)
                    )
            else:
                tokenized = tokenizer(
                    batch, return_length=True, truncation=False, padding=False
                )
                # Handle tokenizer output shape (some return List, some Dict with batch encoding)
                lengths = (
                    tokenized["length"]
                    if "length" in tokenized
                    else [int(len(enc)) for enc in tokenized["input_ids"]]
                )
                if reduce_length:
                    total_tokens += sum(lengths)
                    int(total_tokens)
                else:
                    total_tokens.extend(lengths)

        return total_tokens

    def split_at_language_tokens(text):
        pattern = r"<(?:yor|urh|ibo|efi|hau|eng|pcm|ff|fuv|ful)>"
        parts = re.split(pattern, text)
        return parts[-1]  # Return the last segment

    SPECIAL_TASK_TAGS = {
        "<translate>": "translation",
        "<sentiment>": "sentiment classification",
        "<topic>": "topic classification",
        "<prompt>": "instruction following/reasoning",
        "<diacritize>": "text diacritization",
        "<clean>": "text cleaning",
        "<summarize>": "text summarization",
        "<NER>": "named entity recognition",
        "<headline>": "headline generation",
        "<qa>": "question answering",
        "<title>": "headline generation",
        "<lang_ID>": "language identification",
    }

    def detect_tasks_per_sample(texts, tasks):
        """
        Detect task for each text sample by looking for known special tags.
        If no tag is found, fall back to the task in the same index in `tasks`.

        Args:
            texts (List[str]): List of sample texts
            tasks (List[str]): List of fallback tasks (same length)

        Returns:
            List[str]: List of detected tasks for each sample
        """
        detected_tasks = []
        for idx, text in enumerate(texts):
            task_found = None
            for tag, name in SPECIAL_TASK_TAGS.items():
                if tag in text:
                    task_found = name
                    break

            # If no task tag is found, use the provided fallback task
            if not task_found:
                task_found = tasks[idx]

            detected_tasks.append(task_found)

        return detected_tasks

    def prepare(text, num_chars=80):
        text = re.split(r"<(?:yor|urh|ibo|efi|hau|eng|pcm|ff|fuv|ful)>", text)[-1]
        text = re.sub(
            r"<(?:lang_ID|lang_ID_label|classify|translate|diacritize|clean|prompt|NER|qa|)>",
            "",
            text,
        )
        return text[:num_chars].strip()

    def detect_lang_batch(
        sample_texts, allow_cpu=False, batch_size=8, max_new_tokens=2, num_chars=20
    ):
        """
        Detects language for multiple texts using the SabiYarn-125M model.

        Args:
            sample_texts (List[str]): A list of input texts to classify.
            allow_cpu (bool): Whether to allow running on CPU if CUDA is not available.
            batch_size (int): Number of samples per batch for processing.
            max_new_tokens (int): Max number of new tokens to generate for prediction.

        Returns:
            List[str or None]: List of predicted language tags or <unk> if failed.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        tokenizer.padding_side = "left"

        predictions = []

        try:
            for i in tqdm(
                range(0, len(sample_texts), batch_size),
                desc=f"Processing samples using batch size of {batch_size}",
            ):
                batch = sample_texts["text"][i : i + batch_size]
                prompts = [
                    f"<lang_ID> {prepare(text, num_chars)} <lang_ID_label>: "
                    for text in batch
                ]

                encoded = tokenizer(
                    prompts, return_tensors="pt", padding=True, truncation=False
                ).to(device)

                with torch.no_grad():
                    outputs = model.generate(
                        encoded["input_ids"],
                        generation_config=generation_config,
                        max_new_tokens=max_new_tokens,
                    )

                input_ids = encoded["input_ids"]

                for j in range(len(batch)):
                    generated_ids = outputs[j][len(input_ids[j]) :]
                    decoded = (
                        tokenizer.decode(generated_ids, skip_special_tokens=False)
                        .replace("|end_of_text|", "")
                        .strip()
                    )
                    pred_lang = LANG_TAGS_TO_NAMES.get(decoded, "<unk>")
                    if pred_lang == "<unk>":
                        print(
                            f"⚠️ Failed to detect language for sample {i+j}: Decoded: '{decoded}'"
                        )
                    predictions.append(pred_lang)

                # Free up GPU memory for this batch
                del encoded, outputs, input_ids, batch, prompts
                torch.cuda.empty_cache()
                gc.collect()

        except Exception as e:
            print(f"⚠️ Language batch detection failed: {e}")
            predictions.extend([None] * (len(sample_texts) - len(predictions)))

        return predictions

    def update_distribution(
        file_name, lang, tasks, num_samples, num_tokens, stats_dict
    ):  # num_tokens_per_sample,
        # print('yes')
        for i, task in enumerate(tasks):
            stats_dict["samples_per_task"].setdefault(task, 0)
            # stats_dict["samples_per_task"][task] += num_tokens_per_sample[i]
            stats_dict["tokens_per_task"].setdefault(task, 0)
            stats_dict["tokens_per_task"][task] += 1

            stats_dict["samples_tokens_per_lang_task"].setdefault(task, {})

        for each in lang:
            stats_dict["samples_per_language"][each] += 1
            stats_dict["tokens_per_language"][each] += num_tokens[each]
            stats_dict["samples_tokens_per_lang_task"][task].setdefault(
                each, {"samples": 0, "tokens": 0}
            )

            stats_dict["samples_tokens_per_lang_task"][task][each]["samples"] += 1
            stats_dict["samples_tokens_per_lang_task"][task][each][
                "tokens"
            ] += num_tokens[each]

        stats_dict["processed_files"].append(file_name)
        return stats_dict

    def analyze_repo(
        hf_repo_id,
        files,
        allow_cpu=False,
        tokenizer_batch_size=100,
        model_batch_size=4000,
        chunk_size=50000,
        chunk_each_sample=False,
        max_new_tokens=2,
    ):
        if os.path.exists(OUTPUT_PATH):
            with open(OUTPUT_PATH, "r") as f:
                stats = json.load(f)
        else:
            with open(OUTPUT_PATH, "w") as f:
                stats = {
                    "samples_per_language": {l: 0 for l in languages},
                    "tokens_per_language": {l: 0 for l in languages},
                    "samples_per_task": {},
                    "samples_tokens_per_lang_task": {},
                    "tokens_per_task": {},
                    "samples_per_file": {},
                    "processed_files": [],
                }
                json.dump(stats, f)

        with open(PROCESSED_LOG, "a+") as proc_log, open(
            LANG_UNKNOWN_LOG, "a+"
        ) as lang_log:

            for file in tqdm(files, desc="Processing files"):
                print(f"file being processed: {file}")
                if file in stats["processed_files"]:
                    continue

                try:
                    ds = load_dataset(
                        hf_repo_id, data_files=file, split="train", token=read_token
                    )
                    # sample_texts = ds["text"]

                    lang, task = extract_file_info(file)
                    tasks = [task] * len(ds["text"])
                    tasks = detect_tasks_per_sample(ds["text"], tasks)
                    # num_tokens_per_sample = count_tokens([sample], chunk_size) for sample in ds['text']]
                    # num_tokens_per_sample = [count_tokens(sample, chunk_size) for sample in ds['text']]

                    print("lang: ", lang)
                    if lang is not None:
                        lang_num_samples = {lang: len(ds["text"])}
                        lang_tokens = {
                            lang: count_tokens(
                                ds["text"],
                                tokenizer_batch_size,
                                chunk_size,
                                chunk_each_sample,
                            )
                        }
                        detected_langs = [lang] * len(ds["text"])  # Fixed this line

                    else:
                        print(
                            "Unable to detect languages manually, loading model for intelligent detection..."
                        )
                        detected_langs = detect_lang_batch(
                            ds,
                            allow_cpu=allow_cpu,
                            batch_size=model_batch_size,
                            max_new_tokens=max_new_tokens,
                        )
                        lang_tokens = {l: 0 for l in set(detected_langs)}
                        token_counts = count_tokens(
                            ds["text"],
                            tokenizer_batch_size,
                            chunk_size,
                            chunk_each_sample,
                            reduce_length=False,
                        )
                        lang_num_samples = {l: 0 for l in set(detected_langs)}
                        for l, num_toks in zip(detected_langs, token_counts):
                            lang_tokens[l] += num_toks
                            lang_num_samples[l] += 1

                    stats = update_distribution(
                        file,
                        detected_langs,
                        tasks,
                        lang_num_samples,
                        lang_tokens,
                        stats,
                    )  # num_tokens_per_sample,

                    # Save and cleanup
                    with open(OUTPUT_PATH, "w") as f:
                        json.dump(stats, f, indent=2)
                    proc_log.write(file + "\n")
                    vol.commit()
                    del ds
                    gc.collect()

                except Exception as e:
                    print(f"❌ Error detected in {file}: {e}")

                    # stats = update_distribution(file, lang, task, num_samples, tokens, stats)

                    # Save and cleanup
                    # with open(OUTPUT_PATH, "w") as f:
                    #     json.dump(stats, f, indent=2)
                    proc_log.write(file + "\n")
                    del ds
                    gc.collect()

                # except Exception as e:
                #     print(f"❌ Error in {file}: {e}")

    def plot_distribution(stats_path=OUTPUT_PATH):
        with open(stats_path, "r") as f:
            data = json.load(f)

        # Language Sample Distribution
        plt.figure(figsize=(10, 6))
        plt.bar(
            data["samples_per_language"].keys(), data["samples_per_language"].values()
        )
        plt.title("Sample Count per Language")
        plt.xlabel("Language")
        plt.ylabel("Number of Samples")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig("/vol/plots/sample_count_per_language.png")  # Save the plot
        plt.close()  # Close the plot to free memory

        # Token Count per Language
        plt.figure(figsize=(10, 6))
        plt.bar(
            data["tokens_per_language"].keys(),
            data["tokens_per_language"].values(),
            color="orange",
        )
        plt.title("Token Count per Language")
        plt.xlabel("Language")
        plt.ylabel("Number of Tokens")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig("/vol/plots/token_count_per_language.png")  # Save
        plt.close()  # Close

        # Task Distribution
        plt.figure(figsize=(10, 6))
        plt.pie(
            data["samples_per_task"].values(),
            labels=data["samples_per_task"].keys(),
            autopct="%1.1f%%",
            startangle=90,
        )
        plt.title("Task Distribution")
        plt.axis("equal")
        plt.tight_layout()
        plt.savefig("/vol/plots/task_distribution.png")  # Save
        plt.close()  # Close

        # Sample and Token distribution per task and language
        for task, lang_data in data["samples_tokens_per_lang_task"].items():
            samples = [lang_data[lang]["samples"] for lang in lang_data]
            tokens = [lang_data[lang]["tokens"] for lang in lang_data]
            langs = list(lang_data.keys())

            plt.figure(figsize=(12, 6))

            plt.subplot(1, 2, 1)
            plt.bar(langs, samples)
            plt.title(f"Sample Distribution for {task}")
            plt.xlabel("Language")
            plt.ylabel("Number of Samples")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()

            plt.subplot(1, 2, 2)
            plt.bar(langs, tokens, color="orange")
            plt.title(f"Token Distribution for {task}")
            plt.xlabel("Language")
            plt.ylabel("Number of Tokens")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.savefig(f"/vol/plots/distribution_for_{task}.png")  # Save
            plt.close()  # Close

    analyze_repo(
        "Aletheia-ng/NLP-base-data",
        language_files,
        tokenizer_batch_size=5000,
        model_batch_size=1600,
        chunk_size=250000,
        chunk_each_sample=False,
        max_new_tokens=2,
    )

    vol.commit()

    plot_distribution()

    vol.commit()


@stub.function(
    gpu="A100",
    timeout=60 * 60 * 20,
    cpu=4.0,
    secrets=[Secret.from_name("wandb-api"), Secret.from_name("hf-secret")],
    volumes={VOL_MOUNT_PATH: output_vol},
    memory=32768,
)
def run(vol=output_vol):
    prepare_env()
    vol.commit()
