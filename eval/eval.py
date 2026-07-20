import numpy as np
import random
import torch
import re
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import MultiLabelBinarizer


from datasets import load_dataset
import json
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import GenerationConfig
import structlog
import os
from dotenv import load_dotenv

load_dotenv()

model_name = "BeardedMonster/SabiYarn-125M-finetune"
repo_name = "BeardedMonster/SabiYarn-125M"
device = "cuda"

LOG = structlog.stdlib.get_logger()

HF_TOKEN = os.environ.get("HF_API_KEY") or os.environ.get("HF_TOKEN")

model = AutoModelForCausalLM.from_pretrained(
    model_name, trust_remote_code=True, token=HF_TOKEN
).to(device)
tokenizer = AutoTokenizer.from_pretrained(repo_name, trust_remote_code=True)

max_new_tokens = 163
num_beams = 5
decode_config = {
    "beam_search": GenerationConfig(
        max_length=100,
        max_new_tokens=max_new_tokens,  # Maximum length of the generated sequence
        num_beams=num_beams,  # Number of beams for beam search
        temperature=1,
        do_sample=False,  # Sampling temperature
        #         top_k=20,                 # Top-P (nucleus) sampling
        #         repetition_penalty=4.0,    # Repetition penalty to reduce repetitive outputs
        length_penalty=3.0,  # Length penalty to favor longer sequences
        early_stopping=True,  # Stop early when all beams have finished
    ),
    "nucleus": GenerationConfig(
        max_length=100,
        max_new_tokens=max_new_tokens,  # Maximum length of the generated sequenc
        do_sample=True,  # Whether to use sampling instead of greedy decoding
        temperature=0.99,  # Sampling temperature
        top_k=50,  # Top-K sampling
        top_p=0.95,  # Top-P (nucleus) sampling
        repetition_penalty=4.0,  # Repetition penalty to reduce repetitive outputs
        #     length_penalty=3.0,        # Length penalty to favor longer sequences
        #     early_stopping=True        # Stop early when all beams have finished
    ),
    "greedy": GenerationConfig(
        max_length=100,
        max_new_tokens=max_new_tokens,  # Maximum length of the generated sequence
        num_beams=1,  # Number of beams for beam search
        do_sample=False,  # Whether to use sampling instead of greedy decoding
        temperature=1,  # Sampling temperature/
        # top_k=50,                  # Top-K sampling
        # top_p=0.95,                # Top-P (nucleus) sampling
        repetition_penalty=4.0,  # Repetition penalty to reduce repetitive outputs
        #     length_penalty=3.0,        # Length penalty to favor longer sequences
        #     early_stopping=True        # Stop early when all beams have finished
    ),
}

generation_config = GenerationConfig(
    max_new_tokens=50,  # Maximum length of the generated sequence
    num_beams=5,  # Number of beams for beam search
    do_sample=False,  # Whether to use sampling instead of greedy decoding             # Sampling temperature
    top_k=50,  # Top-K sampling
    top_p=0.95,  # Top-P (nucleus) sampling
    repetition_penalty=4.0,  # Repetition penalty to reduce repetitive outputs
    length_penalty=3.0,  # Length penalty to favor longer sequences
    early_stopping=True,  # Stop early when all beams have finished
)

popular_topics = [
    "Sports",
    "Entertainment",
    "Politics",
    "Travel",
    "Technology",
    "Health",
    "Business",
    "Science",
    "Education",
    "Lifestyle",
    "Culture",
    "Environment",
    "Finance",
    "Food",
    "Gaming",
    "History",
    "Law",
    "Literature",
    "Music",
    "News",
    "Africa",
    "Philosophy",
    "Religion",
    "Society",
    "World",
]

topic_class_labels = [
    "business",
    "entertainment",
    "health",
    "politics",
    "religion",
    "sports",
    "technology",
]


def extract_answer(text):
    pattern = r"[a-z][A-Z]"
    result = re.split(pattern, text)[0]
    result = text[: len(result) + 1]
    return result


def assign_topic(generated_text, topic_list=topic_class_labels):
    # generated_text = extract_answer(generated_text)
    lower_generated_text = generated_text.lower()
    for topic in topic_list:
        if topic.lower() in lower_generated_text:
            return topic.lower()
        elif "football" in lower_generated_text:
            return "sports"
        elif "music" in lower_generated_text:
            return "entertainment"
        elif (
            "africa" in lower_generated_text
            or "afrikaan" in lower_generated_text
            or "africa" in lower_generated_text
        ):
            return "politics"

    return "unknown"


def calculate_metrics(class_labels, predicted_labels):
    # Calculate metrics
    accuracy = accuracy_score(class_labels, predicted_labels)
    f1_macro = f1_score(class_labels, predicted_labels, average="macro")
    f1_micro = f1_score(class_labels, predicted_labels, average="micro")
    recall_macro = recall_score(class_labels, predicted_labels, average="macro")
    recall_micro = recall_score(class_labels, predicted_labels, average="micro")
    precision_macro = precision_score(class_labels, predicted_labels, average="macro")
    precision_micro = precision_score(class_labels, predicted_labels, average="micro")
    return {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_micro": f1_micro,
        "recall_macro": recall_macro,
        "recall_micro": recall_micro,
        "precision_macro": precision_macro,
        "precision_micro": precision_micro,
    }

    # Print results
    print(f"Accuracy: {accuracy}")
    print(f"F1 Score (Macro): {f1_macro}")
    print(f"F1 Score (Micro): {f1_micro}")
    print(f"Recall (Macro): {recall_macro}")
    print(f"Recall (Micro): {recall_micro}")
    print(f"Precision (Macro): {precision_macro}")
    print(f"Precision (Micro): {precision_micro}")


class_ind = {
    "business": 0,
    "entertainment": 1,
    "health": 2,
    "politics": 3,
    "religion": 4,
    "sports": 5,
    "technology": 6,
    "unknown": 7,
}


def generate(text, model, tokenizer, config):
    inputs = tokenizer(text, return_tensors="pt")["input_ids"].to(model.device)
    if len(inputs[0]) > 1024:
        print(len(inputs[0]))
        inputs = torch.cat([inputs[:, :100], inputs[:, -10:]], dim=1)
    outputs = model.generate(inputs, generation_config=config)
    return tokenizer.decode(
        outputs[0][len(inputs[0]) :],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )


def store_metrics(vol, metrics, task_name, lang):
    root_dir = "/vol"
    with open(f"{root_dir}/{task_name}_{lang}.txt", "w") as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")
    vol.commit()


def topic_classification(model, lang, vol):

    model.to("cuda")
    model.eval()
    dataset = load_dataset("masakhane/masakhanews", lang)["test"]

    true = []
    pred = []
    raw_text = []
    LOG.info("generating model outputs...")
    for count, data in enumerate(tqdm(dataset)):
        source_text = (
            f"<classify> {' '.join(data['headline_text'].split()[:128])} <topic> :"
        )

        try:
            with torch.no_grad():
                generated_text = generate(
                    source_text, model, tokenizer, config=decode_config["beam_search"]
                )
        except Exception as e:
            print(e)
            continue
        raw_text.append(generated_text)
        true.append(class_ind[topic_class_labels[int(data["label"])]])
        pred.append(class_ind[assign_topic(generated_text)])

    predicted_labels = [
        class_ind[assign_topic(extract_answer(text))] for text in raw_text
    ]
    LOG.info(f"storing metrics for {lang}")
    metrics = calculate_metrics(true, predicted_labels)
    store_metrics(vol=vol, metrics=metrics, lang=lang, task_name="topic_classification")


def sentiment_analysis(model, lang, vol):
    dataset = load_dataset("HausaNLP/AfriSenti-Twitter", lang, trust_remote_code=True)[
        "test"
    ]
    model.to("cuda")
    model.eval()

    true = []
    pred = []
    raw_text = []
    for count, data in enumerate(tqdm(dataset)):
        source_text = f"<classify> {data['tweet']} <sentiment> :"

        try:
            with torch.no_grad():
                generated_text = generate(
                    source_text,
                    model,
                    tokenizer,
                    config=decode_config["beam_search"],
                )
        except Exception as e:
            print(e)
            continue
        raw_text.append(generated_text)
        ans = generated_text.strip().split()[-1]
        pred.append(ans)

        true.append(data["label"])

    label_map = {"negative": 1, "positive": 0, "neutral": 2, "unknown": 3}
    predicted_labels = [label_map[each] for each in pred]
    metrics = calculate_metrics(true, predicted_labels)
    store_metrics(vol=vol, metrics=metrics, lang=lang, task_name="sentiment_analysis")


def NER(model, lang, vol):
    data = load_dataset("masakhaner", lang, trust_remote_code=True)
    test_data = data["test"]
    ner_tags = [
        "O",
        "B-PER",
        "I-PER",
        "B-ORG",
        "I-ORG",
        "B-LOC",
        "I-LOC",
        "B-DATE",
        "I-DATE",
    ]
    idx_to_tag = {idx: tag for idx, tag in enumerate(ner_tags)}

    def convert_idx_to_string(idx):
        tags = []
        for i in idx:
            tags.append(idx_to_tag[i])
        return tags

    true, pred = [], []

    def generate_metrics(dataset):
        model.eval()

        mlb = MultiLabelBinarizer()  # Initialize the MultiLabelBinarizer
        LOG.info(f"generating model outputs for Named Entity Recognition")
        with torch.no_grad():
            for count, data in enumerate(tqdm(dataset)):
                source_text = f"<NER> {' '.join(data['tokens'])} <tag> :"
                target_text = convert_idx_to_string(data["ner_tags"])

                try:
                    generated_text = generate(
                        source_text,
                        model,
                        tokenizer,
                        config=decode_config["beam_search"],
                    )
                    generated_tokens = generated_text.split(" ")

                    if len(generated_tokens) != len(target_text):
                        generated_tokens = generated_tokens[: len(target_text)]

                    # Collect true and predicted labels
                    pred.append(generated_tokens)
                    true.append(target_text)

                except Exception as e:
                    print(f"Failed with exception: {e}")

        # Convert true and predicted labels to binary format
        true_binary = mlb.fit_transform(true)  # Fit the true labels and transform
        pred_binary = mlb.transform(
            pred
        )  # Transform the predicted labels based on the same fit

        # Compute precision, recall, and F1-score
        metrics = precision_recall_fscore_support(
            true_binary, pred_binary, average="weighted"
        )
        return metrics

    precision, recall, f1, _ = generate_metrics(test_data)
    accuracy = accuracy_score(true, pred)
    metrics = {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy}
    store_metrics(
        vol=vol,
        metrics=metrics,
        lang=lang,
        task_name="NER",
    )


langs = ["yor", "pcm", "ibo", "hau"]


def run_all(vol):
    for lang in langs:
        LOG.info(f"starting evaluation for {lang}")
        topic_classification(model=model, lang=lang, vol=vol)
        sentiment_analysis(model=model, lang=lang, vol=vol)
        NER(model=model, lang=lang, vol=vol)
