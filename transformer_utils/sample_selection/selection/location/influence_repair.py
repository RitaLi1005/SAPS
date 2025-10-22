import os
import sys
import re
import json
import random
import numpy as np
import torch
from torch.autograd import grad
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend, suitable for server environments
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import List, Literal, Tuple

from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances

import transformers
import hdbscan  # Alternative clustering method

# Add project path
sys.path.append(os.path.abspath(os.path.join(__file__, "../../../..")))
from irepair.constants import DATA_DIR, GPT2_PAD_IDX, DEVICE


# =========================================================
# === Basic Functions: Model Structure and Parameter Extraction ===
# =========================================================

def get_transformer_blocks(model):
    """
    Automatically identify the transformer block list in the model.
    Compatible with common architectures: GPT2, BERT, LLaMA, RoBERTa.
    """
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):  # GPT2
        return model.transformer.h
    elif hasattr(model, "bert") and hasattr(model.bert, "encoder"):  # BERT
        return model.bert.encoder.layer
    elif hasattr(model, "model") and hasattr(model.model, "layers"):  # LLaMA
        return model.model.layers
    elif hasattr(model, "encoder") and hasattr(model.encoder, "layer"):  # RoBERTa
        return model.encoder.layer
    else:
        raise ValueError("Unable to identify transformer block path, please specify manually.")


def get_layer_params(model, layer_ids):
    """
    Extract all parameters of the corresponding layers based on layer indices.
    """
    transformer_blocks = get_transformer_blocks(model)
    selected_params = []
    for idx in layer_ids:
        layer = transformer_blocks[idx]
        selected_params.extend(list(layer.parameters()))
    return selected_params


# =========================================================
# === Sample Influence Calculation (Influence Functions) ===
# =========================================================

def compute_influence_layer(model, tokenizer, ds, rank1, selected_layer_ids, device="cuda"):
    """
    Calculate the influence score for each sample in the given dataset ds,
    using only the parameters of the specified layers (selected_layer_ids) for gradient computation.
    """
    influence_scores = []
    target_params = get_layer_params(model, selected_layer_ids)

    for data in ds:
        data = json.loads(data)
        prompt = data["prompt_text"]
        target = data["gold_text"]
        full_text = prompt + " " + target

        # Tokenization
        tokenized_full = tokenizer(full_text, return_tensors="pt", padding=True, truncation=True)
        tokenized_prompt = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)

        input_ids = tokenized_full.input_ids.to(device)
        prompt_len = tokenized_prompt.input_ids.shape[1]

        labels = input_ids.clone()
        labels[:, :prompt_len] = -100  # Mask prompt part to exclude from loss calculation

        model.zero_grad()
        outputs = model(input_ids, labels=labels)
        loss = outputs.loss

        grads = grad(loss, target_params, retain_graph=False, create_graph=False)
        influence_score = sum(g.norm().item() for g in grads)
        influence_scores.append((data, influence_score))
        print(influence_scores)

    # Sort by score, take the top rank1 high-influence samples
    influence_scores.sort(key=lambda x: x[1], reverse=True)
    top_ds = [x[0] for x in influence_scores[:rank1]]
    return [json.dumps(x) for x in top_ds]


def compute_influence_single(model, tokenizer, data_item, device="cuda", selected_layer_ids=None):
    """
    Calculate the influence score for a single sample, with optional layer specification for computation.
    """
    data = json.loads(data_item)
    prompt = data["prompt_text"]
    target = data["gold_text"]
    full_text = prompt + " " + target

    # Tokenize
    tokenized_full = tokenizer(full_text, return_tensors="pt", padding=True, truncation=True).to(device)
    tokenized_prompt = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)

    input_ids = tokenized_full["input_ids"]
    prompt_len = tokenized_prompt["input_ids"].shape[1]

    labels = input_ids.clone()
    labels[:, :prompt_len] = -100  # mask prompt

    model.zero_grad()
    outputs = model(input_ids, labels=labels)
    loss = outputs.loss

    # Extract parameters by layer
    if selected_layer_ids is None:
        grads = grad(loss, model.parameters(), retain_graph=False, create_graph=False)
    else:
        target_params = get_layer_params(model, selected_layer_ids)
        grads = grad(loss, target_params, retain_graph=False, create_graph=False)

    influence_score = sum(g.norm().item() for g in grads if g is not None)
    return data_item, influence_score


def compute_influence_with_cache(model, tokenizer, data_list, device="cuda", selected_layer_ids=None):
    """
    Calculate influence scores for a batch of samples and return the indices of the top 50% most influential samples.
    """
    influence_scores = []
    for idx, item in enumerate(tqdm(data_list, desc="Computing influence scores")):
        try:
            _, score = compute_influence_single(model, tokenizer, item,
                                                device=device,
                                                selected_layer_ids=selected_layer_ids)
            influence_scores.append((idx, score))
        except Exception as e:
            print(f"[WARN] Skipping item due to error: {e}")
            continue

    influence_scores.sort(key=lambda x: x[1], reverse=True)
    top_k = len(influence_scores) // 2
    selected_indices = [idx for idx, _ in influence_scores[:top_k]]
    print(f"[INFO] Selected top 50% most influential samples: {len(selected_indices)}")
    return selected_indices


# =========================================================
# === Helper Functions ===
# =========================================================

def extract_number(filename):
    """Extract the trailing number from a filename (e.g., file_12.jsonl → 12)"""
    match = re.search(r'(\d+)(?=\.jsonl$)', filename)
    return int(match.group(1)) if match else 0


def count_jsonl_lines(filepath):
    """Count the number of lines in a jsonl file"""
    count = 0
    with open(filepath, 'r', encoding='utf-8') as f:
        for _ in f:
            count += 1
    return count


def load_top_k_jsonl(path, k=200):
    """Load the first k lines of a jsonl file (skip empty lines)"""
    results = []
    with open(path, "r") as f:
        for i, line in enumerate(f):
            if i >= k:
                break
            if line.strip():
                results.append(line.strip())
    return results


# =========================================================
# === Data Bucketing and Sampling Methods ===
# =========================================================

def load_scores(score_dir, valid_size, trainsize, seed=42):
    """
    Load all score files in the toxicity_scores directory, and perform randomization and truncation.
    """
    all_score_files = [f for f in sorted(os.listdir(score_dir)) if f.endswith(".jsonl")]
    all_score_files = sorted(all_score_files, key=extract_number)
    all_score_files = [os.path.join(score_dir, f) for f in all_score_files]

    scores = []
    for f in all_score_files:
        with open(f, 'r') as fp:
            for line in fp:
                obj = json.loads(line)
                scores.append(obj.get("toxicity_score", 0.0))

    # Remove validation set portion
    scores = scores[:-valid_size]

    random.seed(seed)
    indices = list(range(len(scores)))
    random.shuffle(indices)
    scores = [scores[i] for i in indices]

    return scores[:trainsize]


def CCS(scores, n_total, thresholds=(0.2, 0.6), seed=42):
    """
    Bucket sampling strategy based on toxicity score (CCS).
    Divide samples into several score intervals, and randomly sample proportionally from each interval.
    """
    random.seed(seed)
    buckets = [[] for _ in range(len(thresholds) + 1)]

    for i, score in enumerate(scores):
        placed = False
        for j, t in enumerate(thresholds):
            if score <= t:
                buckets[j].append(i)
                placed = True
                break
        if not placed:
            buckets[-1].append(i)

    per_group = n_total // len(buckets)
    selected = []
    for group in buckets:
        k = min(per_group, len(group))
        selected += random.sample(group, k)
    return selected


def k_center_greedy(embeddings: np.ndarray, k: int) -> list:
    """
    k-center greedy sampling method.
    Select k samples from the embedding space to ensure maximum coverage.
    """
    np.random.seed(42)
    n = embeddings.shape[0]
    selected = [np.random.randint(0, n)]
    distances = pairwise_distances(embeddings, embeddings[selected])[:, 0]

    for _ in range(k - 1):
        next_idx = np.argmax(distances)
        selected.append(next_idx)
        new_distances = pairwise_distances(embeddings, embeddings[[next_idx]])[:, 0]
        distances = np.minimum(distances, new_distances)

    return selected


# =========================================================
# === Main Function: Embedding-based Data Selection ===
# =========================================================

def top_data_embedding(
    model,
    trainsize,
    valid_size,
    seed,
    per_cluster,
    total_samples,
    strategy: Literal["SAPS_random", "SAPS", "stratified", "kcenter", "GraNd", "CCS"] = "SAPS_random",
    SAPS_mode: Literal["top", "mix", "global", "density"] = "top",
    alpha: float = 0.5,
    cache_path=None
):
    """
    Use sentence embeddings and clustering strategies to select representative samples from the dataset.
    Supported strategies:
        - SAPS_random: Random sampling per cluster of SAPS
        - SAPS: Sampling based on margin distance
        - stratified: Stratified sampling proportional to cluster size
        - kcenter: Sample selection based on coverage
        - GraNd: Filtering through gradient influence
        - CCS: Sampling based on toxicity score intervals
    """
    data_dir = os.path.join(DATA_DIR, "toxicity_pairwise")
    all_train_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".jsonl")]

    # Aggregate all training data
    paired_data = []
    for f in all_train_files:
        with open(f, 'r') as fp:
            paired_data.extend(fp.readlines())

    paired_data = paired_data[:-valid_size]
    random.seed(seed)
    random.shuffle(paired_data)
    paired_data = paired_data[:trainsize]
    # print(paired_data[5])

    # === GraNd Strategy: Select samples by gradient influence ===
    if strategy == "GraNd":
        gpt2model = transformers.AutoModelForCausalLM.from_pretrained(
            '/root/autodl-tmp/mnt/memit/model/gpt-neo-1.3b',
            output_hidden_states=True
        ).to("cuda").eval()

        tok = transformers.AutoTokenizer.from_pretrained("/root/autodl-tmp/mnt/memit/model/gpt-neo-1.3b")
        tok.pad_token = tok.eos_token

        selected_indices = compute_influence_with_cache(gpt2model, tok, paired_data)
        selected_texts = [paired_data[i] for i in selected_indices]

        if cache_path:
            with open(cache_path, "w") as f:
                for item in selected_texts:
                    f.write(item.strip() + "\n")
            print(f"[INFO] Saved selected data to {cache_path}")

        return selected_texts

    # === CCS Strategy: Stratified sampling based on scores ===
    if strategy == "CCS":
        SCORE_DIR = os.path.join(DATA_DIR, "toxicity_scores")
        toxicity_scores = load_scores(SCORE_DIR, valid_size, trainsize, seed)
        selected_indices = CCS(toxicity_scores, total_samples, thresholds=(0.02, 0.03, 0.04, 0.05), seed=seed)
        selected_texts = [paired_data[i] for i in selected_indices]

        if cache_path:
            with open(cache_path, "w") as f:
                for item in selected_texts:
                    f.write(item.strip() + "\n")
            print(f"[INFO] Saved selected data to {cache_path}")

        return selected_texts

    # === Other Strategies: Based on clustering and margin distance ===
    embeddings = model.encode(paired_data, batch_size=64, show_progress_bar=True)

    # PCA dimensionality reduction for clustering
    pca_full = PCA(random_state=42)
    pca_full.fit(embeddings)
    cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
    target_dim = np.argmax(cumulative_variance >= 0.60) + 1
    print(f"Selected target dimension: {target_dim}, cumulative explained variance: {cumulative_variance[target_dim - 1]:.4f}")

    target_dim = 50
    pca = PCA(n_components=target_dim, random_state=42)
    embeddings_reduced = pca.fit_transform(embeddings)

    if strategy == "kcenter":
        k = trainsize // 2
        selected_indices = k_center_greedy(embeddings_reduced, k)
    else:
        # MiniBatch KMeans clustering
        num_clusters = min(10, len(paired_data))
        kmeans = MiniBatchKMeans(n_clusters=num_clusters, random_state=42, batch_size=1000)
        kmeans.fit(embeddings_reduced)
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_

        selected_indices = []

        for i in range(num_clusters):
            cluster_indices = np.where(labels == i)[0]
            if len(cluster_indices) == 0:
                continue

            # Different strategies correspond to different sampling logic
            if strategy == "SAPS_random":
                k = min(per_cluster, len(cluster_indices))
                sampled = random.sample(list(cluster_indices), k)
                selected_indices.extend(sampled)

            elif strategy == "SAPS":
                # Calculate distance from each point to cluster center
                distances = [(idx, np.linalg.norm(embeddings_reduced[idx] - centers[i])) for idx in cluster_indices]

                if SAPS_mode == "top":  # Take the farthest points
                    distances.sort(key=lambda x: x[1], reverse=True)
                    selected = [idx for idx, _ in distances[:per_cluster]]

                elif SAPS_mode == "mix":  # Mixed sampling of nearest and farthest
                    distances.sort(key=lambda x: x[1])
                    top_k = int(per_cluster * alpha)
                    bottom_k = per_cluster - top_k
                    bottom = [idx for idx, _ in distances[:bottom_k]] if bottom_k > 0 else []
                    top = [idx for idx, _ in distances[-top_k:]] if top_k > 0 else []
                    selected = top + bottom
                    selected.sort(key=lambda i: dict(distances)[i], reverse=True)

                elif SAPS_mode == "global":  # Combine local and global centers
                    scores = []
                    global_center = np.mean(embeddings_reduced, axis=0)
                    for idx in cluster_indices:
                        local_dist = np.linalg.norm(embeddings_reduced[idx] - centers[i])
                        global_dist = np.linalg.norm(embeddings_reduced[idx] - global_center)
                        score = local_dist + global_dist
                        scores.append((idx, score))
                    scores.sort(key=lambda x: x[1], reverse=True)
                    selected = [idx for idx, _ in scores[:per_cluster]]

                elif SAPS_mode == "density":  # Density-aware margin
                    nbrs = NearestNeighbors(n_neighbors=5).fit(embeddings_reduced)
                    densities = []
                    for idx in cluster_indices:
                        dists_, _ = nbrs.kneighbors([embeddings_reduced[idx]])
                        densities.append((idx, np.mean(dists_)))

                    scores = []
                    for (idx, dens) in densities:
                        margin = np.linalg.norm(embeddings_reduced[idx] - centers[i])
                        score = margin / (dens + 1e-6)
                        scores.append((idx, score))
                    scores.sort(key=lambda x: x[1], reverse=True)
                    selected = [idx for idx, _ in scores[:per_cluster]]

                else:
                    raise ValueError(f"Unsupported SAPS_mode: {SAPS_mode}")

                selected_indices.extend(selected)

            elif strategy == "stratified":
                if total_samples is None:
                    raise ValueError("You must specify total_samples for stratified sampling.")
                weight = len(cluster_indices) / len(paired_data)
                k = max(1, int(total_samples * weight))
                k = min(k, len(cluster_indices))
                sampled = random.sample(list(cluster_indices), k)
                selected_indices.extend(sampled)

    # === Write results ===
    selected_texts = [paired_data[i] for i in selected_indices]
    if cache_path:
        with open(cache_path, "w") as f:
            for item in selected_texts:
                f.write(item.strip() + "\n")
        print(f"[INFO] Saved selected data to {cache_path}")

    return selected_texts

