"""
Train loop for DPO.
"""
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import re
from typing import Optional, Dict, List, Union, Tuple

import random
import os
from collections import defaultdict
import time
import json
import functools
import contextlib
from collections import Counter

import numpy as np
import wandb
from torch.autograd.profiler import record_function
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import KLDivLoss
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    StateDictType,
    BackwardPrefetch,
    ShardingStrategy,
    CPUOffload,
)
from torch.distributed.fsdp.api import (
    FullStateDictConfig,
    FullOptimStateDictConfig,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import transformers
from omegaconf import DictConfig

from generate_sensitivity import init_fixed_selection
from pplm_dataset import get_pplm_batch_iterator
from dpo_utils import (
    slice_and_move_batch_for_device,
    formatted_dict,
    all_gather_if_needed,
    pad_to_length,
    get_block_class_from_model,
    rank0_print,
    get_local_dir, dpo_loss, get_kl_div, dpo_forward,
)
from sr_utils import sr_loss
import os,sys
sys.path.append(os.path.abspath(os.path.join(__file__, "../../../..")))
from irepair.constants import GPT2_PAD_IDX

from irepair.util.torch_layer_util import get_batch_logps
from sample_selection.selection.location import load_top_k_jsonl,count_jsonl_lines
torch.backends.cuda.matmul.allow_tf32 = True


def generate(
        model,
        batch,
        max_new_tokens,
        pad_token_id,
        include_ngram_blocked=False,
        include_ref=False,
        fsdp=False,
        ref_model=None,
):
    """
    Return greedy and n-gram blocked generations.
    """
    prompt_shape = batch["prompt_input_ids"].shape[1]
    with torch.no_grad():
        # FSDP generation according to https://github.com/pytorch/pytorch/issues/100069
        ctx = lambda: (
            FSDP.summon_full_params(model, writeback=False, recurse=False)
            if fsdp
            else contextlib.nullcontext()
        )
        with ctx():
            greedy_resp = model.generate(
                input_ids=batch["prompt_input_ids"],
                attention_mask=batch["prompt_attention_mask"],
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=pad_token_id,
            )

        greedy_resp_labels = greedy_resp.detach().clone()
        greedy_resp_labels[:, :prompt_shape] = -100
        output = {
            "policy_input_ids": greedy_resp,
            "policy_attention_mask": greedy_resp != GPT2_PAD_IDX,
            "policy_labels": greedy_resp_labels,
        }

    return output


def parse_time(stat, type='cuda'):
    if type == 'cuda':
        sig = stat.cuda_time_total_str
    else:
        sig = stat.cpu_time_total_str
    match = re.match(r"([0-9.]+)([a-zA-Z]+)", sig)
    number = float(match.group(1))
    unit = match.group(2)

    if unit == 'ms':
        return number / 1000
    if unit == 'us':
        return number / 1000000
    if unit == 's':
        return number
    raise Exception('Unknwon time unit: ' + unit)


def parse_memory(stat, type='cuda'):
    if type == 'cuda':
        sig = stat.cuda_memory_usage
    else:
        sig = stat.cpu_memory_usage

    sig /= 8  # to byte
    sig /= 1000  # to KB
    sig /= 1000  # to MB
    return sig


class BasicTrainer(object):
    def __init__(
            self,
            policy: nn.Module,
            config: DictConfig,
            seed: int,
            run_dir: str,
            reference_model: Optional[nn.Module] = None,
            rank: int = 0,
            world_size: int = 1,
            device='cuda'
    ):
        """
        A trainer for a language model, supporting either SFT or DPO training.

        If multiple GPUs are present, naively splits the model across them, effectively
        offering N times available memory, but without any parallel computation.
        """
        self.seed = seed
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=False)

        self.most_toxic_layer = ''
        self.rank = rank
        self.world_size = world_size
        self.config = config
        self.epoch = config.n_epochs
        self.run_dir = run_dir
        self.example_counter = 0
        self.batch_counter = 0
        self.last_log = None
        self.patience = 0
        self.val_metric_value = -1
        self.device = device
        if config.validation_direction == "max":
            self.val_direction = 1
            self.best_val_metric = -1

        else:
            self.val_direction = -1
            self.best_val_metric = 1e10

        tokenizer_name_or_path = (
                config.model.tokenizer_name_or_path or config.model.name_or_path
        )
        rank0_print(f"Loading tokenizer {tokenizer_name_or_path}")
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            tokenizer_name_or_path, cache_dir=get_local_dir(config.local_dirs)
        )
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.policy = policy
        self.reference_model = reference_model
        self.kl_criterion = KLDivLoss(reduction="none", log_target=True)


        path=f"/root/autodl-tmp/mnt/SAPS/transformer_utils/irepair/data/top_data_embedding_{config.sample_seed}_{config.sample_method}_top_{config.sample_alpha}.jsonl"
        count=count_jsonl_lines(path)
        self.top_data=load_top_k_jsonl(path, k=count)

        self.eval_iterator = get_pplm_batch_iterator(
            self.tokenizer,
            self.config,
            self.top_data,
            split="valid",
            device=device
        )
        self.eval_batches = list(self.eval_iterator)
        rank0_print(
            f"Loaded {len(self.eval_batches)} eval batches of size {config.eval_batch_size}"
        )

        if hasattr(config.loss, 'fixed_selection') and config.loss.fixed_selection:
            init_fixed_selection(self.config, self.policy)

    def get_batch_samples(
            self, batch: Dict[str, torch.LongTensor]
    ) -> Tuple[str, str]:
        """
        Generate samples from the policy (and reference model, if doing DPO training)
        for the given batch of inputs
        """

        # FSDP generation according to https://github.com/pytorch/pytorch/issues/100069
        ctx = lambda: (
            FSDP.summon_full_params(
                self.policy, writeback=False, recurse=False
            )
            if "FSDP" in self.config.trainer
            else contextlib.nullcontext()
        )
        with ctx():
            policy_output = self.policy.generate(
                batch["prompt_input_ids"],
                attention_mask=batch["prompt_attention_mask"],
                max_length=self.config.max_length,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        if self.config.loss.name == "dpo":
            ctx = lambda: (
                FSDP.summon_full_params(
                    self.reference_model, writeback=False, recurse=False
                )
                if "FSDP" in self.config.trainer
                else contextlib.nullcontext()
            )
            with ctx():
                reference_output = self.reference_model.generate(
                    batch["prompt_input_ids"],
                    attention_mask=batch["prompt_attention_mask"],
                    max_length=self.config.max_length,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

        policy_output = pad_to_length(
            policy_output, self.config.max_length, self.tokenizer.pad_token_id
        )
        policy_output = all_gather_if_needed(
            policy_output, self.rank, self.world_size
        )
        policy_output_decoded = self.tokenizer.batch_decode(
            policy_output, skip_special_tokens=True
        )

        reference_output_decoded = []
        if self.config.loss.name == "dpo":
            reference_output = pad_to_length(
                reference_output,
                self.config.max_length,
                self.tokenizer.pad_token_id,
            )
            reference_output = all_gather_if_needed(
                reference_output, self.rank, self.world_size
            )
            reference_output_decoded = self.tokenizer.batch_decode(
                reference_output, skip_special_tokens=True
            )

        return policy_output_decoded, reference_output_decoded

    def get_batch_metrics(
            self,
            batch: Dict[str, Union[List, torch.LongTensor]],
            loss_config: DictConfig,
            train=True,
    ):
        """
        Compute the SFT or DPO loss and other metrics for the given batch of inputs.
        """

        metrics = {}
        train_test = "train" if train else "valid"
        kl_loss = None

        if loss_config.name == "sr" or loss_config.name == "sftkl":
            losses, negative_losses = sr_loss(
                self.policy, self.reference_model, batch,
                beta=loss_config.beta,
                selection=loss_config.dynamic_selection,
                gamma=loss_config.gamma,
                loss_scale=loss_config.loss_scale
            )
            all_devices_losses = all_gather_if_needed(
                losses.detach(), self.rank, self.world_size
            )
            metrics[f"loss/{train_test}"] = (
                all_devices_losses.cpu().numpy().tolist()
            )
        elif loss_config.name == "dpo":
            (
                policy_pos_logps,
                policy_neg_logps,
                policy_pos_logits,
                policy_neg_logits,
            ) = dpo_forward(self.policy, batch)
            with torch.no_grad():
                (
                    ref_pos_logps,
                    ref_neg_logps,
                    ref_pos_logits,
                    ref_neg_logits,
                ) = dpo_forward(self.reference_model, batch)
            losses, pos_rewards, neg_rewards = dpo_loss(
                policy_pos_logps,
                policy_neg_logps,
                ref_pos_logps,
                ref_neg_logps,
                beta=loss_config.beta,
                reference_free=loss_config.reference_free,
            )
            pos_kl_div, neg_kl_div = get_kl_div(
                self.kl_criterion,
                policy_pos_logits,
                policy_neg_logits,
                ref_pos_logits,
                ref_neg_logits,
            )
            if loss_config.kl_gamma > 0:
                kl_loss = loss_config.kl_gamma * (pos_kl_div + neg_kl_div)
                losses += kl_loss

            reward_accuracies = (pos_rewards > neg_rewards).float()

            pos_rewards = all_gather_if_needed(
                pos_rewards, self.rank, self.world_size
            )
            neg_rewards = all_gather_if_needed(
                neg_rewards, self.rank, self.world_size
            )
            reward_accuracies = all_gather_if_needed(
                reward_accuracies, self.rank, self.world_size
            )

            metrics[f"rewards_{train_test}/positive"] = (
                pos_rewards.cpu().numpy().tolist()
            )
            metrics[f"rewards_{train_test}/negative"] = (
                neg_rewards.cpu().numpy().tolist()
            )
            metrics[f"rewards_{train_test}/accuracies"] = (
                reward_accuracies.cpu().numpy().tolist()
            )
            metrics[f"rewards_{train_test}/margins"] = (
                (pos_rewards - neg_rewards).cpu().numpy().tolist()
            )

            policy_neg_logps = all_gather_if_needed(
                policy_neg_logps.detach(), self.rank, self.world_size
            )
            metrics[f"logps_{train_test}/negative"] = (
                policy_neg_logps.cpu().numpy().tolist()
            )

            metrics[f"kl_div_{train_test}/positive"] = (
                pos_kl_div.detach().cpu().numpy().tolist()
            )

            metrics[f"kl_div_{train_test}/negative"] = (
                neg_kl_div.detach().cpu().numpy().tolist()
            )

            if loss_config.kl_gamma > 0 and kl_loss is not None:
                metrics[f"kl_loss_{train_test}"] = (
                    kl_loss.detach().cpu().numpy().tolist()
                )
            policy_pos_logps = all_gather_if_needed(
                policy_pos_logps.detach(), self.rank, self.world_size
            )
            metrics[f"logps_{train_test}/positive"] = (
                policy_pos_logps.cpu().numpy().tolist()
            )

            all_devices_losses = all_gather_if_needed(
                losses.detach(), self.rank, self.world_size
            )
            metrics[f"loss/{train_test}"] = (
                all_devices_losses.cpu().numpy().tolist()
            )

        elif loss_config.name == "sft":
            policy_pos_logits = self.policy(
                batch["pos_input_ids"],
                attention_mask=batch["pos_attention_mask"],
            ).logits.to(torch.float32)
            policy_pos_logps = get_batch_logps(
                policy_pos_logits,
                batch["pos_labels"],
                average_log_prob=False,
            )

            losses = -policy_pos_logps

            policy_pos_logps = all_gather_if_needed(
                policy_pos_logps.detach(), self.rank, self.world_size
            )
            metrics[f"logps_{train_test}/positive"] = (
                policy_pos_logps.cpu().numpy().tolist()
            )

            all_devices_losses = all_gather_if_needed(
                losses.detach(), self.rank, self.world_size
            )
            metrics[f"loss/{train_test}"] = (
                all_devices_losses.cpu().numpy().tolist()
            )

        if loss_config.name == "sr" and loss_config.dynamic_selection and train_test == 'train':
            return losses.mean(), negative_losses.mean(), metrics

        return losses.mean(), metrics

    def train_loop(self):
        """Begin either SFT or DPO training, with periodic evaluation."""

        rank0_print(f"Using {self.config.optimizer} optimizer")
        self.optimizer = getattr(torch.optim, self.config.optimizer)(
            self.policy.parameters(), lr=self.config.lr
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: min(
                1.0, (step + 1) / (self.config.warmup_steps + 1)
            ),
        )

        if self.reference_model is not None:
            self.reference_model.eval()

        # Initialize CUDA events
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()

        # with torch.profiler.profile(
        #         activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        #         record_shapes=False,
        #         profile_memory=True,
        #         with_flops=True
        # ) as prof:
        #     with record_function("model_training"):
        done = False
        for e in range(self.epoch):
            if done:
                break
            rank0_print(f"Starting epoch: {e + 1}")
            self.train_iterator = get_pplm_batch_iterator(
                self.tokenizer,
                self.config,
                self.top_data,
                split="train",
                device=self.device,
            )
            for batch in self.train_iterator:
                if not self.config.enable_max_step and self.example_counter % self.config.eval_every == 0 and self.example_counter > 0:
                    result = self.eval()
                    if result == -1:
                        done = True
                        break
                if self.config.enable_max_step and self.example_counter >= self.config.max_step:
                    self.eval()
                    done = True
                    break

                if self.config.loss.name == 'sr' and self.config.loss.dynamic_selection:
                    self.train_sr(batch)
                else:
                    self.train(batch)

        # Record the end time
        end_event.record()

        # Wait for the events to be recorded
        torch.cuda.synchronize()
        #
        # # Calculate the elapsed time in milliseconds
        elapsed_time_ms = start_event.elapsed_time(end_event)
        elapsed_time_sec = elapsed_time_ms / 1000
        #
        json_save_path="/root/autodl-tmp/mnt/SAPS/transformer_utils/irepair/data/result/time.json"
        if os.path.exists(json_save_path):
            with open(json_save_path, "r", encoding="utf-8") as f:
                try:
                    existing_data = json.load(f)
                except json.JSONDecodeError:
                    existing_data = []
        else:
            existing_data = []

        # 追加新结果
        existing_data.append(elapsed_time_sec)

        # 写回 JSON 文件
        with open(json_save_path, "w", encoding="utf-8") as f:
            json.dump(existing_data, f, indent=2, ensure_ascii=False)

        # print(f"GPU time used: {elapsed_time_sec:.4f} seconds")

        # flops = sum(stat.flops for stat in prof.key_averages())
        # print(f"Total TFLOPs: {flops / 1E12}")


    def train(self, batch):
        """
        Run single train step.
        """
        self.policy.train()

        start_time = time.time()
        batch_metrics = defaultdict(list)
        for microbatch_idx in range(self.config.gradient_accumulation_steps):
            # batch:
            # {
            #   "pos_input_ids": Tensor[batch, seq],
            #   "pos_attention_mask": Tensor[batch, seq],
            #   "neg_input_ids": Tensor[batch, seq],
            #   "neg_attention_mask": Tensor[batch, seq],
            # }
            self.policy.train()
            global_microbatch = slice_and_move_batch_for_device(
                batch,
                microbatch_idx,
                self.config.gradient_accumulation_steps,
                device=self.device
            )
            local_microbatch = slice_and_move_batch_for_device(
                global_microbatch, self.rank, self.world_size,
                device=self.device
            )
            loss, metrics = self.get_batch_metrics(
                local_microbatch, self.config.loss, train=True
            )
            (loss / self.config.gradient_accumulation_steps).backward()

            for k, v in metrics.items():
                batch_metrics[k].extend(v)

        # if hasattr(self.config.loss, 'select') and self.config.loss.select:
        #     self.sr_selective_update()

        grad_norm = self.clip_gradient()

        # to debug if model is changing
        # initial_params = {name: param.clone().detach() for name, param in self.policy.named_parameters()}

        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()

        # for name, param in self.policy.named_parameters():
        #         if param.requires_grad:
        #             if not torch.equal(initial_params[name], param.data):
        #                 print(f"Param {name} changed")

        step_time = time.time() - start_time
        examples_per_second = self.config.batch_size / step_time
        batch_metrics["examples_per_second"].append(examples_per_second)
        batch_metrics["grad_norm"].append(grad_norm)

        self.batch_counter += 1
        self.example_counter += self.config.batch_size

        if (
                self.last_log is None
                or time.time() - self.last_log
                > self.config.minimum_log_interval_secs
        ):
            mean_train_metrics = {
                k: sum(v) / len(v) for k, v in batch_metrics.items()
            }
            mean_train_metrics["counters/examples"] = self.example_counter
            mean_train_metrics["counters/updates"] = self.batch_counter
            rank0_print(
                f"train stats after {self.example_counter} examples: {formatted_dict(mean_train_metrics)}"
            )

            if self.config.wandb.enabled and self.rank == 0:
                wandb.log(mean_train_metrics, step=self.example_counter)

            self.last_log = time.time()

    def train_sr(self, batch):
        self.policy.train()
        start_time = time.time()
        batch_metrics = defaultdict(list)

        loss, negative_loss, metrics = self.get_batch_metrics(
            batch, self.config.loss, train=True
        )
        negative_loss.backward(retain_graph=True)
        # print(f"[DEBUG] step={self.batch_counter},  negative_loss={negative_loss.item()}")
        del negative_loss  # 释放中间图
        torch.cuda.empty_cache()  # 清理缓存

        # print(f"[DEBUG] step={self.batch_counter}")
        sensitivities = self.calculate_block_sensitivities()
        # print(f"[DEBUG] sensitivities max={max(sensitivities.values())}, min={min(sensitivities.values())}")
        mtl = max(sensitivities, key=sensitivities.get)
        # if self.most_toxic_layer!=mtl:
        #     print('Most toxic block (before, now): ',self.most_toxic_layer,mtl)
        self.most_toxic_layer = mtl

        # Clear gradients and set requires_grad=False for non-toxic layers
        for name, param in self.policy.named_parameters():
            layer_name = self.extract_block_name(name)
            if layer_name != self.most_toxic_layer:
                param.requires_grad = False  # Disable gradient computation for non-toxic layers
            else:
                # print(f"[DEBUG] Toxic layer in step {self.batch_counter}: {name}")
                param.requires_grad = True  # Ensure gradient computation for the toxic layer

        self.optimizer.zero_grad()
        loss.backward()
        # print(f"[DEBUG] step={self.batch_counter}, loss={loss.item()}")
        del loss
        torch.cuda.empty_cache()  # 清理缓存

        for k, v in metrics.items():
            # batch_metrics[k].extend(v)
            batch_metrics[k].extend([x.detach().cpu() if torch.is_tensor(x) else x for x in v])
        del metrics

        grad_norm = self.clip_gradient()

        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()

        for _, param in self.policy.named_parameters():
            param.requires_grad = True

        step_time = time.time() - start_time
        examples_per_second = self.config.batch_size / step_time
        batch_metrics["examples_per_second"].append(examples_per_second)
        batch_metrics["grad_norm"].append(grad_norm)

        self.batch_counter += 1
        self.example_counter += self.config.batch_size

        if (
                self.last_log is None
                or time.time() - self.last_log
                > self.config.minimum_log_interval_secs
        ):
            mean_train_metrics = {
                k: sum(v) / len(v) for k, v in batch_metrics.items()
            }
            mean_train_metrics["counters/examples"] = self.example_counter
            mean_train_metrics["counters/updates"] = self.batch_counter
            rank0_print(
                f"train stats after {self.example_counter} examples: {formatted_dict(mean_train_metrics)}"
            )

            if self.config.wandb.enabled and self.rank == 0:
                wandb.log(mean_train_metrics, step=self.example_counter)

            self.last_log = time.time()

    def eval(self):
        """
        Run evaluation.
        """
        rank0_print(
            f"Running evaluation after {self.example_counter} train examples"
        )
        self.policy.eval()

        all_eval_metrics = defaultdict(list)
        if self.config.sample_during_eval:
            all_policy_samples, all_reference_samples = [], []

        for eval_batch in (
                tqdm(self.eval_batches, desc="Computing eval metrics")
                if self.rank == 0
                else self.eval_batches
        ):

            local_eval_batch = slice_and_move_batch_for_device(
                eval_batch, self.rank, self.world_size, self.device
            )
            with torch.no_grad():
                _, eval_metrics = self.get_batch_metrics(
                    local_eval_batch, self.config.loss, train=False
                )

            for k, v in eval_metrics.items():
                all_eval_metrics[k].extend(v)

        if (
                self.config.sample_during_eval
                and self.example_counter % self.config.sample_every == 0
        ):
            if self.config.n_eval_model_samples < self.config.eval_batch_size:
                rank0_print(
                    f"Warning: n_eval_model_samples ({self.config.n_eval_model_samples}) < \
                    eval_batch_size ({self.config.eval_batch_size}). \
                    Sampling from the first complete eval batch of prompts."
                )
                sample_batches = self.eval_batches[:1]
            else:
                n_sample_batches = (
                        self.config.n_eval_model_samples
                        // self.config.eval_batch_size
                )
                sample_batches = self.eval_batches[:n_sample_batches]

            for eval_batch in (
                    tqdm(sample_batches, desc="Generating samples...")
                    if self.rank == 0
                    else sample_batches
            ):
                local_eval_batch = slice_and_move_batch_for_device(
                    eval_batch, self.rank, self.world_size, self.rank
                )
                (
                    policy_samples,
                    reference_samples,
                ) = self.get_batch_samples(local_eval_batch)

                all_policy_samples.extend(policy_samples)
                all_reference_samples.extend(reference_samples)

            rank0_print("Policy samples:")
            rank0_print(json.dumps(all_policy_samples[:10], indent=2))

        mean_eval_metrics = {
            k: sum(v) / len(v) for k, v in all_eval_metrics.items()
        }
        print(all_eval_metrics)
        self.val_metric_value = mean_eval_metrics[
            self.config.validation_metric
        ]

        rank0_print(
            f"eval after {self.example_counter}: {formatted_dict(mean_eval_metrics)}"
        )

        if self.config.wandb.enabled and self.rank == 0:
            wandb.log(mean_eval_metrics, step=self.example_counter)

        if self.example_counter == 0:
            return 0

        if (
                self.val_metric_value is not None
                and self.val_metric_value * self.val_direction
                > self.val_direction * self.best_val_metric
        ):
            self.best_val_metric = self.val_metric_value

            rank0_print(
                f"\n=====\nNew best for {self.config.validation_metric}: {self.best_val_metric}.\n=====\n"
            )
            self.patience = 0

            if self.config.debug:
                rank0_print("skipping save in debug mode")
            else:
                output_dir = os.path.join(self.run_dir, "checkpoints")
                rank0_print(
                    f"Creating checkpoint to write to {output_dir}..."
                )
                self.save(output_dir, mean_eval_metrics)
        else:
            self.patience += 1
            if self.patience >= self.config.validation_patience:
                rank0_print("Ran out of patience, stopping training...")
                return -1

        return 0

    def clip_gradient(self):
        """Clip the gradient norm of the parameters of a non-FSDP policy."""
        return torch.nn.utils.clip_grad_norm_(
            self.policy.parameters(), self.config.max_grad_norm
        ).item()

    def calculate_layer_sensitivities(self):
        sensitivities = {}
        for name, param in self.policy.named_parameters():
            if param.requires_grad:
                layer_name = name[:name.rindex('.')]  # Get the layer name
                if layer_name not in sensitivities:
                    sensitivities[layer_name] = []
                sensitivities[layer_name].append(param.grad.norm(2).item() ** 2)

        for layer in sensitivities:
            sensitivities[layer] = sum(sensitivities[layer]) ** 0.5

    def extract_block_name(self, s):
        # Define the regex pattern to match until the second dot
        match = re.match(r'^(.*?\.\d+)', s)

        # Return the matched group
        return match.group(0) if match else None

    def calculate_block_sensitivities(self):
        sensitivities = {}
        for name, param in self.policy.named_parameters():
            if param.grad is not None:
                layer_name = self.extract_block_name(name)
                if layer_name is None:
                    continue
                if layer_name not in sensitivities:
                    sensitivities[layer_name] = []
                sensitivities[layer_name].append(param.grad.norm(2).item() ** 2)

        for layer in sensitivities:
            sensitivities[layer] = sum(sensitivities[layer]) ** 0.5

        return sensitivities

    def write_state_dict(
            self,
            step: int,
            state: Dict[str, torch.Tensor],
            metrics: Dict,
            filename: str,
            dir_name: Optional[str] = None,
    ):
        """Write a checkpoint to disk."""
        if dir_name is None:
            dir_name = os.path.join(self.run_dir, f"LATEST")

        os.makedirs(dir_name, exist_ok=True)
        output_path = os.path.join(dir_name, filename)
        rank0_print(f"writing checkpoint to {output_path}...")
        torch.save(
            {
                "step_idx": step,
                "state": state,
                "metrics": metrics if metrics is not None else {},
            },
            output_path,
        )

    def save(
            self, output_dir: Optional[str] = None, metrics: Optional[Dict] = None
    ):
        """Save policy, optimizer, and scheduler state to disk."""

        policy_state_dict = self.policy.state_dict()
        self.write_state_dict(
            self.example_counter,
            policy_state_dict,
            metrics,
            "policy.pt",
            output_dir,
        )
        del policy_state_dict

        optimizer_state_dict = self.optimizer.state_dict()
        self.write_state_dict(
            self.example_counter,
            optimizer_state_dict,
            metrics,
            "optimizer.pt",
            output_dir,
        )
        del optimizer_state_dict

        scheduler_state_dict = self.scheduler.state_dict()
        self.write_state_dict(
            self.example_counter,
            scheduler_state_dict,
            metrics,
            "scheduler.pt",
            output_dir,
        )


class FSDPTrainer(BasicTrainer):
    def __init__(
            self,
            policy: nn.Module,
            config: DictConfig,
            seed: int,
            run_dir: str,
            reference_model: Optional[nn.Module] = None,
            rank: int = 0,
            world_size: int = 1,
    ):
        """A trainer subclass that uses PyTorch FSDP to shard the model across multiple GPUs.

        This trainer will shard both the policy and reference model across all available GPUs.
        Models are sharded at the block level, where the block class name is provided in the config.
        """

        super().__init__(
            policy, config, seed, run_dir, reference_model, rank, world_size
        )
        assert (
                config.model.block_name is not None
        ), "must specify model.block_name (e.g., GPT2Block or GPTNeoXLayer) for FSDP"

        wrap_class = get_block_class_from_model(
            policy, config.model.block_name
        )
        model_auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={wrap_class},
        )

        shared_fsdp_kwargs = dict(
            auto_wrap_policy=model_auto_wrap_policy,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            cpu_offload=CPUOffload(offload_params=False),
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            device_id=rank,
            ignored_modules=None,
            limit_all_gathers=False,
            use_orig_params=False,
            sync_module_states=False,
        )

        rank0_print("Sharding policy...")
        mp_dtype = (
            getattr(torch, config.model.fsdp_policy_mp)
            if config.model.fsdp_policy_mp is not None
            else None
        )
        policy_mp_policy = MixedPrecision(
            param_dtype=mp_dtype, reduce_dtype=mp_dtype, buffer_dtype=mp_dtype
        )
        self.policy = FSDP(
            policy, **shared_fsdp_kwargs, mixed_precision=policy_mp_policy
        )

        if config.activation_checkpointing:
            rank0_print("Attempting to enable activation checkpointing...")
            try:
                # use activation checkpointing, according to:
                # https://pytorch.org/blog/scaling-multimodal-foundation-models-in-torchmultimodal-with-pytorch-distributed/
                #
                # first, verify we have FSDP activation support ready by importing:
                from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
                    checkpoint_wrapper,
                    apply_activation_checkpointing,
                    CheckpointImpl,
                )

                non_reentrant_wrapper = functools.partial(
                    checkpoint_wrapper,
                    offload_to_cpu=False,
                    checkpoint_impl=CheckpointImpl.NO_REENTRANT,
                )
            except Exception as e:
                rank0_print("FSDP activation checkpointing not available:", e)
            else:
                check_fn = lambda submodule: isinstance(submodule, wrap_class)
                rank0_print(
                    "Applying activation checkpointing wrapper to policy..."
                )
                apply_activation_checkpointing(
                    self.policy,
                    checkpoint_wrapper_fn=non_reentrant_wrapper,
                    check_fn=check_fn,
                )
                rank0_print("FSDP activation checkpointing enabled!")

        if config.loss.name == "dpo":
            rank0_print("Sharding reference model...")
            self.reference_model = FSDP(reference_model, **shared_fsdp_kwargs)

        print("Loaded model on rank", rank)
        dist.barrier()

    def clip_gradient(self):
        """
        Clip the gradient norm of the parameters of an FSDP policy,
        gathering the gradients across all GPUs.
        """
        return self.policy.clip_grad_norm_(self.config.max_grad_norm).item()

    def save(self, output_dir=None, metrics=None):
        """
        Save policy, optimizer, and scheduler state to disk,
        gathering from all processes and saving only on the rank 0 process.
        """
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(
                self.policy,
                StateDictType.FULL_STATE_DICT,
                state_dict_config=save_policy,
        ):
            policy_state_dict = self.policy.state_dict()

        if self.rank == 0:
            self.write_state_dict(
                self.example_counter,
                policy_state_dict,
                metrics,
                "policy.pt",
                output_dir,
            )
        del policy_state_dict
        dist.barrier()

        save_policy = FullOptimStateDictConfig(
            offload_to_cpu=True, rank0_only=True
        )
        with FSDP.state_dict_type(
                self.policy,
                StateDictType.FULL_STATE_DICT,
                optim_state_dict_config=save_policy,
        ):
            optimizer_state_dict = FSDP.optim_state_dict(
                self.policy, self.optimizer
            )

        if self.rank == 0:
            self.write_state_dict(
                self.example_counter,
                optimizer_state_dict,
                metrics,
                "optimizer.pt",
                output_dir,
            )
        del optimizer_state_dict
        dist.barrier()

        if self.rank == 0:
            scheduler_state_dict = self.scheduler.state_dict()
            self.write_state_dict(
                self.example_counter,
                scheduler_state_dict,
                metrics,
                "scheduler.pt",
                output_dir,
            )
        dist.barrier()
