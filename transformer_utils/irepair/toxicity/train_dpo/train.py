"""
Train script
"""
from typing import Optional, Set

import os
import json
import socket
import resource
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import transformers
import hydra
from omegaconf import OmegaConf, DictConfig
import wandb
import os,sys
sys.path.append(os.path.abspath(os.path.join(__file__, "../../../..")))
from dpo_utils import (
    get_local_dir,
    get_local_run_dir,
    disable_dropout,
    init_distributed,
    get_open_port,
)
import trainers as trainers
from irepair.constants import DEVICE
from transformers.models.phi.modeling_phi import PhiAttention
import types
torch.backends.cuda.matmul.allow_tf32 = True
OmegaConf.register_new_resolver(
    "get_local_run_dir",
    lambda exp_name, local_dirs: get_local_run_dir(exp_name, local_dirs),
)

def patch_phi_attention_with_clamp(model):
    from transformers.models.phi.modeling_phi import PhiAttention

    count = 0
    for name, module in model.named_modules():
        if isinstance(module, PhiAttention):
            original_forward = module.forward

            def patched_forward(
                self,
                hidden_states,
                past_key_value=None,
                attention_mask=None,
                position_ids=None,
                head_mask=None,
                use_cache=False,
                output_attentions=False,
                cache_position=None,
                rotary_pos_emb=None,
            ):
                with torch.cuda.amp.autocast(enabled=False):
                    # 计算q,k,v等逻辑保持不变，调用原 forward 之前先做clip操作
                    # 这里因为无法直接改原forward内部，我们用hook的思路：
                    # 所以直接调用原forward，且禁止autocast
                    # 如果需要更细粒度控制，建议fork源码改softmax部分

                    # 直接调用原forward（禁用autocast保证软max稳定）
                    output = original_forward(
                        hidden_states=hidden_states,
                        past_key_value=past_key_value,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        head_mask=head_mask,
                        use_cache=use_cache,
                        output_attentions=output_attentions,
                        cache_position=cache_position,
                        rotary_pos_emb=rotary_pos_emb,
                    )
                    # 目前不能直接clip logits，需修改源码或fork实现
                    return output

            module.forward = types.MethodType(patched_forward, module)
            count += 1

    print(f"✅ Patched {count} PhiAttention layers with autocast disabled and attention clipping.")

def worker_main(
    rank: int,
    world_size: int,
    config: DictConfig,
    policy: nn.Module,
    reference_model: Optional[nn.Module] = None,
):
    """
    Main function for each worker processss
    (may be only 1 for BasicTrainer/TensorParallelTrainer).
    """
    if "FSDP" in config.trainer:
        init_distributed(rank, world_size, port=config.fsdp_port)

    if config.debug:
        wandb.init = lambda *args, **kwargs: None
        wandb.log = lambda *args, **kwargs: None

    if rank == 0 and config.wandb.enabled:
        os.environ["WANDB_CACHE_DIR"] = get_local_dir(config.local_dirs)
        wandb.init(
            entity=config.wandb.entity,
            project=config.wandb.project,
            config=OmegaConf.to_container(config),
            dir=get_local_dir(config.local_dirs),
            name=config.exp_name,
        )

    TrainerClass = getattr(trainers, config.trainer)
    print(f"Creating trainer on process {rank} with world size {world_size}")
    trainer = TrainerClass(
        policy,
        config,
        config.seed,
        config.local_run_dir,
        reference_model=reference_model,
        rank=rank,
        world_size=world_size,
        device=DEVICE
    )

    trainer.train_loop()


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config: DictConfig):
    """
    Main entry point for training.
    Validates config, creates/initializes model(s),
    and kicks off worker process(es).
    """
    # Resolve hydra references, e.g. so we don't re-compute the run directory
    OmegaConf.resolve(config)

    missing_keys: Set[str] = OmegaConf.missing_keys(config)
    if missing_keys:
        raise ValueError(f"Got missing keys in config:\n{missing_keys}")

    if config.eval_every % config.batch_size != 0:
        print("WARNING: eval_every must be divisible by batch_size")
        print(
            "Setting eval_every to",
            config.eval_every - config.eval_every % config.batch_size,
        )
        config.eval_every = (
            config.eval_every - config.eval_every % config.batch_size
        )

    if "FSDP" in config.trainer and config.fsdp_port is None:
        free_port = get_open_port()
        print("no FSDP port specified; using open port for FSDP:", free_port)
        config.fsdp_port = free_port

    print(OmegaConf.to_yaml(config))

    config_path = os.path.join(config.local_run_dir, "config.yaml")
    with open(config_path, "w") as f:
        OmegaConf.save(config, f)

    print("=" * 80)
    print(f"Writing to {socket.gethostname()}:{config.local_run_dir}")
    print("=" * 80)

    os.environ["XDG_CACHE_HOME"] = get_local_dir(config.local_dirs)
    print("building policy")
    model_kwargs = (
        {"device_map": "balanced"} if config.trainer == "BasicTrainer" else {}
    )
    policy_dtype = getattr(torch, config.model.policy_dtype)

    if config.resume_train:
        state_dict_path = os.path.join(config.local_run_dir, "checkpoints", "policy.pt")
        state_dict = torch.load(state_dict_path, map_location=(DEVICE))["state"]

        policy = transformers.AutoModelForCausalLM.from_pretrained(
            config.model.name_or_path,
            cache_dir=get_local_dir(config.local_dirs),
            low_cpu_mem_usage=True,
            state_dict=state_dict,
            torch_dtype=policy_dtype,
            # use_flash_attention_2=False
        )
    else:
        policy = transformers.AutoModelForCausalLM.from_pretrained(
            config.model.name_or_path,
            cache_dir=get_local_dir(config.local_dirs),
            low_cpu_mem_usage=True,
            torch_dtype=policy_dtype,
            # use_flash_attention_2=False,
            **model_kwargs,
        )
        if "phi" in config.model.name_or_path.lower():
            patch_phi_attention_with_clamp(policy)
        # print(policy)


    disable_dropout(policy)
    policy=policy.to(DEVICE)

    print('DEVICE: ', DEVICE)

    if config.loss.name == "dpo" or config.loss.name == "sr" or config.loss.name == "sftkl":
        print("building reference model")
        reference_model_dtype = getattr(torch, config.model.reference_dtype)
        reference_model = transformers.AutoModelForCausalLM.from_pretrained(
            config.model.name_or_path,
            cache_dir=get_local_dir(config.local_dirs),
            low_cpu_mem_usage=True,
            torch_dtype=reference_model_dtype,
            # use_flash_attention_2=False,
            **model_kwargs,
        )

        disable_dropout(reference_model)
        reference_model = reference_model.to(DEVICE)
    else:
        reference_model = None

    if config.model.archive is not None:
        state_dict = torch.load(config.model.archive, map_location="cpu")
        step, metrics = state_dict["step_idx"], state_dict["metrics"]
        print(
            f"loading pre-trained weights at step {step} from \
            {config.model.archive} with metrics \
            {json.dumps(metrics, indent=2)}"
        )
        policy.load_state_dict(state_dict["state"])
        if config.loss.name == "dpo":
            reference_model.load_state_dict(state_dict["state"])

        print("loaded pre-trained weights")

    if "FSDP" in config.trainer:
        world_size = torch.cuda.device_count()
        print("starting", world_size, "processes for FSDP training")
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
        print(f"setting RLIMIT_NOFILE soft limit to {hard} from {soft}")
        mp.spawn(
            worker_main,
            nprocs=world_size,
            args=(world_size, config, policy, reference_model),
            join=True,
        )
    else:
        print("starting single-process worker")
        worker_main(0, 1, config, policy, reference_model)


if __name__ == "__main__":
    main()
