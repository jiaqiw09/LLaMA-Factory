# Copyright 2025 the LlamaFactory team.
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

from collections.abc import Iterable
from typing import Literal, TypedDict

from peft import LoraConfig, PeftModel, get_peft_model

from ...utils import logging
from ...utils.plugin import BasePlugin
from ...utils.types import HFModel


logger = logging.get_logger(__name__)


class LoraConfigDict(TypedDict, total=False):
    name: Literal["lora"]
    """Plugin name."""
    r: int
    """Lora rank."""
    lora_alpha: int
    """Lora alpha."""
    lora_dropout: float
    target_modules: list[str] | str | None
    adapter_name_or_path: list[str] | str | None
    create_new_adapter: bool


class FreezeConfigDict(TypedDict, total=False):
    name: Literal["freeze"]
    """Plugin name."""
    freeze_trainable_layers: int
    freeze_trainable_modules: list[str] | str | None
    """Freeze trainable modules."""


class PeftPlugin(BasePlugin):
    def __call__(self, model: HFModel, config: dict, is_train: bool) -> HFModel:
        return super().__call__(model, config, is_train)


def _as_list(value: list[str] | str | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def _merge_and_resume(model: HFModel, config: LoraConfigDict, is_train: bool) -> tuple[HFModel, str | None]:
    adapter_to_resume = None
    adapter_name_or_path = _as_list(config.get("adapter_name_or_path"))
    if adapter_name_or_path:
        if is_train and not config.get("create_new_adapter", False):
            adapter_to_merge = adapter_name_or_path[:-1]
            adapter_to_resume = adapter_name_or_path[-1]
        elif not is_train:
            adapter_to_merge = adapter_name_or_path
        else:
            adapter_to_merge = adapter_name_or_path

        for adapter in adapter_to_merge:
            model = PeftModel.from_pretrained(model, adapter)
            model = model.merge_and_unload()

        if adapter_to_merge:
            logger.info_rank0(f"Merged {len(adapter_to_merge)} adapter(s).")

        if adapter_to_resume is not None:
            model = PeftModel.from_pretrained(model, adapter_to_resume, is_trainable=is_train)
        logger.info_rank0("Loaded adapter(s): {}".format(",".join(adapter_name_or_path)))

    return model, adapter_to_resume


def _find_all_linear_modules(model: HFModel) -> list[str]:
    model_type = getattr(model.config, "model_type", None)
    forbidden_modules = {"lm_head"}
    if model_type == "chatglm":
        forbidden_modules.add("output_layer")
    elif model_type == "internlm2":
        forbidden_modules.add("output")

    module_names = set()
    for name, module in model.named_modules():
        if any(forbidden_module in name for forbidden_module in forbidden_modules):
            continue
        if "Linear" in module.__class__.__name__ and "Embedding" not in module.__class__.__name__:
            module_names.add(name.split(".")[-1])

    logger.info_rank0("Found linear modules: {}".format(",".join(sorted(module_names))))
    return list(module_names)


def _resolve_target_modules(model: HFModel, target_modules: list[str] | str | None) -> list[str]:
    modules = _as_list(target_modules)
    if not modules or (len(modules) == 1 and modules[0] == "all"):
        return _find_all_linear_modules(model)
    return modules


def _build_lora_config(config: LoraConfigDict, target_modules: list[str]) -> LoraConfig:
    if not target_modules:
        raise ValueError("No target modules found for LoRA.")
    lora_kwargs = {
        key: value
        for key, value in config.items()
        if key not in {"name", "adapter_name_or_path", "create_new_adapter", "target_modules"} and value is not None
    }
    lora_kwargs["target_modules"] = target_modules
    return LoraConfig(**lora_kwargs)


def _get_num_layers(model: HFModel) -> int:
    config = getattr(model.config, "text_config", None) or model.config
    num_layers = (
        getattr(config, "num_hidden_layers", None)
        or getattr(config, "num_layers", None)
        or getattr(config, "n_layer", None)
    )
    if not num_layers:
        raise ValueError("Current model does not support freeze tuning.")
    return num_layers


def _get_trainable_layer_ids(num_layers: int, freeze_trainable_layers: int) -> Iterable[int]:
    if freeze_trainable_layers > 0:
        return range(max(0, num_layers - freeze_trainable_layers), num_layers)
    return range(min(-freeze_trainable_layers, num_layers))


def _resolve_freeze_layers(model: HFModel, config: FreezeConfigDict) -> list[str]:
    num_layers = _get_num_layers(model)
    freeze_trainable_layers = config.get("freeze_trainable_layers") or 0
    trainable_layer_ids = _get_trainable_layer_ids(num_layers, freeze_trainable_layers)

    hidden_modules = set()
    for name, _ in model.named_parameters():
        if ".0." in name:
            hidden_modules.add(name.split(".0.")[-1].split(".")[0])
        elif ".1." in name:
            hidden_modules.add(name.split(".1.")[-1].split(".")[0])

    freeze_trainable_modules = _as_list(config.get("freeze_trainable_modules"))
    if not freeze_trainable_modules:
        freeze_trainable_modules = ["all"]

    trainable_layers = []
    for module_name in freeze_trainable_modules:
        if module_name != "all" and module_name not in hidden_modules:
            raise ValueError(
                "Module {} is not found, please choose from {}".format(module_name, ", ".join(hidden_modules))
            )
        for idx in trainable_layer_ids:
            trainable_layers.append(".{:d}.{}".format(idx, module_name if module_name != "all" else ""))
    return trainable_layers


@PeftPlugin("lora").register()
def get_lora_model(model: HFModel, config: LoraConfigDict, is_train: bool) -> PeftModel:
    model, adapter_to_resume = _merge_and_resume(model, config, is_train)
    if is_train and adapter_to_resume is None:
        target_modules = _resolve_target_modules(model, config.get("target_modules"))
        peft_config = _build_lora_config(config, target_modules)
        model = get_peft_model(model, peft_config)
    return model


@PeftPlugin("freeze").register()
def get_freeze_model(model: HFModel, config: FreezeConfigDict, is_train: bool) -> HFModel:
    if not is_train:
        return model

    logger.info_rank0("Fine-tuning method: Freeze")
    trainable_layers = _resolve_freeze_layers(model, config)
    for name, param in model.named_parameters():
        if any(trainable_layer in name for trainable_layer in trainable_layers):
            param.requires_grad_(True)
        else:
            param.requires_grad_(False)

    logger.info_rank0("Set trainable layers: {}".format(",".join(trainable_layers)))
    return model
