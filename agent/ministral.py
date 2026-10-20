"""Offline Transformers backend for the locally cached Ministral 3 VLM."""
import time
import gc

import torch
from transformers import (
    AutoConfig,
    AutoTokenizer,
    Mistral3ForConditionalGeneration,
    PixtralImageProcessor,
    PixtralProcessor,
    StoppingCriteria,
    StoppingCriteriaList,
)
from .protocol import GenerationCancelled
from .messages import attach_image

MODEL_ID = "mistralai/Ministral-3-3B-Instruct-2512"
FP8_KERNEL_REVISION = "8c178950e8710e26a2210e4e909cd02dc16c8715"


class _CancelOnEvent(StoppingCriteria):
    def __init__(self, event):
        self.event = event

    def __call__(self, input_ids, scores, **kwargs):
        return torch.full(
            (input_ids.shape[0],),
            self.event.is_set(),
            dtype=torch.bool,
            device=input_ids.device,
        )


class MinistralBackend:
    def __init__(self, model_path: str = MODEL_ID, image_size: int = 448):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA GPU unavailable. Run in mm_planner with GPU access."
            )
        if torch.cuda.get_device_capability() < (8, 9):
            raise RuntimeError("Native FP8 requires CUDA compute capability >= 8.9.")
        if image_size <= 0:
            raise ValueError("image_size must be positive")
        # Pin the compatible v4 build: kernels 0.17 resolves both legacy model
        # and kernel refs, and a colliding legacy v4 ref can select a wrong SHA.
        # This mapping is specific to the pinned Transformers 5.18 dependency.
        from transformers.integrations import hub_kernels

        hub_kernels._HUB_KERNEL_MAPPING["finegrained-fp8"] = {
            "repo_id": "kernels-community/finegrained-fp8",
            "revision": FP8_KERNEL_REVISION,
        }
        started = time.perf_counter()
        config = AutoConfig.from_pretrained(model_path, local_files_only=True)
        quantization = getattr(config, "quantization_config", {})
        if config.model_type != "mistral3" or quantization.get("quant_method") != "fp8":
            raise ValueError("Expected the official Mistral3 FP8 checkpoint.")
        if quantization.get("dequantize", False):
            raise ValueError("Expected native FP8 weights, dequantize=False.")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
            mistral_format=False,
            fix_mistral_regex=True,
        )
        # This cached snapshot has no preprocessor_config.json. Derive patch
        # geometry from the checkpoint instead of using Pixtral's 16px default.
        self.processor = PixtralProcessor(
            image_processor=PixtralImageProcessor(size={"longest_edge": image_size}),
            tokenizer=self.tokenizer,
            patch_size=config.vision_config.patch_size,
            spatial_merge_size=config.spatial_merge_size,
            chat_template=self.tokenizer.chat_template,
        )
        self.model = Mistral3ForConditionalGeneration.from_pretrained(
            model_path,
            config=config,
            local_files_only=True,
            trust_remote_code=False,
            device_map="cuda",
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        ).eval()
        self.fp8_tensors = sum(
            p.dtype == torch.float8_e4m3fn for p in self.model.parameters()
        )
        quantizer = getattr(self.model, "hf_quantizer", None)
        if (
            not self.fp8_tensors
            or quantizer is None
            or quantizer.quantization_config.dequantize
        ):
            raise RuntimeError("Checkpoint did not load as native FP8.")
        self.load_seconds = time.perf_counter() - started

    @torch.inference_mode()
    def generate(
        self,
        prompt: str | None = None,
        *,
        system: str | None = None,
        image=None,
        extra_images=None,
        max_new_tokens: int = 128,
        messages=None,
        image_message_index=None,
        cancel_event=None
    ) -> dict:
        if cancel_event is not None and cancel_event.is_set():
            raise GenerationCancelled("Generation cancelled before start")
        if messages is not None:
            if prompt is not None or system is not None:
                raise ValueError("Use messages or prompt/system, not both")
            messages = [dict(message) for message in messages]
        else:
            if not isinstance(prompt, str):
                raise ValueError("prompt must be a string")
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
        if not messages or messages[-1].get("role") != "user":
            raise ValueError("Generation messages must end with a user observation")
        if image is not None:
            messages = attach_image(
                messages, image_message_index, tuple(extra_images or {})
            )
        # Render the HF text chat template, then let PixtralProcessor expand
        # [IMG] into the correct number of image tokens for this checkpoint.
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if image is None:
            inputs = self.tokenizer(text, add_special_tokens=False, return_tensors="pt")
        else:
            inputs = self.processor(
                text=text,
                images=[image, *(extra_images or {}).values()],
                add_special_tokens=False,
                return_tensors="pt",
            )
        inputs = inputs.to("cuda")
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
        input_tokens = inputs["input_ids"].shape[-1]
        torch.cuda.synchronize()
        started = time.perf_counter()
        output = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
            stopping_criteria=StoppingCriteriaList([_CancelOnEvent(cancel_event)])
            if cancel_event is not None
            else None,
        )
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        generated = output[0, input_tokens:]
        if cancel_event is not None and cancel_event.is_set():
            raise GenerationCancelled("Generation cancelled; output discarded")
        return {
            "text": self.tokenizer.decode(generated, skip_special_tokens=True).strip(),
            "input_tokens": input_tokens,
            "generated_tokens": generated.numel(),
            "seconds": seconds,
            "tokens_per_second": generated.numel() / seconds,
        }

    def unload(self):
        """Call on the model worker, after generation has exited."""
        self.model = None
        self.processor = None
        self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
