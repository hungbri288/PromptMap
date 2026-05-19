import asyncio
import hashlib
import time

from backend.app.config import Settings
from backend.app.schemas import PromptVariant, ResponseResult, RunRequest


async def run_inference(
    request: RunRequest,
    variants: list[PromptVariant],
    settings: Settings,
) -> list[ResponseResult]:
    if request.mode == "demo":
        return await _run_synthetic(request, variants)
    if request.mode == "mock":
        return await _run_mock_async(request, variants)
    if request.mode == "local":
        if not settings.allow_local_generation:
            raise RuntimeError("Local generation is disabled. Set allow_local_generation=True in settings.")
        return await _run_local(request, variants, settings)
    if request.mode == "live":
        if not settings.gemini_api_key:
            raise RuntimeError("Gemini live mode requires GEMINI_API_KEY.")
        return await _run_gemini(request, variants, settings)
    raise ValueError(f"Unsupported run mode: {request.mode}")


async def _run_gemini(
    request: RunRequest,
    variants: list[PromptVariant],
    settings: Settings,
) -> list[ResponseResult]:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=settings.gemini_api_key)
    semaphore = asyncio.Semaphore(settings.max_concurrent_api_calls)
    model = request.model or settings.gemini_model

    async def call(variant: PromptVariant) -> ResponseResult:
        started = time.perf_counter()
        async with semaphore:
            for attempt in range(3):
                try:
                    config = types.GenerateContentConfig(
                        temperature=request.temperature,
                        max_output_tokens=512,
                    )
                    if variant.system_prompt:
                        config.system_instruction = variant.system_prompt
                    response = await asyncio.to_thread(
                        client.models.generate_content,
                        model=model,
                        contents=variant.prompt,
                        config=config,
                    )
                    return ResponseResult(
                        variant_id=variant.id,
                        output=(response.text or "").strip(),
                        latency_ms=int((time.perf_counter() - started) * 1000),
                    )
                except Exception as exc:
                    if attempt == 2:
                        return ResponseResult(
                            variant_id=variant.id,
                            output="",
                            latency_ms=int((time.perf_counter() - started) * 1000),
                            error=str(exc),
                        )
                    await asyncio.sleep(0.75 * (attempt + 1))

        return ResponseResult(variant_id=variant.id, output="", latency_ms=0, error="unreachable")

    return await asyncio.gather(*(call(variant) for variant in variants))


async def _run_anthropic(
    request: RunRequest,
    variants: list[PromptVariant],
    settings: Settings,
) -> list[ResponseResult]:
    from anthropic import AsyncAnthropic

    client = AsyncAnthropic(api_key=settings.anthropic_api_key)
    semaphore = asyncio.Semaphore(settings.max_concurrent_api_calls)
    model = request.model or settings.anthropic_model

    async def call(variant: PromptVariant) -> ResponseResult:
        started = time.perf_counter()
        async with semaphore:
            for attempt in range(3):
                try:
                    response = await client.messages.create(
                        model=model,
                        max_tokens=512,
                        temperature=request.temperature,
                        system=variant.system_prompt or None,
                        messages=[{"role": "user", "content": variant.prompt}],
                    )
                    text = "\n".join(
                        block.text for block in response.content if getattr(block, "type", None) == "text"
                    )
                    return ResponseResult(
                        variant_id=variant.id,
                        output=text.strip(),
                        latency_ms=int((time.perf_counter() - started) * 1000),
                    )
                except Exception as exc:
                    if attempt == 2:
                        return ResponseResult(
                            variant_id=variant.id,
                            output="",
                            latency_ms=int((time.perf_counter() - started) * 1000),
                            error=str(exc),
                        )
                    await asyncio.sleep(0.75 * (attempt + 1))

        return ResponseResult(variant_id=variant.id, output="", latency_ms=0, error="unreachable")

    return await asyncio.gather(*(call(variant) for variant in variants))


async def _run_synthetic(request: RunRequest, variants: list[PromptVariant]) -> list[ResponseResult]:
    await asyncio.sleep(0)
    return [_synthetic_response(request, variant) for variant in variants]


async def _run_mock_async(request: RunRequest, variants: list[PromptVariant]) -> list[ResponseResult]:
    async def call(variant: PromptVariant) -> ResponseResult:
        started = time.perf_counter()
        digest = hashlib.sha256(f"mock:{request.seed}:{variant.id}:{variant.prompt}".encode()).hexdigest()
        await asyncio.sleep(0.12 + (int(digest[:2], 16) / 255.0) * 0.38)
        response = (
            "Mock async placeholder. This run uses a delayed offline response instead of a live model call. "
            f"Variant {variant.category} is being tracked with signal {digest[:6]}."
        )
        return ResponseResult(
            variant_id=variant.id,
            output=response,
            latency_ms=max(5, int((time.perf_counter() - started) * 1000)),
        )

    return await asyncio.gather(*(call(variant) for variant in variants))


def _synthetic_response(request: RunRequest, variant: PromptVariant) -> ResponseResult:
    started = time.perf_counter()
    digest = hashlib.sha256(f"{request.seed}:{variant.id}:{variant.prompt}".encode()).hexdigest()
    style = int(digest[:2], 16) % 4
    emphasis = {
        "base": "balanced answer",
        "lexical": "wording-sensitive answer",
        "syntactic": "structure-sensitive answer",
        "persona": "role-framed answer",
        "politeness": "tone-sensitive answer",
        "specificity": "constraint-sensitive answer",
        "negation": "negation-sensitive answer",
        "position": "placement-sensitive answer",
    }[variant.category]
    endings = [
        "The core recommendation stays stable, but details shift around wording.",
        "The answer changes emphasis and introduces a different priority order.",
        "The response becomes more cautious and adds caveats.",
        "The response is shorter and more operational.",
    ]
    output = (
        f"This is a {emphasis}. For the prompt request, the model would likely identify the main task, "
        f"produce an actionable response, and adapt tone based on the perturbation. {endings[style]} "
        f"Signal code {digest[:6]}."
    )
    return ResponseResult(
        variant_id=variant.id,
        output=output,
        latency_ms=max(5, int((time.perf_counter() - started) * 1000)),
    )


_local_pipeline = None
_local_lock = asyncio.Lock()


async def _run_local(request: RunRequest, variants: list[PromptVariant], settings: Settings) -> list[ResponseResult]:
    """Generate responses using a local Hugging Face model. Loads model lazily and uses CUDA when available.

    Note: model weights must be present or downloadable and this may require substantial GPU memory.
    """
    import torch

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    except Exception as exc:
        raise RuntimeError("transformers is required for local generation: pip install transformers") from exc

    model_name = settings.local_model or "gpt2"
    device = 0 if torch.cuda.is_available() else -1

    async with _local_lock:
        global _local_pipeline
        if _local_pipeline is None:
            loop = asyncio.get_running_loop()

            def init_pipeline():
                tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
                # Use device_map="auto" to place model on GPU if available
                model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto" if torch.cuda.is_available() else None)
                return pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

            _local_pipeline = await loop.run_in_executor(None, init_pipeline)

    async def call(variant: PromptVariant) -> ResponseResult:
        started = time.perf_counter()

        def generate():
            # keep generation parameters conservative to limit memory/use
            out = _local_pipeline(variant.prompt, max_new_tokens=256, do_sample=False, temperature=float(request.temperature), return_full_text=False)[0]
            # Pipeline may return dict or string depending on transformers version
            if isinstance(out, dict):
                return out.get("generated_text", "")
            return str(out)

        loop = asyncio.get_running_loop()
        try:
            text = await loop.run_in_executor(None, generate)
        except Exception as exc:
            return ResponseResult(variant_id=variant.id, output="", latency_ms=int((time.perf_counter() - started) * 1000), error=str(exc))

        return ResponseResult(variant_id=variant.id, output=(text or "").strip(), latency_ms=max(5, int((time.perf_counter() - started) * 1000)))

    return await asyncio.gather(*(call(v) for v in variants))
