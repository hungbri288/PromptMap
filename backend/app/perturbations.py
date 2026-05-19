import random
from collections.abc import Iterable

from backend.app.schemas import PerturbationCategory, PromptVariant


LEXICAL_SWAPS = {
    "explain": ["describe", "clarify", "summarize"],
    "write": ["draft", "compose", "create"],
    "make": ["produce", "build", "generate"],
    "good": ["strong", "effective", "useful"],
    "bad": ["weak", "risky", "poor"],
    "quickly": ["briefly", "efficiently", "concisely"],
}


def generate_variants(
    base_prompt: str,
    categories: Iterable[PerturbationCategory],
    sample_count: int,
    seed: int,
) -> list[PromptVariant]:
    rng = random.Random(seed)
    selected = list(categories)
    variants = [
        PromptVariant(
            id="base",
            prompt=base_prompt.strip(),
            category="base",
            transform="Original prompt",
        )
    ]

    generators = {
        "lexical": _lexical,
        "syntactic": _syntactic,
        "persona": _persona,
        "politeness": _politeness,
        "specificity": _specificity,
        "negation": _negation,
        "position": _position,
    }

    candidates: list[PromptVariant] = []
    for category in selected:
        candidates.extend(generators[category](base_prompt.strip()))

    rng.shuffle(candidates)
    wanted = max(0, sample_count - 1)
    for index, variant in enumerate(candidates[:wanted], start=1):
        variants.append(variant.model_copy(update={"id": f"v{index:03d}"}))

    while len(variants) < sample_count:
        category = selected[(len(variants) - 1) % len(selected)]
        prompt = _generic_variant(base_prompt, category, len(variants), rng)
        variants.append(
            PromptVariant(
                id=f"v{len(variants):03d}",
                prompt=prompt,
                category=category,
                transform=f"Seeded {category} variant",
            )
        )

    return variants


def _lexical(prompt: str) -> list[PromptVariant]:
    variants = []
    lowered = prompt.lower()
    for word, replacements in LEXICAL_SWAPS.items():
        if word in lowered:
            for replacement in replacements:
                variants.append(
                    PromptVariant(
                        id="",
                        prompt=_replace_case_insensitive(prompt, word, replacement),
                        category="lexical",
                        transform=f"Replace '{word}' with '{replacement}'",
                    )
                )
    variants.extend(
        [
            PromptVariant(id="", prompt=f"{prompt} Keep the answer concise.", category="lexical", transform="Add concise wording"),
            PromptVariant(id="", prompt=f"{prompt} Use plain language.", category="lexical", transform="Add plain language cue"),
        ]
    )
    return variants


def _syntactic(prompt: str) -> list[PromptVariant]:
    return [
        PromptVariant(id="", prompt=f"Please answer this request: {prompt}", category="syntactic", transform="Question to instruction"),
        PromptVariant(id="", prompt=f"The task is to {prompt[0].lower() + prompt[1:] if prompt else prompt}", category="syntactic", transform="Declarative framing"),
        PromptVariant(id="", prompt=f"{prompt}\nRespond as a numbered list.", category="syntactic", transform="List structure"),
        PromptVariant(id="", prompt=f"{prompt}\nRespond in one paragraph.", category="syntactic", transform="Paragraph structure"),
    ]


def _persona(prompt: str) -> list[PromptVariant]:
    personas = [
        "You are a cautious senior engineer.",
        "You are a direct product strategist.",
        "You are a skeptical reviewer.",
        "You are a patient teacher.",
    ]
    return [
        PromptVariant(id="", prompt=prompt, category="persona", transform=persona, system_prompt=persona)
        for persona in personas
    ]


def _politeness(prompt: str) -> list[PromptVariant]:
    return [
        PromptVariant(id="", prompt=f"Please {prompt[0].lower() + prompt[1:] if prompt else prompt}", category="politeness", transform="Polite prefix"),
        PromptVariant(id="", prompt=f"{prompt} Thanks.", category="politeness", transform="Polite suffix"),
        PromptVariant(id="", prompt=f"{prompt} Be direct.", category="politeness", transform="Direct style"),
        PromptVariant(id="", prompt=f"I would appreciate it if you could {prompt[0].lower() + prompt[1:] if prompt else prompt}", category="politeness", transform="Formal request"),
    ]


def _specificity(prompt: str) -> list[PromptVariant]:
    return [
        PromptVariant(id="", prompt=f"{prompt}\nInclude exactly three key points.", category="specificity", transform="Add count constraint"),
        PromptVariant(id="", prompt=f"{prompt}\nInclude assumptions, risks, and next steps.", category="specificity", transform="Add output requirements"),
        PromptVariant(id="", prompt=f"{prompt}\nAvoid unnecessary detail.", category="specificity", transform="Add brevity constraint"),
        PromptVariant(id="", prompt=f"{prompt}\nUse concrete examples where useful.", category="specificity", transform="Add example requirement"),
    ]


def _negation(prompt: str) -> list[PromptVariant]:
    return [
        PromptVariant(id="", prompt=f"{prompt}\nDo not over-explain.", category="negation", transform="Negative brevity constraint"),
        PromptVariant(id="", prompt=f"{prompt}\nDo not avoid tradeoffs.", category="negation", transform="Double-negative tradeoff cue"),
        PromptVariant(id="", prompt=f"{prompt}\nAvoid vague claims.", category="negation", transform="Avoid vagueness"),
        PromptVariant(id="", prompt=f"{prompt}\nDo not use bullet points.", category="negation", transform="Format prohibition"),
    ]


def _position(prompt: str) -> list[PromptVariant]:
    return [
        PromptVariant(id="", prompt=prompt, category="position", transform="Instruction in system prompt", system_prompt="Follow the user's request with concise precision."),
        PromptVariant(id="", prompt=f"Context: prioritize accuracy.\n\nUser request: {prompt}", category="position", transform="Context before user request"),
        PromptVariant(id="", prompt=f"{prompt}\n\nSystem-style note: prioritize accuracy.", category="position", transform="System note in user turn"),
        PromptVariant(id="", prompt=f"First identify the task, then answer.\n\n{prompt}", category="position", transform="Meta instruction before prompt"),
    ]


def _generic_variant(prompt: str, category: str, index: int, rng: random.Random) -> str:
    suffixes = [
        "Keep the answer compact.",
        "Prioritize practical details.",
        "State any uncertainty.",
        "Use a neutral tone.",
        "Focus on the final answer.",
    ]
    return f"{prompt}\n{category.title()} variation {index}: {rng.choice(suffixes)}"


def _replace_case_insensitive(text: str, old: str, new: str) -> str:
    index = text.lower().find(old.lower())
    if index == -1:
        return text
    return text[:index] + new + text[index + len(old) :]
