"""
pro_mode.py — CLI *and* importable library for multi‑agent answer generation + synthesis
compatible with any OpenAI‑compatible back‑end (tested with a local Ollama server).
"""

from __future__ import annotations

from typing import List, Dict, Any
import time
from argparse import ArgumentParser
from openai import OpenAI, OpenAIError

DEFAULT_MODEL = "gpt-oss:20b"
DEFAULT_MAX_TOKENS = 30_000
DEFAULT_N_AGENTS = 5
DEFAULT_STREAM = True

client = OpenAI(base_url="http://localhost:11434/v1", api_key="") #i use ollama, but this may be cerebras, groq, openrouter, etc


def _stream_print(reasoning: str | None, content: str | None, sep_printed: bool) -> bool:
    if reasoning:
        print(reasoning, end="", flush=True)

    if content:
        if not sep_printed:
            print("\n---\n", end="", flush=True)
            sep_printed = True
        print(content, end="", flush=True)

    return sep_printed


def _one_completion(
    prompt: str,
    *,
    model: str,
    max_tokens: int,
    temperature: float,
    stream: bool,
) -> str:

    delay = 0.5
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=1,
                stream=stream,
            )

            if stream:
                sep_printed = False
                content_buffer: List[str] = []

                for chunk in resp:
                    delta = chunk.choices[0].delta
                    reasoning = getattr(delta, "reasoning", None)
                    content = getattr(delta, "content", None)

                    sep_printed = _stream_print(reasoning, content, sep_printed)

                    if content:
                        content_buffer.append(content)

                print() #stream finishes
                return "".join(content_buffer)

            return resp.choices[0].message.content

        except OpenAIError as e:
            if attempt == 2:
                raise
            print(f"Error: {e}. Retrying in {delay}s...")
            time.sleep(delay)
            delay *= 2


def _build_synthesis_messages(candidates: List[str]) -> List[Dict[str, str]]:
    numbered = "\n\n".join(
        f"<cand {i+1}>\n{txt}\n</cand {i+1}>" for i, txt in enumerate(candidates)
    )
    system = (
        "You are an expert editor. You are given several answers from candidates. "
        "Your task is to review the answers and synthesize ONE best answer from the "
        "candidate answers provided by merging them, merging strengths, correcting errors, "
        "and removing repetition. Do not mention the candidates or the synthesis process. "
        "Be decisive and clear."
    )
    user = (
        f"You are given {len(candidates)} candidate answers delimited by <cand i> tags.\n\n"
        f"{numbered}\n\nReturn the final answer."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def pro_mode(
    prompt: str,
    *,
    n_agents: int = DEFAULT_N_AGENTS,
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    stream_candidates: bool = DEFAULT_STREAM,
) -> Dict[str, Any]:
    """Run multi‑agent generation.

    Args:
        prompt: The user prompt.
        n_agents: Number of candidate answers to generate.
        model: Model name / id.
        max_tokens: Per‑request token cap.
        stream_candidates: Whether to stream candidate tokens to stdout.

    Returns:
        {
            "final": <synthesised answer>,
            "candidates": [list of raw candidates]
        }
    """
    assert n_agents >= 1, "`n_agents` must be ≥ 1"

    candidates: List[str] = []

    for i in range(n_agents):
        print(f"\n--- Generating candidate {i+1}/{n_agents} ---")
        try:
            candidate = _one_completion(
                prompt,
                model=model,
                max_tokens=max_tokens,
                temperature=0.9,
                stream=stream_candidates,
            )
            candidates.append(candidate)
        except Exception as e:
            print(f"Candidate generation {i+1} failed: {e}")
            candidates.append(f"Error: {e}")

        if i < n_agents - 1:
            time.sleep(0.5)

    print("\n--- Synthesizing final answer ---")
    messages = _build_synthesis_messages(candidates)
    synthesis_prompt = f"{messages[0]['content']}\n\n{messages[1]['content']}"
    final = _one_completion(
        synthesis_prompt,
        model=model,
        max_tokens=max_tokens,
        temperature=0.2,
        stream=True,
    )

    return {"final": final, "candidates": candidates}


def _parse_cli() -> tuple[str, dict[str, Any]]:
    parser = ArgumentParser(description="Multi‑agent LLM ‘pro mode’.")
    parser.add_argument("prompt", nargs="+", help="Prompt to send to the model")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name / id")
    parser.add_argument("--max_tokens", type=int, default=DEFAULT_MAX_TOKENS, help="Max completion tokens")
    parser.add_argument("--n_agents", type=int, default=DEFAULT_N_AGENTS, help="Number of candidate agents")
    parser.add_argument("--no_stream", action="store_true", help="Disable token streaming")

    ns = parser.parse_args()
    kwargs = {
        "n_agents": ns.n_agents,
        "model": ns.model,
        "max_tokens": ns.max_tokens,
        "stream_candidates": not ns.no_stream
    }
    return " ".join(ns.prompt), kwargs


def main() -> None:
    prompt, kwargs = _parse_cli()
    result = pro_mode(prompt, **kwargs)
    print(f"\n=== FINAL ===\n{result['final']}")


if __name__ == "__main__":
    main()
