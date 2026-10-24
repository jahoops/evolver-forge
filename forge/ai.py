# forge_ai.py
import time
import threading
import ollama
from rich.console import Console
from typing import Optional
from .constants import CURRENT_AI_MODEL

console = Console()

def ai_call(
    prompt: str,
    model: str,
    role: str = "user",
    system_instruction: Optional[str] = None,
    num_predict: int = 1024,
) -> str:
    """
    Unified AI call wrapper (spinner, timing, error handling).
    Caller provides the model explicitly (usually CURRENT_AI_MODEL).
    """
    stop = False

    def spinner():
        while not stop:
            console.print(".", end="", style="yellow", soft_wrap=False)
            time.sleep(0.4)

    t = threading.Thread(target=spinner)
    t.start()
    start = time.time()
    try:
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": role, "content": prompt})

        resp = ollama.chat(
            model=model,
            options={"num_predict": num_predict},
            messages=messages,
        )
        text = resp["message"]["content"].strip()
    except Exception as e:
        text = f"[Error calling {model}: {e}]"
    finally:
        stop = True
        t.join()
    console.print(f"\n[gray]Model {model} completed in {time.time() - start:.1f}s[/gray]")
    return text


def summarize_text(text: str, model_override: Optional[str] = None, context: Optional[str] = None) -> str:
    """
    Summarize experiment text with optional project context.
    If model_override is None, uses CURRENT_AI_MODEL.
    """
    model = model_override or CURRENT_AI_MODEL

    context_block = (
        f"Project Purpose:\n{context}\n\n"
        "Use this background to interpret the experiment log below.\n"
        "If the log is short or empty, summarize based only on the context.\n"
        if context else
        "Interpret the experiment log below. If it's short, be concise.\n"
    )

    # Light safety on huge logs
    if len(text) > 12000:
        console.print("[yellow]Truncating long log for summarization...[/yellow]")
        text = text[-12000:]

    full_prompt = f"{context_block}\n=== Experiment Log ===\n{text}"

    return ai_call(
        full_prompt,
        model=model,
        system_instruction="You are a precise technical summarizer.",
        num_predict=1024,
    )

def interpret_trend(summary_text: str, model: Optional[str] = None, context: Optional[str] = None) -> str:
    """
    Ask a model to interpret numeric metric trends (used after plotting or auto-runs).
    """
    model = model or CURRENT_AI_MODEL
    prompt = (
        "Given these numeric metrics, describe whether they show improvement, regression, or stability. "
        "Focus on direction and trend rather than exact numbers. Be concise and factual.\n\n"
        f"{summary_text}"
    )
    return ai_call(
        prompt,
        model=model,
        system_instruction="You are a data scientist interpreting experiment metrics.",
    )


def compare_summaries(old_summary: str, new_summary: str, model: Optional[str] = None) -> str:
    """
    Ask the AI to compare two experiment summaries and explain key differences in results or direction.
    """
    model = model or CURRENT_AI_MODEL
    prompt = (
        "Compare these two experiment summaries and explain key differences "
        "in results, direction, and possible implications.\n\n"
        f"Old Summary:\n{old_summary}\n\nNew Summary:\n{new_summary}"
    )
    return ai_call(
        prompt,
        model=model,
        system_instruction="You are an analytical assistant comparing experiment summaries.",
    )
