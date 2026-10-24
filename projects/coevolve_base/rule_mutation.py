"""
rule_mutation.py — symbolic rule mutation for coevolution
"""

import re
import random
from pathlib import Path


def mutate_rules(rules_text: str, mutation_rate: float = 0.2) -> tuple[str, int]:
    """
    Randomly mutates numeric thresholds or comparison signs
    in the exported decision tree text.

    Example:
      "x <= 0.4" -> "x <= 0.37" or "x > 0.4"
    Returns the mutated rules and the number of modified lines.
    """

    lines = rules_text.splitlines()
    new_lines = []
    mutations = 0

    for line in lines:
        original_line = line
        if "<=" in line or ">" in line:
            if random.random() < mutation_rate:
                # Extract numeric value
                nums = re.findall(r"[-+]?\d*\.\d+|\d+", line)
                if nums:
                    val = float(nums[0])
                    # Apply a small random offset
                    delta = random.uniform(-0.1, 0.1)
                    mutated_val = round(val + delta, 3)
                    line = re.sub(r"[-+]?\d*\.\d+|\d+", str(mutated_val), line, 1)

                # 10% chance to flip comparison direction
                if random.random() < 0.1:
                    line = line.replace("<=", ">TEMP")
                    line = line.replace(">", "<=")
                    line = line.replace(">TEMP", ">")

        if line != original_line:
            mutations += 1

        new_lines.append(line)

    return "\n".join(new_lines), mutations


def save_mutated_rules(
    rules_text: str,
    proj_path: Path,
    mutation_rate: float = 0.2,
    seed_rules: list[str] | None = None,
) -> tuple[Path, int]:
    """Save mutated rules to a file, injecting optional seed rules for reuse."""

    mutated, mutations = mutate_rules(rules_text, mutation_rate=mutation_rate)
    if seed_rules:
        seed_block = "\n".join(seed_rules)
        mutated = f"{mutated}\n\n# Memory Seeds\n{seed_block}"

    out_path = proj_path / "mutated_rules.txt"
    out_path.write_text(mutated, encoding="utf-8")
    return out_path, mutations
