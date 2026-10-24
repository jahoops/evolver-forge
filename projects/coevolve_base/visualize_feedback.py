from pathlib import Path
import math
from rich.console import Console

from .robustness import compute_robustness

console = Console()


def plot_coevolution_feedback(csv_path: Path):
    """Plot NN accuracy, rule fidelity, and combined fitness from metrics CSV."""
    import csv
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        console.print("[yellow]matplotlib not installed; skipping feedback plot.[/yellow]")
        return

    def _clean_float(value, default=0.0):
        if value in (None, "", "nan", "NaN"):
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    csv_path = Path(csv_path)
    if not csv_path.exists():
        console.print(f"[red]Metrics file not found: {csv_path}[/red]")
        return

    generations = []
    nn_accuracy = []
    rule_fidelity = []
    combined_fitness = []
    consistency_generations = []
    consistency_loss = []
    disagreement_generations = []
    avg_disagreement = []
    agreement_rate = []
    n_new_samples = []
    transfer_pairs = []
    transfer_metrics = []
    transfer_fidelities = []
    memory_generations = []
    memory_embeddings_used = []
    memory_rules_reused = []
    memory_boosts = []
    curriculum_runs = []
    curriculum_gains = []
    curriculum_labels = []
    meta_generations = []
    meta_learning_rates = []
    meta_mutation_rates = []
    meta_lambda_values = []
    meta_rewards = []
    arch_generations = []
    arch_labels = []
    composite_scores = []
    param_counts = []
    label_noise_series = []
    feature_noise_series = []
    target_noise_series = []
    label_smoothing_series = []
    eval_with_dropout_series = []
    is_probe_flags = []
    metrics_records = []

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                gen_val = int(row["generation"])
                generations.append(gen_val)
                metrics_records.append(dict(row))
                label_noise_series.append(_clean_float(row.get("label_noise")))
                feature_noise_series.append(_clean_float(row.get("feature_noise")))
                target_noise_series.append(_clean_float(row.get("target_noise")))
                label_smoothing_series.append(_clean_float(row.get("label_smoothing")))
                eval_with_dropout_series.append(int(_clean_float(row.get("eval_with_dropout")) > 0))
                is_probe_flags.append(bool(_clean_float(row.get("is_probe"))))
                nn_accuracy.append(float(row["nn_accuracy"]))
                rule_fidelity.append(float(row["rule_fidelity"]))
                combined_fitness.append(float(row["combined_fitness"]))
                raw_consistency = row.get("consistency_loss")
                if raw_consistency not in (None, ""):
                    try:
                        consistency_loss.append(float(raw_consistency))
                        consistency_generations.append(gen_val)
                    except ValueError:
                        pass

                raw_disagreement = row.get("avg_disagreement")
                if raw_disagreement not in (None, ""):
                    try:
                        avg_disagreement.append(float(raw_disagreement))
                        disagreement_generations.append(gen_val)
                        raw_agreement = row.get("rule_agreement_rate")
                        if raw_agreement not in (None, ""):
                            agreement_rate.append(float(raw_agreement))
                        raw_n_new = row.get("n_new_samples")
                        if raw_n_new not in (None, ""):
                            n_new_samples.append(int(float(raw_n_new)))
                    except ValueError:
                        pass

                from_task = row.get("from_task")
                to_task = row.get("to_task")
                boot_metric = row.get("boot_metric")
                if from_task and to_task and boot_metric not in (None, ""):
                    try:
                        transfer_pairs.append(f"{from_task}→{to_task}")
                        transfer_metrics.append(float(boot_metric))
                        fidelity_val = row.get("boot_fidelity")
                        if fidelity_val not in (None, ""):
                            transfer_fidelities.append(float(fidelity_val))
                        else:
                            transfer_fidelities.append(float("nan"))
                    except ValueError:
                        pass

                mem_embed_val = row.get("memory_embeddings_used")
                if mem_embed_val not in (None, ""):
                    try:
                        memory_generations.append(gen_val)
                        memory_embeddings_used.append(int(float(mem_embed_val)))
                        rules_val = row.get("memory_rules_reused")
                        if rules_val not in (None, ""):
                            memory_rules_reused.append(int(float(rules_val)))
                        else:
                            memory_rules_reused.append(0)
                        boost_val = row.get("memory_boost")
                        memory_boosts.append(float(boost_val) if boost_val not in (None, "") else 0.0)
                    except ValueError:
                        pass

                transfer_gain_val = row.get("transfer_gain")
                if transfer_gain_val not in (None, ""):
                    try:
                        gain_val = float(transfer_gain_val)
                        run_idx = len(curriculum_runs)
                        curriculum_runs.append(run_idx)
                        curriculum_gains.append(gain_val)
                        label_from = row.get("from_task", "?") or "?"
                        label_to = row.get("next_task") or row.get("to_task") or "?"
                        curriculum_labels.append(f"{label_from}→{label_to}")
                    except ValueError:
                        pass

                meta_lr = row.get("meta_learning_rate")
                meta_mut = row.get("meta_mutation_rate")
                meta_lambda = row.get("meta_lambda_consistency")
                if any(val not in (None, "") for val in (meta_lr, meta_mut, meta_lambda)):
                    try:
                        meta_generations.append(gen_val)
                        meta_learning_rates.append(float(meta_lr) if meta_lr not in (None, "") else float("nan"))
                        meta_mutation_rates.append(float(meta_mut) if meta_mut not in (None, "") else float("nan"))
                        meta_lambda_values.append(float(meta_lambda) if meta_lambda not in (None, "") else float("nan"))
                        reward_val = row.get("meta_reward")
                        meta_rewards.append(float(reward_val) if reward_val not in (None, "") else float("nan"))
                    except ValueError:
                        meta_generations.append(gen_val)
                        meta_learning_rates.append(float("nan"))
                        meta_mutation_rates.append(float("nan"))
                        meta_lambda_values.append(float("nan"))
                        meta_rewards.append(float("nan"))

                arch_layers_val = row.get("arch_hidden_layers")
                if arch_layers_val not in (None, ""):
                    try:
                        layers = int(float(arch_layers_val))
                        width = int(float(row.get("arch_hidden_width") or 0)) or layers
                        activation = (row.get("arch_activation") or "relu").strip()
                        normalization = (row.get("arch_normalization") or "none").strip()
                        dropout_val = float(row.get("arch_dropout") or 0.0)
                        depth = int(float(row.get("arch_max_depth") or 0))
                        leaf = int(float(row.get("arch_min_samples_leaf") or 0))
                        prune_alpha = float(row.get("arch_prune_alpha") or 0.0)
                        explore_k = int(float(row.get("arch_uncertainty_k") or 0))
                        explore_mut = float(row.get("arch_mutation_rate") or 0.0)
                        explore_lambda = float(row.get("arch_lambda_consistency") or 0.0)
                        comp_val = row.get("composite_score")
                        if comp_val not in (None, ""):
                            comp_score = float(comp_val)
                        else:
                            comp_score = float("nan")
                        params_val = row.get("model_param_count")
                        param_val = int(float(params_val)) if params_val not in (None, "") else 0
                        arch_generations.append(gen_val)
                        composite_scores.append(comp_score)
                        param_counts.append(param_val)
                        label = (
                            f"{layers}x{width} {activation}/{normalization} drop={dropout_val:.2f} | "
                            f"depth={depth} leaf={leaf} α={prune_alpha:.3f} | λ={explore_lambda:.2f} μ={explore_mut:.2f} k={explore_k}"
                        )
                        arch_labels.append(label)
                    except ValueError:
                        pass
            except (KeyError, ValueError) as exc:
                console.print(f"[yellow]Skipping row due to missing data: {exc}[/yellow]")

    if not generations:
        console.print("[yellow]No data to plot in metrics file.[/yellow]")
        return

    stability_scores: list[float] = []
    if metrics_records:
        running_records: list[dict] = []
        for record in metrics_records:
            running_records.append(record)
            try:
                robustness = compute_robustness(running_records)
                stability_scores.append(float(robustness.get("stability_score", 0.0)))
            except Exception:
                stability_scores.append(0.0)

    noise_banner = ""
    for record, probe_flag in reversed(list(zip(metrics_records, is_probe_flags))):
        if probe_flag:
            continue
        noise_banner = (
            "LN={ln:.2f} FN={fn:.2f} TN={tn:.2f} LS={ls:.2f} EWD={ewd}".format(
                ln=_clean_float(record.get("label_noise")),
                fn=_clean_float(record.get("feature_noise")),
                tn=_clean_float(record.get("target_noise")),
                ls=_clean_float(record.get("label_smoothing")),
                ewd=int(_clean_float(record.get("eval_with_dropout"))),
            )
        )
        break

    try:
        import numpy as np
    except ImportError:
        np = None
        console.print("[yellow]numpy not installed; trend lines disabled.[/yellow]")

    custom_legend_axes = set()

    fig, axs_raw = plt.subplots(8, 1, figsize=(8, 21), sharex=False)
    if np is not None:
        axes_seq = list(np.ravel(axs_raw))
    else:
        axes_seq = list(axs_raw if isinstance(axs_raw, (list, tuple)) else [axs_raw])

    ax_nn, ax_rule, ax_loss, ax_transfer, ax_memory, ax_curriculum, ax_meta, ax_arch = axes_seq[:8]
    fig.subplots_adjust(bottom=0.08)

    ax_nn.plot(generations, nn_accuracy, color="tab:blue", marker="o", label="NN Accuracy")
    ax_nn.plot(generations, combined_fitness, color="tab:blue", linestyle="--", marker="s", label="Combined Fitness")
    ax_nn.set_ylabel("NN Accuracy / Fitness")
    ax_nn.set_title("Neural & Rule Co-Evolution Metrics")

    ax_rule.plot(generations, rule_fidelity, color="tab:red", marker="^", label="Rule Fidelity")
    if agreement_rate:
        ax_rule.plot(disagreement_generations, agreement_rate, color="tab:red", linestyle="--", marker="v", label="Rule Agreement")
    ax_rule.set_ylabel("Rule Fidelity / Consistency")

    if avg_disagreement:
        ax_loss.plot(disagreement_generations, avg_disagreement, color="tab:orange", marker="d", label="Rule–NN Disagreement")
    if consistency_generations:
        ax_loss.plot(consistency_generations, consistency_loss, color="tab:purple", linestyle="--", marker="x", label="Consistency Loss")
    ax_loss.set_ylabel("Disagreement / Loss")
    ax_loss.set_xlabel("Generation")

    if avg_disagreement and np is not None:
        peak_idx = int(np.argmax(avg_disagreement))
        peak_gen = disagreement_generations[peak_idx]
        peak_val = avg_disagreement[peak_idx]
        ax_loss.annotate(
            f"Peak {peak_val:.3f}",
            xy=(peak_gen, peak_val),
            xytext=(peak_gen, peak_val + 0.02),
            textcoords="data",
            arrowprops=dict(arrowstyle="->", color="tab:orange"),
            color="tab:orange",
            fontsize=9,
            fontweight="bold",
        )

    if stability_scores:
        window = min(len(stability_scores), len(generations), 30)
        stab_vals = stability_scores[-window:]
        stab_gens = generations[-window:]
        ax_stability = ax_loss.twinx()
        ax_stability.plot(
            stab_gens,
            stab_vals,
            color="gray",
            linestyle="--",
            linewidth=1.0,
            alpha=0.6,
            label="Stability",
        )
        ax_stability.set_ylim(0.0, 1.05)
        ax_stability.set_ylabel("Stability", color="gray")
        ax_stability.tick_params(axis="y", colors="gray", labelsize=8)
        ax_stability.spines["right"].set_color("lightgray")
        custom_legend_axes.add(ax_stability)

    if transfer_pairs:
        indices = np.arange(len(transfer_pairs)) if np is not None else list(range(len(transfer_pairs)))
        ax_transfer.bar(indices, transfer_metrics, color="steelblue", label="Boot Metric")
        fidelity_series = transfer_fidelities
        ax_transfer.plot(indices, fidelity_series, "o-", color="orange", label="Boot Fidelity")
        ax_transfer.set_xticks(indices)
        ax_transfer.set_xticklabels(transfer_pairs, rotation=30, ha="right")
        ax_transfer.set_ylabel("Boot Metric")
        ax_transfer.set_title("Transfer Boot Performance")
    else:
        ax_transfer.axis("off")

    if memory_generations:
        ax_memory.bar(memory_generations, memory_embeddings_used, color="mediumseagreen", alpha=0.6, label="Embeddings Used")
        ax_memory.step(memory_generations, memory_rules_reused, where="mid", color="olive", linestyle="--", label="Rules Reused")
        ax_memory.plot(memory_generations, memory_boosts, color="darkgreen", marker="o", label="Memory Boost")
        ax_memory.set_ylabel("Memory Activity")
        ax_memory.set_xlabel("Generation")
        ax_memory.set_title("Memory Reuse & Knowledge Transfer")
    else:
        ax_memory.axis("off")

    if curriculum_gains:
        ax_curriculum.bar(curriculum_runs, curriculum_gains, color="slateblue", alpha=0.75, label="Transfer Gain")
        ax_curriculum.axhline(0.0, color="black", linewidth=0.8, linestyle="-")
        for idx, (x_pos, gain_val) in enumerate(zip(curriculum_runs, curriculum_gains)):
            label = curriculum_labels[idx] if idx < len(curriculum_labels) else ""
            if label:
                va = "bottom" if gain_val >= 0 else "top"
                offset = 0.01 if gain_val >= 0 else -0.01
                ax_curriculum.text(
                    x_pos,
                    gain_val + offset,
                    label,
                    ha="center",
                    va=va,
                    fontsize=8,
                )
        ax_curriculum.set_ylabel("Transfer Gain")
        ax_curriculum.set_xlabel("Run Index")
        ax_curriculum.set_title("Curriculum Dynamics")
    else:
        ax_curriculum.axis("off")

    if meta_generations and np is not None:
        ax_meta.set_title("Meta-Dynamics")
        ax_meta.set_xlabel("Generation")
        ax_meta.set_ylabel("Controller Params")

        def _plot_series(series_gens, series_vals, color, label):
            valid_pairs = [(g, v) for g, v in zip(series_gens, series_vals) if not math.isnan(v)]
            if not valid_pairs:
                return None
            gens, vals = zip(*valid_pairs)
            line = ax_meta.plot(gens, vals, color=color, marker="o", label=label)[0]
            if len(vals) >= 3:
                deltas = np.diff(vals)
                signs = np.sign(deltas)
                for idx in range(1, len(signs)):
                    if signs[idx] == 0 or signs[idx - 1] == 0:
                        continue
                    if signs[idx] != signs[idx - 1]:
                        flip_gen = gens[idx + 1]
                        flip_val = vals[idx + 1]
                        ax_meta.scatter(flip_gen, flip_val, color=color, marker="*", s=60)
            return line

        lr_line = _plot_series(meta_generations, meta_learning_rates, "tab:blue", "Learning Rate")
        mut_line = _plot_series(meta_generations, meta_mutation_rates, "tab:orange", "Mutation Rate")
        lam_line = _plot_series(meta_generations, meta_lambda_values, "tab:green", "Lambda Consistency")

        reward_axis = ax_meta.twinx()
        reward_axis.set_ylabel("Meta Reward")
        reward_valid = [(g, r) for g, r in zip(meta_generations, meta_rewards) if not math.isnan(r)]
        reward_line = None
        if reward_valid:
            reward_gens, reward_vals = zip(*reward_valid)
            window = min(5, len(reward_vals))
            if window <= 1:
                reward_trend = np.array(reward_vals, dtype=float)
                trend_gens = reward_gens
            else:
                kernel = np.ones(window) / window
                reward_trend = np.convolve(reward_vals, kernel, mode="valid")
                trend_gens = reward_gens[window - 1 :]
            reward_axis.plot(trend_gens, reward_trend, color="tab:red", linestyle="--", label="Meta-Reward Trend")
            reward_line = True
            reward_axis.axhline(0.0, color="gray", linewidth=0.8, linestyle=":")

        handles = [line for line in [lr_line, mut_line, lam_line] if line is not None]
        labels = [h.get_label() for h in handles]
        if reward_line:
            handles.append(reward_axis.lines[-1])
            labels.append("Meta-Reward Trend")
        if handles:
            ax_meta.legend(handles, labels, loc="best", fontsize=8)
            custom_legend_axes.add(ax_meta)
        ax_meta.grid(True, linestyle="--", alpha=0.4)
    else:
        ax_meta.axis("off")

    if arch_generations and composite_scores:
        valid_pairs = [
            (gen, score)
            for gen, score in zip(arch_generations, composite_scores)
            if not math.isnan(score)
        ]
        if valid_pairs:
            gens, scores = zip(*valid_pairs)
            ax_arch.plot(gens, scores, color="tab:green", marker="o", label="Composite Score")
        ax_arch.set_ylabel("Composite Score")
        ax_arch.set_xlabel("Generation")
        for idx, gen_val in enumerate(arch_generations):
            label = arch_labels[idx] if idx < len(arch_labels) else ""
            comp_score = composite_scores[idx] if idx < len(composite_scores) else float("nan")
            if label and not math.isnan(comp_score):
                ax_arch.text(
                    gen_val,
                    comp_score + 0.015,
                    label,
                    fontsize=7,
                    color="dimgray",
                    ha="left",
                )

        if param_counts:
            ax_arch_params = ax_arch.twinx()
            ax_arch_params.plot(
                arch_generations,
                param_counts,
                color="tab:brown",
                linestyle="--",
                marker="s",
                label="Param Count",
            )
            ax_arch_params.set_ylabel("Param Count")
            custom_legend_axes.add(ax_arch_params)
        ax_arch.grid(True, linestyle="--", alpha=0.4)
        custom_legend_axes.add(ax_arch)
    else:
        ax_arch.axis("off")

    if noise_banner:
        fig.text(0.5, 0.02, noise_banner, ha="center", fontsize=9, color="dimgray")

    for axis in axes_seq:
        if axis in custom_legend_axes:
            if axis.has_data():
                axis.grid(True, linestyle="--", alpha=0.5)
            continue
        if axis.has_data():
            axis.legend(loc="best", fontsize=8)
            axis.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    split_output = csv_path.with_name("coevolve_metrics_split.png")
    fig.savefig(split_output, dpi=150)
    plt.close(fig)

    # Optional dense overview for comparison
    dense_fig, ax_main = plt.subplots(figsize=(7.2, 4.8))
    ax_secondary = ax_main.twinx()

    ax_main.plot(generations, nn_accuracy, color="tab:blue", marker="o", label="NN Accuracy")
    ax_main.plot(generations, combined_fitness, color="tab:blue", linestyle="--", marker="s", label="Combined Fitness")
    ax_secondary.plot(generations, rule_fidelity, color="tab:red", marker="^", label="Rule Fidelity")
    if consistency_generations:
        ax_secondary.plot(consistency_generations, consistency_loss, color="tab:purple", linestyle=":", marker="x", label="Consistency Loss")
    if avg_disagreement:
        ax_secondary.plot(disagreement_generations, avg_disagreement, color="tab:orange", linestyle="-.", marker="d", label="Disagreement")

    ax_main.set_xlabel("Generation")
    ax_main.set_ylabel("NN Metrics (blue)")
    ax_secondary.set_ylabel("Rule / Loss Metrics")
    ax_main.grid(True, linestyle=":", linewidth=0.6)

    lines_main, labels_main = ax_main.get_legend_handles_labels()
    lines_sec, labels_sec = ax_secondary.get_legend_handles_labels()
    ax_main.legend(lines_main + lines_sec, labels_main + labels_sec, loc="best")

    dense_output = csv_path.with_name("coevolve_metrics_dense.png")
    dense_fig.tight_layout()
    dense_fig.savefig(dense_output, dpi=150)
    plt.close(dense_fig)

    avg_acc = sum(nn_accuracy) / len(nn_accuracy)
    avg_fidelity = sum(rule_fidelity) / len(rule_fidelity)
    summary_parts = [
        f"Avg NN accuracy: {avg_acc:.3f}",
        f"Avg rule fidelity: {avg_fidelity:.3f}",
    ]
    if consistency_generations:
        avg_consistency = sum(consistency_loss) / len(consistency_loss)
        summary_parts.append(f"Avg consistency loss: {avg_consistency:.3f}")
    if avg_disagreement:
        avg_dis = sum(avg_disagreement) / len(avg_disagreement)
        summary_parts.append(f"Avg disagreement: {avg_dis:.3f}")
    if memory_boosts:
        avg_memory_boost = sum(memory_boosts) / len(memory_boosts)
        summary_parts.append(f"Avg memory boost: {avg_memory_boost:.3f}")
    if curriculum_gains:
        avg_curriculum_gain = sum(curriculum_gains) / len(curriculum_gains)
        summary_parts.append(f"Avg transfer gain: {avg_curriculum_gain:.3f}")
    if meta_rewards:
        filtered_meta_rewards = []
        for value in meta_rewards:
            try:
                cast_value = float(value)
            except (TypeError, ValueError):
                continue
            if math.isnan(cast_value):
                continue
            filtered_meta_rewards.append(cast_value)
        if filtered_meta_rewards:
            avg_meta_reward = sum(filtered_meta_rewards) / len(filtered_meta_rewards)
            summary_parts.append(f"Avg meta reward: {avg_meta_reward:.3f}")
    console.print(f"[green]{', '.join(summary_parts)}[/green]")
    console.print(f"[green]Split metrics plot saved to {split_output}[/green]")
    console.print(f"[green]Dense overview saved to {dense_output}[/green]")
