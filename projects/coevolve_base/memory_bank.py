"""Persistent memory bank for cross-task knowledge transfer."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class MemoryConfig:
    embeddings_key: str = "embeddings"
    rules_key: str = "rules"
    labels_key: str = "labels"
    meta_key: str = "meta"


class MemoryBank:
    """Lightweight persistent store of reusable embeddings and rules."""

    def __init__(self, path: Path, config: MemoryConfig | None = None) -> None:
        self.path = Path(path)
        self.config = config or MemoryConfig()
        self.data: Dict[str, Any] = {
            self.config.embeddings_key: [],
            self.config.rules_key: [],
            self.config.labels_key: [],
            self.config.meta_key: [],
        }
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            for key in (
                self.config.embeddings_key,
                self.config.rules_key,
                self.config.labels_key,
                self.config.meta_key,
            ):
                self.data[key] = payload.get(key, [])
        except (json.JSONDecodeError, OSError):
            # Fall back to empty memory if file is corrupt or unreadable.
            self.data = {
                self.config.embeddings_key: [],
                self.config.rules_key: [],
                self.config.labels_key: [],
                self.config.meta_key: [],
            }

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        serialisable = {
            self.config.embeddings_key: self.data.get(self.config.embeddings_key, []),
            self.config.rules_key: self.data.get(self.config.rules_key, []),
            self.config.labels_key: self.data.get(self.config.labels_key, []),
            self.config.meta_key: self.data.get(self.config.meta_key, []),
        }
        with self.path.open("w", encoding="utf-8") as handle:
            json.dump(serialisable, handle, indent=2)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(
        self,
        task_name: str,
        embeddings: np.ndarray,
        rules: List[str],
        meta: Dict[str, Any],
        labels: Optional[np.ndarray] = None,
    ) -> None:
        """Persist embeddings, optional labels, symbolic rules, and metadata."""

        embeddings_array = np.asarray(embeddings, dtype=np.float32)
        if embeddings_array.ndim == 1:
            embeddings_array = embeddings_array.reshape(1, -1)

        self.data[self.config.embeddings_key].append(embeddings_array.tolist())
        self.data[self.config.rules_key].append(list(rules))

        labels_array: Optional[np.ndarray]
        if labels is None:
            labels_array = None
        else:
            labels_array = np.asarray(labels)
            if labels_array.ndim == 0:
                labels_array = np.repeat(labels_array, embeddings_array.shape[0])
            else:
                labels_array = labels_array.reshape(-1)
            if labels_array.shape[0] != embeddings_array.shape[0]:
                labels_array = np.resize(labels_array, embeddings_array.shape[0])
        self.data[self.config.labels_key].append([] if labels_array is None else labels_array.tolist())

        enriched_meta = dict(meta)
        enriched_meta.setdefault("task", task_name)
        self.data[self.config.meta_key].append(enriched_meta)
        self._save()

    def sample_pairs(
        self,
        k: int = 128,
        feature_dim: Optional[int] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Return up to *k* (embedding, label) pairs aggregated across all tasks."""

        stored_embeddings = self.data.get(self.config.embeddings_key, [])
        stored_labels = self.data.get(self.config.labels_key, [])
        if not stored_embeddings:
            empty_X = np.zeros((0, 0), dtype=np.float32) if feature_dim is None else np.zeros((0, feature_dim), dtype=np.float32)
            return empty_X, None

        matrices: List[np.ndarray] = []
        labels_list: List[np.ndarray] = []

        for idx, entry in enumerate(stored_embeddings):
            arr = np.asarray(entry, dtype=np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            elif arr.ndim > 2:
                arr = arr.reshape(-1, arr.shape[-1])

            matrices.append(arr)

            raw_labels = stored_labels[idx] if idx < len(stored_labels) else []
            if raw_labels:
                lbl_arr = np.asarray(raw_labels)
                lbl_arr = lbl_arr.reshape(-1)
                if lbl_arr.shape[0] != arr.shape[0]:
                    lbl_arr = np.resize(lbl_arr, arr.shape[0])
                labels_list.append(lbl_arr)
            else:
                labels_list.append(np.full(arr.shape[0], np.nan, dtype=np.float32))

        if not matrices:
            empty_X = np.zeros((0, 0), dtype=np.float32) if feature_dim is None else np.zeros((0, feature_dim), dtype=np.float32)
            return empty_X, None

        stacked = np.vstack(matrices)
        labels_stacked = np.concatenate(labels_list) if labels_list else np.asarray([], dtype=np.float32)

        if feature_dim is not None and stacked.shape[1] != feature_dim:
            # If dimensionality mismatches the requested shape, truncate or pad.
            if stacked.shape[1] > feature_dim:
                stacked = stacked[:, :feature_dim]
            else:
                pad_width = feature_dim - stacked.shape[1]
                stacked = np.pad(stacked, ((0, 0), (0, pad_width)), mode="constant")

        total = len(stacked)
        if total == 0:
            return stacked, None if labels_stacked.size == 0 else labels_stacked

        sample_size = min(k, total)
        indices = np.random.choice(total, sample_size, replace=False)
        sampled_X = stacked[indices]
        sampled_labels = labels_stacked[indices] if labels_stacked.size == total else labels_stacked

        if sampled_labels is not None and np.asarray(sampled_labels).size:
            sampled_labels = np.asarray(sampled_labels)
            if sampled_labels.dtype.kind in {"f", "c"} and np.isnan(sampled_labels).all():
                sampled_labels = None
        else:
            sampled_labels = None

        return sampled_X, sampled_labels

    def sample_embeddings(self, k: int = 128, feature_dim: Optional[int] = None) -> np.ndarray:
        """Backward compatible helper returning only embeddings."""

        embeddings, _ = self.sample_pairs(k=k, feature_dim=feature_dim)
        return embeddings

    def get_rules(self, top_n: int = 10) -> List[str]:
        """Return a list of up to *top_n* recent rules for reuse."""

        stored_rules = self.data.get(self.config.rules_key, [])
        if not stored_rules:
            return []
        recent = stored_rules[-top_n:]
        result: List[str] = []
        for group in recent:
            result.extend(str(rule) for rule in group if rule)
        return result

    def task_meta(self, task_name: str) -> List[Dict[str, Any]]:
        """Retrieve metadata entries recorded for a given task."""

        entries = self.data.get(self.config.meta_key, [])
        return [entry for entry in entries if entry.get("task") == task_name]

    def average_metric(self, task_name: str, key: str) -> Optional[float]:
        """Return the average value of *key* for the specified task."""

        entries = self.task_meta(task_name)
        values = [entry[key] for entry in entries if key in entry and isinstance(entry[key], (int, float))]
        if not values:
            return None
        return float(np.mean(values))
