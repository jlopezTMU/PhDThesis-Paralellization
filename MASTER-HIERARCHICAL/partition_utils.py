"""
DLMP: Deep Learning Multi-Processing Simulator

Author: Jorge A. Lopez
Affiliation: Toronto Metropolitan University

Description:
Shared utilities for dataset partitioning.

Repository:
https://github.com/DLMPsim/DLMP

License:
MIT License
"""

import numpy as np


def make_dirichlet_cifar10_indices(labels, num_processors, alpha=0.5, seed=42):
    labels = np.asarray(labels)
    num_classes = 10

    expected_per_agent = len(labels) // num_processors
    min_samples_per_agent = int(0.5 * expected_per_agent)
    min_classes_per_agent = 3
    max_attempts = 100

    rng = np.random.default_rng(int(seed))

    for attempt in range(max_attempts):
        agent_indices = [[] for _ in range(num_processors)]

        for class_id in range(num_classes):
            class_indices = np.where(labels == class_id)[0]
            rng.shuffle(class_indices)

            proportions = rng.dirichlet(
                np.full(num_processors, float(alpha))
            )

            split_points = (
                np.cumsum(proportions)[:-1] * len(class_indices)
            ).astype(int)

            class_splits = np.split(class_indices, split_points)

            for agent_id, split in enumerate(class_splits):
                agent_indices[agent_id].extend(split.tolist())

        valid = True

        for agent_id in range(num_processors):
            if len(agent_indices[agent_id]) < min_samples_per_agent:
                valid = False
                break

            class_counts = np.bincount(
                labels[agent_indices[agent_id]],
                minlength=num_classes
            )

            if np.count_nonzero(class_counts) < min_classes_per_agent:
                valid = False
                break

        if valid:
            print(
                f"Accepted non-IID CIFAR-10 partition "
                f"after {attempt + 1} attempt(s)."
            )

            for agent_id in range(num_processors):
                rng.shuffle(agent_indices[agent_id])

                class_counts = np.bincount(
                    labels[agent_indices[agent_id]],
                    minlength=num_classes
                )

                print(
                    f"Node {agent_id + 1} non-IID class distribution: "
                    f"{class_counts.tolist()} | total={len(agent_indices[agent_id])}"
                )

            return [
                np.asarray(idx, dtype=int)
                for idx in agent_indices
            ]

    raise RuntimeError(
        "Could not create a valid non-IID CIFAR-10 partition after "
        f"{max_attempts} attempts. Try increasing --dirichlet_alpha."
    )