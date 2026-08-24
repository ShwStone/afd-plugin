# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Print the busy/bubble/CAM metrics from a profile_trace summary JSON."""

from __future__ import annotations

import json
import sys


def main() -> None:
    doc = json.load(open(sys.argv[1]))
    flat: dict[str, float] = {}

    def walk(node: object, prefix: str = "") -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                walk(value, f"{prefix}{key}.")
        elif isinstance(node, (int, float)):
            flat[prefix[:-1]] = node

    walk(doc)
    for key in sorted(flat):
        if any(token in key for token in ("busy", "bubble", "cam", "span", "ratio")):
            value = flat[key]
            if abs(value) > 10:
                print(f"  {key}: {value:.1f}")
            else:
                print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()
