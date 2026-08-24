# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Print a compact summary of fixed-batch result files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path)
    args = parser.parse_args()
    for result_path in args.results:
        result = json.loads(result_path.read_text())
        summary = result["summary"]
        name = result_path.parent.name + "/" + result_path.name.removesuffix(
            ".fixed_batch.json"
        )
        print(
            f"{name}: tokens={result['batch_prompt_tokens']} "
            f"reqs={result['batch_requests']} "
            f"wall_s min={summary['wall_s_min']:.3f} "
            f"med={summary['wall_s_median']:.3f} "
            f"max={summary['wall_s_max']:.3f} "
            f"all_ok={summary['all_successful']}"
        )


if __name__ == "__main__":
    main()
