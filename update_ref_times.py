#!/usr/bin/env python3
"""
update_ref_times.py — 更新 ref_times.json 中的参考实现耗时记录

用法:
  python3 update_ref_times.py -t ref/001_LeakyReLU.py
  python3 update_ref_times.py -t 001_LeakyReLU
  python3 update_ref_times.py --all

说明:
  - 单题更新: 仅重跑一个参考实现并更新对应记录
  - 全量重跑: 遍历 ref/ 下所有题目并重写/补全记录
  - 输出格式与 ref_times.json 保持一致
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REF_DIR = SCRIPT_DIR / "ref"
REF_TIMES_PATH = SCRIPT_DIR / "ref_times.json"


def _load_ref_times() -> dict[str, dict]:
    if not REF_TIMES_PATH.exists():
        return {}
    with open(REF_TIMES_PATH, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{REF_TIMES_PATH} 顶层必须是 JSON object")
    return data


def _write_ref_times(data: dict[str, dict]) -> None:
    ordered = dict(sorted(data.items(), key=lambda item: item[0]))
    with open(REF_TIMES_PATH, "w", encoding="utf-8") as f:
        json.dump(ordered, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _resolve_problem_path(target: str) -> Path:
    raw = Path(target)
    candidates: list[Path] = []

    if raw.exists():
        candidates.append(raw)

    if not raw.suffix:
        candidates.append(REF_DIR / f"{raw.name}.py")

    candidates.append(REF_DIR / raw.name)

    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists() and candidate.is_file():
            return candidate

    raise FileNotFoundError(f"找不到题目文件: {target}")


def _list_all_problem_paths() -> list[Path]:
    return sorted(REF_DIR.glob("*.py"))


def _measure_problem(problem_path: Path, iterations: int) -> dict:
    import torch_tester

    result = torch_tester.run(str(problem_path), iterations)
    if result.get("error"):
        raise RuntimeError(result["error"])

    return {
        "torch_us": result["torch_us"],
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "iterations": iterations,
        "output_shape": result.get("output_shape"),
        "output_dtype": result.get("output_dtype"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="更新 ref_times.json")
    parser.add_argument(
        "-t",
        "--torch-model",
        help="单个题目路径或题目名，例如 ref/001_LeakyReLU.py 或 001_LeakyReLU",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="重跑 ref/ 下所有题目并更新 ref_times.json",
    )
    parser.add_argument(
        "-n",
        "--iterations",
        type=int,
        default=10,
        help="性能测量迭代次数（默认 10）",
    )
    args = parser.parse_args()

    if bool(args.torch_model) == bool(args.all):
        parser.error("必须且只能指定 --torch-model 或 --all 其中之一")

    if args.iterations <= 0:
        parser.error("--iterations 必须大于 0")

    import torch

    if not torch.mlu.is_available():
        print("MLU 不可用", file=sys.stderr)
        sys.exit(1)

    ref_times = _load_ref_times()

    if args.all:
        problem_paths = _list_all_problem_paths()
    else:
        problem_paths = [_resolve_problem_path(args.torch_model)]

    failures: list[tuple[str, str]] = []

    for problem_path in problem_paths:
        name = problem_path.stem
        print(f"[RUN] {name}", file=sys.stderr)
        try:
            ref_times[name] = _measure_problem(problem_path, args.iterations)
            print(
                f"[OK] {name}: {ref_times[name]['torch_us']:.3f} us",
                file=sys.stderr,
            )
        except Exception as exc:
            failures.append((name, str(exc)))
            print(f"[FAIL] {name}: {exc}", file=sys.stderr)

    _write_ref_times(ref_times)
    print(f"[WRITE] {REF_TIMES_PATH}", file=sys.stderr)

    if failures:
        print("", file=sys.stderr)
        print("以下题目更新失败:", file=sys.stderr)
        for name, message in failures:
            print(f"  - {name}: {message}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
