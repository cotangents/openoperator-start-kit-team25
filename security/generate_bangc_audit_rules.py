#!/usr/bin/env python3
"""Generate a static BangC audit rules file from config.yaml and ref/."""

from __future__ import annotations

import argparse
import ast
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import yaml


GENERAL_RULES = [
    {
        "id": "FORBIDDEN_CNNL_HEADER",
        "pattern": r"#\s*include\s*<\s*cnnl(?:/[\w.\-]+)?\.h\s*>",
        "flags": re.IGNORECASE,
        "message": "禁止包含 CNNL 头文件（如 cnnl.h）；比赛要求不得直接调用 CNNL 高级算子。",
    },
    {
        "id": "FORBIDDEN_ATEN_CNNL_HEADER",
        "pattern": r"#\s*include\s*<\s*aten/cnnl/[^>]+\s*>",
        "flags": re.IGNORECASE,
        "message": "禁止包含 aten/cnnl/* 头文件；不得通过 ATen/CNNL 封装间接调用现成算子。",
    },
    {
        "id": "FORBIDDEN_TORCH_MODULE_API",
        "pattern": r"\btorch::(nn|jit|autograd|optim)::",
        "flags": 0,
        "message": "禁止直接使用高层 torch 模块 API：torch::nn:: / torch::jit:: / torch::autograd:: / torch::optim::。",
    },
    {
        "id": "FORBIDDEN_TORCH_HIGH_LEVEL_OP",
        "pattern": None,
        "flags": 0,
        "message": "禁止直接调用 torch:: / at:: 下的现成算子实现，例如 at::conv(...)、torch::conv(...)。",
    },
    {
        "id": "FORBIDDEN_DYNAMIC_LOADING",
        "pattern": r"\b(dlopen|dlsym|dlmopen)\b",
        "flags": 0,
        "message": "禁止在选手 .mlu 中动态加载外部符号。",
    },
    {
        "id": "FORBIDDEN_PROCESS_SPAWN",
        "pattern": r"\b(system|popen|fork|execve|execv|execl|posix_spawn)\b",
        "flags": 0,
        "message": "禁止在选手 .mlu 中启动进程或执行 shell。",
    },
    {
        "id": "FORBIDDEN_CNRT_QUEUE_CREATE",
        "pattern": r"\bcnrtQueueCreate\s*\(",
        "flags": 0,
        "message": "请使用cnrtQueue tqueue =torch mlu::getCurMLustream();获取当前队列环境，禁止重新创建任务队列。",
    },
    {
        "id": "FORBIDDEN_CNNL_ADVANCED_OP",
        "pattern": r"\bcnnl[A-Z_]\w*\s*\(",
        "flags": 0,
        "message": "禁止直接调用 CNNL 高级算子或相关 API；涉及计算的实现只能使用 BangC 允许的 API 与基础数学库。",
    },
]

REF_BRIDGE_CALL_EXCLUSIONS = {
    # Runtime / bridge helpers that do not implement a high-level operator.
    "manual_seed",
    "getCurMLUStream",
    "getCurrentMLUStream",
    # Tensor construction / initialization helpers are treated as setup code
    # rather than a ready-made solution operator.
    "randn",
    "randint",
    "randperm",
    "rand",
}

REF_CONSTRUCTOR_EXCLUSIONS = {
    # Basic tensor/value constructors.
    "Tensor",
    "from_blob",
    "scalar_tensor",
    "tensor",
    "empty_like",
    "empty",
    "full_like",
    "full",
    "zeros_like",
    "zeros",
    "ones_like",
    "ones",
    "arange",
    "range",
    "linspace",
    "logspace",
    # Pure configuration/value-wrapper types that are commonly used to
    # describe tensor metadata rather than invoke a ready-made operator.
    "TensorOptions",
    "Device",
    "Scalar",
    "ScalarType",
    "Layout",
    "MemoryFormat",
    "Dimname",
    "IntArrayRef",
    "SymIntArrayRef",
    "ArrayRef",
    "SymInt",
}


def _build_forbidden_torch_high_level_op_pattern() -> str:
    allowed = "|".join(
        sorted(
            REF_CONSTRUCTOR_EXCLUSIONS | REF_BRIDGE_CALL_EXCLUSIONS,
            key=lambda item: (-len(item), item),
        )
    )
    return (
        r"\b(?:torch::|at::)"
        r"(?:[A-Za-z_]\w*::)*"
        rf"(?!(?:{allowed})\s*\()"
        r"[A-Za-z_]\w*\s*\("
    )


for rule in GENERAL_RULES:
    if rule["id"] == "FORBIDDEN_TORCH_HIGH_LEVEL_OP":
        rule["pattern"] = _build_forbidden_torch_high_level_op_pattern()
        break

REF_METHOD_OP_EXCLUSIONS = {
    "bfloat16",
    "bool",
    "contiguous",
    "cpu",
    "cuda",
    "detach",
    "dim",
    "double",
    "float",
    "half",
    "int",
    "item",
    "long",
    "numel",
    "options",
    "permute",
    "register_buffer",
    "reshape",
    "select",
    "shape",
    "size",
    "sizes",
    "squeeze",
    "to",
    "transpose",
    "unbind",
    "unsqueeze",
    "view",
}

PROBLEM_OPERATOR_EXCLUSIONS = {
    # DropPath needs random-mask construction in the wrapper; this is not the
    # target operator implementation itself.
    "099_DropPath": {"rand"},
}


def _dotted_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return None


def _camel_to_snake(name: str) -> str:
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    s2 = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1)
    return s2.replace("__", "_").lower()


def _expand_operator_aliases(name: str) -> set[str]:
    short = name.split(".")[-1]
    aliases = {short}

    if name.startswith(("torch.linalg.", "at.linalg.")):
        aliases.add(f"linalg_{short}")
    elif name.startswith(("torch.special.", "at.special.")):
        aliases.add(f"special_{short}")

    snake = _camel_to_snake(short)
    aliases.add(snake)

    if snake.endswith(("1d", "2d", "3d")):
        base = re.sub(r"(1d|2d|3d)$", "", snake)
        if base.endswith("_"):
            base = base[:-1]
        if base:
            aliases.add(base)

    if snake.endswith("_loss"):
        aliases.add(snake[:-5])

    return {alias for alias in aliases if alias}


def _collect_model_operator_names(ref_path: Path) -> set[str]:
    tree = ast.parse(ref_path.read_text(encoding="utf-8"), filename=str(ref_path))
    model = next(
        (node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Model"),
        None,
    )
    if model is None:
        return set()

    model_methods = {
        node.name: node
        for node in model.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    module_helpers = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    self_nn_ctor_map: dict[str, str] = {}
    for fn in model_methods.values():
        for node in ast.walk(fn):
            if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
                continue
            ctor_name = _dotted_name(node.value.func)
            if not ctor_name or not ctor_name.startswith("nn."):
                continue
            for target in node.targets:
                if (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"
                ):
                    self_nn_ctor_map[target.attr] = ctor_name

    operator_names: set[str] = set()
    visited_functions: set[str] = set()

    def record_from_call(call_name: str):
        if call_name.startswith("torch.nn.functional."):
            operator_names.update(_expand_operator_aliases(call_name))
            return
        if call_name.startswith(("torch.", "at.", "F.")):
            operator_names.update(_expand_operator_aliases(call_name))
            return
        if call_name.startswith("nn."):
            operator_names.update(_expand_operator_aliases(call_name))
            return
        if call_name.startswith("self."):
            attr_name = call_name.split(".", 1)[1]
            ctor_name = self_nn_ctor_map.get(attr_name)
            if ctor_name:
                operator_names.update(_expand_operator_aliases(ctor_name))
            return

        if "." in call_name:
            method_name = call_name.rsplit(".", 1)[-1]
            if method_name not in REF_METHOD_OP_EXCLUSIONS:
                operator_names.update(_expand_operator_aliases(method_name))

    def scan_function(fn: ast.FunctionDef | ast.AsyncFunctionDef):
        if fn.name in visited_functions:
            return
        visited_functions.add(fn.name)

        for node in ast.walk(fn):
            if not isinstance(node, ast.Call):
                continue

            call_name = _dotted_name(node.func)
            if not call_name:
                continue

            if call_name in model_methods:
                scan_function(model_methods[call_name])
                continue

            if call_name in module_helpers:
                scan_function(module_helpers[call_name])
                continue

            record_from_call(call_name)

    for fn in model_methods.values():
        scan_function(fn)

    return {name for name in operator_names if name not in REF_CONSTRUCTOR_EXCLUSIONS}


def _build_problem_rule(problem: dict, ref_dir: Path) -> dict | None:
    ref_path = ref_dir / problem["torch_model"]
    if not ref_path.exists():
        return None

    operator_names = sorted(
        _collect_model_operator_names(ref_path),
        key=lambda name: (-len(name), name),
    )
    operator_names = [
        name
        for name in operator_names
        if name not in PROBLEM_OPERATOR_EXCLUSIONS.get(problem["name"], set())
    ]
    if not operator_names:
        return None

    joined = "|".join(re.escape(name) for name in operator_names)
    pattern = (
        rf"(?:\b(?:torch::|at::)(?:[A-Za-z_]\w*::)*(?:{joined})\s*\()"
        rf"|(?:(?:\.|->)\s*(?:{joined})\s*\()"
    )

    preview = ", ".join(operator_names[:12])
    if len(operator_names) > 12:
        preview += ", ..."

    return {
        "id": f"FORBIDDEN_PROBLEM_SPECIFIC_OP_{problem['id']}",
        "pattern": pattern,
        "flags": 0,
        "message": (
            f"题目 {problem['name']} 对应算子禁止直接调库实现；"
            f"命中词汇示例: {preview}"
        ),
        "operators": operator_names,
        "torch_model": problem["torch_model"],
    }


def generate_rules(config_path: Path, ref_dir: Path) -> dict:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    problems = cfg.get("problems", [])

    problem_rules = {}
    for problem in problems:
        rule = _build_problem_rule(problem, ref_dir)
        if rule is not None:
            problem_rules[problem["name"]] = rule

    return {
        "version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "general_rules": GENERAL_RULES,
        "problem_rules": problem_rules,
    }


def main():
    root = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(description="Generate static BangC audit rules.")
    parser.add_argument(
        "--config",
        type=Path,
        default=root / "config.yaml",
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--ref-dir",
        type=Path,
        default=root / "ref",
        help="Path to the reference problem directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "security" / "bangc_audit_rules.json",
        help="Path to the generated audit rules file",
    )
    args = parser.parse_args()

    payload = generate_rules(args.config, args.ref_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(
        f"Generated {args.output} with "
        f"{len(payload['general_rules'])} general rules and "
        f"{len(payload['problem_rules'])} problem-specific rules."
    )


if __name__ == "__main__":
    main()
