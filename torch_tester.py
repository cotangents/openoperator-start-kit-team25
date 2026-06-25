#!/usr/bin/env python3
"""
torch_tester.py — 执行单个 PyTorch 参考模型的性能测量

用法:
  python3 torch_tester.py -t ref/001_LeakyReLU.py [-n 10] [--json]

输出:
  人类可读格式（默认）或 JSON 格式（--json）。
  JSON 输出写入 stdout，日志写入 stderr，方便管道解析。
"""
import sys
import json
import time
import traceback
import argparse
import importlib.util
from pathlib import Path

import torch
import torch_mlu
import numpy as np


def load_module(name: str, path: str):
    """动态加载 Python 模块。"""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def benchmark_hardware_time(fn, inputs_generator, repeats: int = 10) -> float:
    """
    使用墙钟时间计时，返回单次调用平均耗时（微秒）。
    每次迭代使用不同输入，避免缓存命中影响结果。
    """
    total_us = 0.0

    for _ in range(repeats):
        call_inputs = inputs_generator()
        torch.mlu.synchronize()
        t0 = time.perf_counter()
        fn(*call_inputs)
        torch.mlu.synchronize()
        t1 = time.perf_counter()
        total_us += (t1 - t0) * 1e6

    return total_us / repeats  # us


def run(torch_model_path: str, iterations: int = 10) -> dict:
    """
    加载并测试单个 PyTorch 参考模型。

    返回:
      {
        "name":          str,   # 模型名（文件 stem）
        "torch_us":      float, # 平均硬件耗时（微秒）
        "output_shape":  list,  # 输出张量 shape
        "output_dtype":  str,   # 输出张量 dtype
        "error":         str|None,
      }
    """
    name = Path(torch_model_path).stem

    # ── 加载模型模块 ──
    try:
        torch_mod = load_module("torch_model", torch_model_path)
    except Exception as e:
        print(f"[ERROR] 加载模型失败: {torch_model_path}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return {"name": name, "torch_us": 0.0, "error": f"加载失败: {e}"}

    # ── 实例化模型 ──
    try:
        init_inputs = torch_mod.get_init_inputs()
        model = torch_mod.Model(*init_inputs).mlu()
    except Exception as e:
        print(f"[ERROR] Model 实例化失败: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return {"name": name, "torch_us": 0.0, "error": f"实例化失败: {e}"}

    # ── 准备输入 ──
    try:
        raw_inputs = torch_mod.get_inputs()
        inputs = [t.mlu() if torch.is_tensor(t) else t for t in raw_inputs]
    except Exception as e:
        print(f"[ERROR] get_inputs() 失败: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return {"name": name, "torch_us": 0.0, "error": f"输入准备失败: {e}"}

    def make_inputs():
        raw = torch_mod.get_inputs()
        return [t.mlu() if torch.is_tensor(t) else t for t in raw]

    # ── 推理验证 ──
    try:
        with torch.no_grad():
            torch_out = model(*inputs)
            torch.mlu.synchronize()
    except Exception as e:
        print(f"[ERROR] 推理失败: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return {"name": name, "torch_us": 0.0, "error": f"推理失败: {e}"}

    # ── 性能测量 ──
    try:
        torch_time = benchmark_hardware_time(model, make_inputs, repeats=iterations)
    except Exception as e:
        print(f"[ERROR] 性能测量失败: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return {"name": name, "torch_us": 0.0, "error": f"性能测量失败: {e}"}

    # ── 收集输出信息 ──
    output_shape = list(torch_out.shape) if torch.is_tensor(torch_out) else None
    output_dtype = str(torch_out.dtype) if torch.is_tensor(torch_out) else None

    return {
        "name": name,
        "torch_us": round(torch_time, 3),
        "output_shape": output_shape,
        "output_dtype": output_dtype,
        "error": None,
    }


def main():
    parser = argparse.ArgumentParser(
        description="测试单个 PyTorch 参考模型的性能"
    )
    parser.add_argument("-t", "--torch-model", required=True,
                        help="PyTorch 模型文件路径")
    parser.add_argument("-n", "--iterations", type=int, default=10,
                        help="性能测量迭代次数（默认 10）")
    parser.add_argument("--json", action="store_true",
                        help="以 JSON 格式输出到 stdout")
    args = parser.parse_args()

    if not torch.mlu.is_available():
        print("MLU 不可用", file=sys.stderr)
        sys.exit(1)

    model_path = Path(args.torch_model)
    if not model_path.exists():
        print(f"文件不存在: {model_path}", file=sys.stderr)
        sys.exit(1)

    result = run(str(model_path), args.iterations)

    if args.json:
        # JSON 模式：仅 stdout 输出 JSON，供管道解析
        print(json.dumps(result, ensure_ascii=False))
    else:
        # 人类可读模式
        print(f"Model        : {result['name']}")
        if result.get("error"):
            print(f"Error        : {result['error']}")
            sys.exit(1)
        print(f"PyTorch time : {result['torch_us']:.3f} us")
        if result.get("output_shape"):
            print(f"Output shape : {result['output_shape']}")
        if result.get("output_dtype"):
            print(f"Output dtype : {result['output_dtype']}")

    if result.get("error"):
        sys.exit(1)


if __name__ == "__main__":
    main()
