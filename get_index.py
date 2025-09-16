#!/usr/bin/env python3
# resample_context_target.py
# 用法: python resample_context_target.py input.json output.json
# 如果不提供 output.json，默认在 input 文件名后加 "_filled.json"

import json
import argparse
from pathlib import Path
import numpy as np

def resample_list(values, target_len):
    """
    将 values 重采样为长度为 target_len 的列表（线性内插）。
    - 若 len(values) >= 2: 使用线性内插（包含端点）。
    - 若 len(values) == 1: 直接复制该值 target_len 次。
    - 若 values 为空或 None: 返回空列表。
    返回整数列表（四舍五入）。
    """
    if values is None:
        return []
    vals = list(values)
    n = len(vals)
    if n == 0:
        return []
    if n == 1:
        return [int(round(vals[0]))] * target_len
    # n >= 2
    # 原始位置从 0 到 1，目标位置也从 0 到 1
    orig_pos = np.linspace(0.0, 1.0, num=n)
    target_pos = np.linspace(0.0, 1.0, num=target_len)
    interpolated = np.interp(target_pos, orig_pos, vals)
    # 四舍五入并转为 int
    return [int(round(float(x))) for x in interpolated]

def process_dict(d, ctx_len=6, tgt_len=8, verbose=False):
    """
    在字典 d 的每个条目上处理 'context' 和 'target' 字段（如果存在）。
    返回新的字典（不修改原始 d）。
    """
    out = {}
    for k, v in d.items():
        item = dict(v) if isinstance(v, dict) else {}
        # 处理 context
        if 'context' in item:
            new_ctx = resample_list(item.get('context'), ctx_len)
            # 如果插值结果仍然小于所需长度（例如空输入），则用最后一个元素补齐（如果存在）
            if len(new_ctx) < ctx_len:
                if len(new_ctx) == 0:
                    # 无数据，填 0 或保持空，下面选择保留空列表
                    pass
                else:
                    last = new_ctx[-1]
                    new_ctx += [last] * (ctx_len - len(new_ctx))
            item['context'] = new_ctx
        # 处理 target
        if 'target' in item:
            new_tgt = resample_list(item.get('target'), tgt_len)
            if len(new_tgt) < tgt_len:
                if len(new_tgt) == 0:
                    pass
                else:
                    last = new_tgt[-1]
                    new_tgt += [last] * (tgt_len - len(new_tgt))
            item['target'] = new_tgt
        out[k] = item
        if verbose and ('context' in v or 'target' in v):
            print(f"Processed {k}: context->{item.get('context')} target->{item.get('target')}")
    return out

def main():
    parser = argparse.ArgumentParser(description="Resample 'context' to length 6 and 'target' to length 8 in a JSON file.")
    parser.add_argument('input', help="输入 JSON 文件路径")
    parser.add_argument('output', nargs='?', default=None, help="输出 JSON 文件路径（可选）")
    parser.add_argument('--verbose', action='store_true', help="打印处理信息")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"输入文件不存在: {input_path}")
        return
    out_path = Path(args.output) if args.output else input_path.with_name(input_path.stem + "_filled" + input_path.suffix)

    with input_path.open('r', encoding='utf-8') as f:
        data = json.load(f)

    processed = process_dict(data, ctx_len=6, tgt_len=8, verbose=args.verbose)

    # 修改点1: 使用紧凑格式输出JSON（所有数据在一行）
    with out_path.open('w', encoding='utf-8') as f:
        # 使用 separators 参数移除不必要的空格
        # 使用 ensure_ascii=False 允许非ASCII字符
        # 使用 indent=None 避免格式化换行
        json.dump(processed, f, ensure_ascii=False, indent=None, separators=(',', ':'))
    
    print(f"已写入: {out_path}")

if __name__ == "__main__":
    main()