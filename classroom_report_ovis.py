# -*- coding: utf-8 -*-
"""
下课报告 Ovis2 增强子进程（由 challenge-cup-ovis 环境运行）。

教师端点击「下课 · 生成报告」后，classroom_dashboard.py（:8008）会以子进程方式
调用本脚本：输入「规则统计报告」JSON，输出 Ovis2-GPTQ 润色后的报告 JSON。

隐私说明：与挑战杯旧版一致，本脚本只接收**已脱敏的统计数字**，图像一律使用
16x16 空白白图满足多模态输入接口，绝不发送课堂视频或人脸画面。

用法::

    python classroom_report_ovis.py --report-json report.json --out-json out.json

环境变量:
    CLASSROOM_OVIS_MODEL_DIR    Ovis2 GPTQ 模型目录（必填，默认由父进程注入）
    CLASSROOM_OVIS_DEVICE       cuda:0 / cpu（默认 cuda:0，无 CUDA 自动报错降级）
    CLASSROOM_OVIS_BACKEND      cuda / torch（默认随 device）
    CLASSROOM_OVIS_ATTENTION    sdpa（默认；Windows 无 flash-attn）
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _decode_json(tokenizer, output_ids) -> tuple[dict, str]:
    """兼容 Ovis 自定义 generate 返回“仅新词元”或“提示词+新词元”两种形式。"""
    raw = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    decoder = json.JSONDecoder()
    candidates: list[tuple[int, dict]] = []
    for index, char in enumerate(raw):
        if char != "{":
            continue
        try:
            value, consumed = decoder.raw_decode(raw[index:])
            if isinstance(value, dict):
                candidates.append((consumed, value))
        except json.JSONDecodeError:
            continue
    # 一个完整报告内部还包含多个子对象。旧逻辑取“最后一个对象”，实际会
    # 错拿最内层学生评语；应选择覆盖字符最多的最外层 JSON。
    return (max(candidates, key=lambda item: item[0])[1] if candidates else {}), raw


def _generate_plain_text(model, tokenizer, prompt: str, max_new_tokens: int = 72) -> str:
    """让 Ovis 生成短文本；报告结构由程序负责，不强迫2B模型输出复杂JSON。"""
    from PIL import Image
    import torch

    query = f"<image>\n{prompt}"
    _, input_ids, pixel_values = model.preprocess_inputs(
        query, [Image.new("RGB", (16, 16), "white")], max_partition=1
    )
    attention_mask = torch.ne(input_ids, tokenizer.pad_token_id)
    input_ids = input_ids.unsqueeze(0).to(model.device)
    attention_mask = attention_mask.unsqueeze(0).to(model.device)
    if pixel_values is not None:
        visual_tokenizer = model.get_visual_tokenizer()
        pixel_values = pixel_values.to(
            device=visual_tokenizer.device, dtype=visual_tokenizer.dtype
        )
    input_length = int(input_ids.shape[-1])
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            pixel_values=[pixel_values],
            attention_mask=attention_mask,
            max_new_tokens=int(os.getenv("CLASSROOM_OVIS_MAX_NEW_TOKENS", str(max_new_tokens))),
            do_sample=False,
            # KV cache可避免每生成一个字都重新计算整段课堂统计提示词；
            # 对多学生报告的生成时间影响很大。
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=model.model.generation_config.eos_token_id,
        )[0]
    if (
        int(output_ids.shape[-1]) >= input_length
        and torch.equal(output_ids[:input_length], input_ids[0])
    ):
        output_ids = output_ids[input_length:]
    return tokenizer.decode(output_ids, skip_special_tokens=True).strip()


def _comment_lines(raw: str) -> list[str]:
    lines = [
        item.strip(" -0123456789.、：:")
        for item in raw.replace("｜", "\n").replace("|", "\n").splitlines()
        if item.strip(" -0123456789.、：:")
    ]
    return lines or ([raw.strip()] if raw.strip() else [])


def _student_sections(raw: str) -> list[str]:
    """提取个人评价、建议和目标，兼容Ovis输出项目符号与独立标题行。"""
    suggestion_mark = "具体改进建议"
    goal_mark = "下节课目标"
    if suggestion_mark in raw and goal_mark in raw:
        assessment, remainder = raw.split(suggestion_mark, 1)
        suggestion, goal = remainder.split(goal_mark, 1)
        clean = lambda value: " ".join(_comment_lines(value)).strip()
        values = [clean(assessment), clean(suggestion), clean(goal)]
        if all(values):
            return values
    lines = _comment_lines(raw)
    return (lines + [lines[-1]] * 3)[:3] if lines else []


def generate_bundle_comment(model, tokenizer, class_report: dict, students: dict) -> tuple[dict, str]:
    """模型加载一次，分别生成全班和个人短评，再可靠地组装固定报告结构。"""
    compact_class = {
        key: class_report.get(key)
        for key in ("duration_seconds", "student_count", "focus_rate", "state_percentages", "stage_trend")
    }
    class_raw = _generate_plain_text(
        model,
        tokenizer,
        "你是课堂学习状态分析助手。根据下列统计写三行简洁中文：第一行总体评价，"
        "第二行指出主要状态变化，第三行给教师可执行建议。不要标题、编号、JSON，"
        "不得作心理或医学诊断。数据："
        + json.dumps(compact_class, ensure_ascii=False),
        max_new_tokens=80,
    )
    class_lines = _comment_lines(class_raw)
    if not class_lines:
        raise RuntimeError("Ovis 未生成全班评语文本")
    class_values = (class_lines + [class_lines[-1]] * 3)[:3]
    generated_students: dict[str, dict] = {}
    raw_parts = [f"CLASS: {class_raw}"]
    for device_id, report in students.items():
        compact = {
            key: report.get(key)
            for key in ("student_name", "focus_rate", "focus_level", "state_percentages", "stage_trend")
        }
        student_raw = _generate_plain_text(
            model,
            tokenizer,
            "你是课堂学习状态分析助手。根据下列学生统计写三行简洁中文：第一行客观评价，"
            "第二行给具体改进建议，第三行给下节课目标。不要标题、编号、JSON，"
            "使用谨慎表达。数据：" + json.dumps(compact, ensure_ascii=False),
            max_new_tokens=64,
        )
        lines = _student_sections(student_raw)
        if not lines:
            raise RuntimeError(f"Ovis 未生成学生 {device_id} 的评语文本")
        values = lines
        generated_students[device_id] = {"a": values[0], "t": values[1], "g": values[2]}
        raw_parts.append(f"{device_id}: {student_raw}")
    return {
        "c": {
            "a": class_values[0], "p": class_values[1], "f": class_values[1],
            "g": [class_values[0]], "i": [class_values[1]], "t": class_values[2],
        },
        "s": generated_students,
    }, "\n".join(raw_parts)


def main() -> int:
    parser = argparse.ArgumentParser(description="Ovis2 课堂报告增强")
    parser.add_argument("--report-json", required=True, help="规则统计报告 JSON 路径")
    parser.add_argument("--out-json", required=True, help="增强后报告 JSON 输出路径")
    args = parser.parse_args()

    with open(args.report_json, "r", encoding="utf-8") as f:
        report = json.load(f)

    # GPTQModel 的 Torch 量化层默认调用 torch.compile/Inductor；Windows 原生
    # 环境通常没有可用 Triton。禁用 Dynamo 后使用 eager/CUDA 扩展执行。
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

    model_dir = Path(os.environ["CLASSROOM_OVIS_MODEL_DIR"])
    if not (model_dir / "config.json").exists():
        raise FileNotFoundError(f"Ovis model is missing: {model_dir}")

    import torch
    from gptqmodel import GPTQModel
    from transformers import GenerationConfig

    device = os.getenv("CLASSROOM_OVIS_DEVICE", "cuda:0").lower()
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CLASSROOM_OVIS_DEVICE=cuda 但当前环境未检测到 CUDA")
    backend = os.getenv(
        "CLASSROOM_OVIS_BACKEND", "cuda" if device.startswith("cuda") else "torch"
    )
    attention = os.getenv("CLASSROOM_OVIS_ATTENTION", "sdpa")
    model = GPTQModel.load(
        str(model_dir),
        device=device,
        backend=backend,
        trust_remote_code=True,
        llm_attn_implementation=attention,
    )
    model.model.generation_config = GenerationConfig.from_pretrained(str(model_dir))
    tokenizer = model.get_text_tokenizer()

    if report.get("report_kind") == "classroom_bundle":
        class_report = dict(report.get("class_report") or {})
        students = dict(report.get("student_reports") or {})
        generated, raw = generate_bundle_comment(model, tokenizer, class_report, students)
        short_class = dict(generated.get("c") or {})
        short_students = dict(generated.get("s") or {})
        class_extra = {
            "overall_assessment": short_class.get("a", ""),
            "attention_pattern": short_class.get("p", ""),
            "fatigue_pattern": short_class.get("f", ""),
            "learning_strengths": short_class.get("g") or [],
            "priority_improvements": short_class.get("i") or [],
            "teacher_suggestion": short_class.get("t", ""),
        }
        if not any(class_extra.values()):
            excerpt = raw.strip().replace("\n", " ")[-500:]
            raise RuntimeError(f"Ovis 未返回有效评语；原始输出：{excerpt or '空'}")
        class_report.update({
            "engine": "classroom-statistics-ovis",
            "ovis_status": "completed",
            "model": "Ovis2-2B-GPTQ-Int4",
            "ovis_report": class_extra,
            "raw_output": raw,
        })
        for device_id, student_report in students.items():
            personal = dict(student_report or {})
            short_extra = dict(short_students.get(device_id) or {})
            extra = {
                "overall_assessment": short_extra.get("a", ""),
                "student_suggestion": short_extra.get("t", ""),
                "next_class_goal": short_extra.get("g", ""),
            }
            extra = {key: value for key, value in extra.items() if value}
            personal.update({
                "engine": "classroom-statistics-ovis",
                "ovis_status": "completed" if extra else "failed",
                "model": "Ovis2-2B-GPTQ-Int4",
                "ovis_comment": extra,
            })
            students[device_id] = personal
        result = {
            "report_kind": "classroom_bundle",
            "class_report": class_report,
            "student_reports": students,
        }
    else:
        extra, raw = generate_comment(model, tokenizer, report, "class")
        result = {
            **report,
            "engine": "classroom-statistics-ovis",
            "model": "Ovis2-2B-GPTQ-Int4",
            "ovis_report": extra,
            "raw_output": raw,
        }
    Path(args.out_json).write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"ok": True, "report_kind": result.get("report_kind", "single")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001 —— 子进程任何失败都由父进程降级处理
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
        sys.exit(1)
