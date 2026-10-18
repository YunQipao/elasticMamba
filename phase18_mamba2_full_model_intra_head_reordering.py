import argparse
import copy
import csv
import random
import statistics

import torch
from transformers import AutoTokenizer

from output_naming import build_output_path
from phase6_eval_prefix_quality import sample_validation_texts
from phase12_mamba2_head_permutation_equivalence import (
    MODEL_NAME,
    PROMPT,
    TOKENIZER_NAME,
    build_model,
    collect_mamba2_mixers,
    compute_logits,
    configure_mamba2_for_equivalence,
    ensure_local_hf_cache,
    print_metrics,
    summarize_logits_diff,
)
from phase17_mamba2_intra_head_channel_permutation_probe import (
    bad_permute_mamba2_intra_head_inproj_only_,
    make_intra_head_channel_perm,
    permute_mamba2_intra_head_channels_,
)


torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
try:
    torch.set_float32_matmul_precision("highest")
except Exception:
    pass


STRICT_MAX = 1e-3
STRICT_MEAN = 1e-4
STRICT_REL = 1e-5
FUNCTIONAL_REL = 2e-4
EXPECTED_BAD_TOP1_MAX = 0.99


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=str, default="1")
    parser.add_argument("--heads", type=str, default="all")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--negative_control", action="store_true")
    parser.add_argument("--multi_prompt", action="store_true")
    parser.add_argument("--num_prompts", type=int, default=16)
    parser.add_argument("--seq_lens", nargs="+", type=int, default=[32, 128, 512, 1024])
    parser.add_argument("--disable_mem_eff_path", action="store_true")
    parser.add_argument("--output_csv", type=str, default=None)
    return parser.parse_args()


@torch.no_grad()
def parse_layer_spec(layers_arg: str, total_layers: int):
    if layers_arg == "all":
        return list(range(total_layers))
    if "," in layers_arg:
        layer_indices = sorted({int(token.strip()) for token in layers_arg.split(",") if token.strip()})
    else:
        num_layers = int(layers_arg)
        if num_layers < 1 or num_layers > total_layers:
            raise ValueError(f"--layers must be in [1, {total_layers}], 'all', or a comma list, got {layers_arg}")
        layer_indices = list(range(num_layers))
    invalid = [layer_idx for layer_idx in layer_indices if layer_idx < 0 or layer_idx >= total_layers]
    if invalid:
        raise ValueError(f"Invalid layer indices: {invalid}")
    return layer_indices


@torch.no_grad()
def parse_head_spec(heads_arg: str, nheads: int):
    if heads_arg == "all":
        return list(range(nheads))
    head_indices = sorted({int(token.strip()) for token in heads_arg.split(",") if token.strip()})
    invalid = [head_id for head_id in head_indices if head_id < 0 or head_id >= nheads]
    if invalid:
        raise ValueError(f"Invalid head indices: {invalid}")
    return head_indices


@torch.no_grad()
def build_perms_by_layer_head(mixers, target_layers, target_heads, seed: int):
    perms = {}
    mixer_lookup = {layer_idx: mixer for layer_idx, mixer in mixers}
    for layer_idx in target_layers:
        layer_perms = {}
        mixer = mixer_lookup[layer_idx]
        for head_id in target_heads:
            layer_perms[head_id] = make_intra_head_channel_perm(mixer, head_id=head_id, seed=seed + layer_idx * 10007)
        perms[layer_idx] = layer_perms
    return perms


@torch.no_grad()
def apply_good_reordering(model, target_layers, target_heads, perms_by_layer_head):
    for layer_idx, mixer in collect_mamba2_mixers(model):
        if layer_idx not in target_layers:
            continue
        for head_id in target_heads:
            permute_mamba2_intra_head_channels_(mixer, head_id, perms_by_layer_head[layer_idx][head_id])


@torch.no_grad()
def apply_bad_reordering(model, target_layers, target_heads, perms_by_layer_head):
    for layer_idx, mixer in collect_mamba2_mixers(model):
        if layer_idx not in target_layers:
            continue
        for head_id in target_heads:
            bad_permute_mamba2_intra_head_inproj_only_(mixer, head_id, perms_by_layer_head[layer_idx][head_id])


@torch.no_grad()
def classify_good(metrics: dict):
    if (
        metrics["max_abs_diff"] < STRICT_MAX
        and metrics["mean_abs_diff"] < STRICT_MEAN
        and metrics["relative_max_error"] < STRICT_REL
    ):
        return "strict_pass"
    if metrics["top1_match_rate"] >= 1.0 - 1e-8 and metrics["relative_max_error"] < FUNCTIONAL_REL:
        return "functional_pass"
    return "fail"


@torch.no_grad()
def classify_bad(good_metrics: dict, bad_metrics: dict):
    bad_is_much_worse = bad_metrics["max_abs_diff"] > max(good_metrics["max_abs_diff"] * 100.0, 1e-2)
    bad_top1_drops = bad_metrics["top1_match_rate"] < min(good_metrics["top1_match_rate"] - 1e-3, EXPECTED_BAD_TOP1_MAX)
    if bad_is_much_worse and bad_top1_drops:
        return "expected_fail"
    return "weak_fail"


@torch.no_grad()
def synchronize_if_cuda(device: str):
    if device == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def compute_metrics_for_input(reference_model, reordered_model, input_ids, device: str):
    batch = input_ids.unsqueeze(0).to(device)
    synchronize_if_cuda(device)
    original_logits = compute_logits(reference_model, batch)
    synchronize_if_cuda(device)
    reordered_logits = compute_logits(reordered_model, batch)
    synchronize_if_cuda(device)
    return summarize_logits_diff(original_logits, reordered_logits)


@torch.no_grad()
def summarize_cases(rows):
    max_abs_values = [float(row["max_abs_diff"]) for row in rows]
    rel_values = [float(row["relative_max_error"]) for row in rows]
    top1_values = [float(row["top1_match_rate"]) for row in rows]
    pass_count = sum(row["status"] in {"strict_pass", "functional_pass", "expected_fail"} for row in rows)
    total_count = len(rows)
    return {
        "pass_count": pass_count,
        "total_count": total_count,
        "mean_max_abs_diff": statistics.mean(max_abs_values) if max_abs_values else 0.0,
        "max_max_abs_diff": max(max_abs_values) if max_abs_values else 0.0,
        "mean_relative_max_error": statistics.mean(rel_values) if rel_values else 0.0,
        "max_relative_max_error": max(rel_values) if rel_values else 0.0,
        "mean_top1_match_rate": statistics.mean(top1_values) if top1_values else 0.0,
        "min_top1_match_rate": min(top1_values) if top1_values else 0.0,
    }


@torch.no_grad()
def print_summary(mode: str, target_layers, target_heads, metrics: dict, status: str, disable_mem_eff_path: bool):
    print(f"reordering mode      = {mode}")
    print(f"selected layers      = {target_layers}")
    print(f"selected heads       = {target_heads}")
    print(f"disable_mem_eff_path = {disable_mem_eff_path}")
    print(f"max_abs_diff         = {metrics['max_abs_diff']:.8e}")
    print(f"mean_abs_diff        = {metrics['mean_abs_diff']:.8e}")
    print(f"relative_max_error   = {metrics['relative_max_error']:.8e}")
    print(f"output_abs_max       = {metrics['logits_abs_max']:.8e}")
    print(f"top1_match_rate      = {metrics['top1_match_rate']:.8f}")
    print(f"final status         = {status}")


@torch.no_grad()
def save_results_csv(rows, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "tag",
                "model_name",
                "tokenizer_name",
                "layers_arg",
                "heads_arg",
                "target_layers",
                "target_heads",
                "reordering_mode",
                "negative_control",
                "prompt_id",
                "seq_len",
                "disable_mem_eff_path",
                "max_abs_diff",
                "mean_abs_diff",
                "relative_max_error",
                "top1_match_rate",
                "output_abs_max",
                "strict_threshold_max",
                "strict_threshold_mean",
                "strict_threshold_relative",
                "functional_threshold_relative",
                "status",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved results to {output_path}")


@torch.no_grad()
def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    output_path = build_output_path(
        args.output_csv,
        "phase18_mamba2_full_model_intra_head_reordering_multiprompt"
        if args.multi_prompt
        else "phase18_mamba2_full_model_intra_head_reordering",
        tags={
            "layers": args.layers,
            "heads": args.heads,
            "neg": args.negative_control,
            "multi_prompt": args.multi_prompt,
            "num_prompts": args.num_prompts if args.multi_prompt else None,
            "seq_lens": "-".join(map(str, args.seq_lens)) if args.multi_prompt else None,
            "disable_mem_eff": args.disable_mem_eff_path,
            "seed": args.seed,
        },
    )
    ensure_local_hf_cache(output_path.parent)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    print(f"Loading tokenizer: {TOKENIZER_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    print(f"Loading model: {MODEL_NAME}")
    model, loader_name = build_model(device=device, dtype=dtype)
    model = configure_mamba2_for_equivalence(model, disable_mem_eff_path=args.disable_mem_eff_path)
    print(f"Model loader: {loader_name}")

    mixers = collect_mamba2_mixers(model)
    total_layers = len(mixers)
    if total_layers == 0:
        raise RuntimeError("No Mamba2 mixer layers found")
    print(f"Found {total_layers} Mamba2 mixer layers")

    target_layers = parse_layer_spec(args.layers, total_layers)
    nheads = int(getattr(mixers[0][1], "nheads"))
    target_heads = parse_head_spec(args.heads, nheads)
    print(f"Target layers: {target_layers}")
    print(f"Target heads: {target_heads}")

    perms_by_layer_head = build_perms_by_layer_head(mixers, target_layers, target_heads, args.seed)

    if not args.multi_prompt:
        encoded = tokenizer(PROMPT, return_tensors="pt")
        single_input_ids = encoded["input_ids"][0]
        eval_inputs = [(0, int(single_input_ids.numel()), single_input_ids)]
    else:
        print("Loading and tokenizing WikiText-2 validation samples")
        eval_inputs = []
        max_seq_len = max(args.seq_lens)
        samples = sample_validation_texts(tokenizer, num_samples=args.num_prompts, seq_len=max_seq_len)
        for prompt_id, sample_input_ids in enumerate(samples):
            for seq_len in args.seq_lens:
                truncated = sample_input_ids[:seq_len]
                if truncated.numel() < 2:
                    continue
                eval_inputs.append((prompt_id, seq_len, truncated))

    rows = []
    if not args.negative_control:
        reordered_model = copy.deepcopy(model).eval()
        reordered_model = configure_mamba2_for_equivalence(reordered_model, disable_mem_eff_path=args.disable_mem_eff_path)
        apply_good_reordering(reordered_model, target_layers, target_heads, perms_by_layer_head)

        for prompt_id, seq_len, input_ids in eval_inputs:
            metrics = compute_metrics_for_input(model, reordered_model, input_ids, device)
            status = classify_good(metrics)
            print(
                f"mode=GOOD prompt_id={prompt_id} seq_len={seq_len} "
                f"max_abs_diff={metrics['max_abs_diff']:.8e} "
                f"mean_abs_diff={metrics['mean_abs_diff']:.8e} "
                f"relative_max_error={metrics['relative_max_error']:.8e} "
                f"top1_match_rate={metrics['top1_match_rate']:.8f} status={status}"
            )
            rows.append(
                {
                    "tag": "good",
                    "model_name": MODEL_NAME,
                    "tokenizer_name": TOKENIZER_NAME,
                    "layers_arg": args.layers,
                    "heads_arg": args.heads,
                    "target_layers": str(target_layers),
                    "target_heads": str(target_heads),
                    "reordering_mode": "GOOD",
                    "negative_control": False,
                    "prompt_id": prompt_id,
                    "seq_len": seq_len,
                    "disable_mem_eff_path": args.disable_mem_eff_path,
                    "max_abs_diff": f"{metrics['max_abs_diff']:.8e}",
                    "mean_abs_diff": f"{metrics['mean_abs_diff']:.8e}",
                    "relative_max_error": f"{metrics['relative_max_error']:.8e}",
                    "top1_match_rate": f"{metrics['top1_match_rate']:.8f}",
                    "output_abs_max": f"{metrics['logits_abs_max']:.8e}",
                    "strict_threshold_max": f"{STRICT_MAX:.8e}",
                    "strict_threshold_mean": f"{STRICT_MEAN:.8e}",
                    "strict_threshold_relative": f"{STRICT_REL:.8e}",
                    "functional_threshold_relative": f"{FUNCTIONAL_REL:.8e}",
                    "status": status,
                }
            )
    else:
        good_baseline_model = copy.deepcopy(model).eval()
        good_baseline_model = configure_mamba2_for_equivalence(
            good_baseline_model,
            disable_mem_eff_path=args.disable_mem_eff_path,
        )
        apply_good_reordering(good_baseline_model, target_layers, target_heads, perms_by_layer_head)

        bad_model = copy.deepcopy(model).eval()
        bad_model = configure_mamba2_for_equivalence(bad_model, disable_mem_eff_path=args.disable_mem_eff_path)
        apply_bad_reordering(bad_model, target_layers, target_heads, perms_by_layer_head)

        for prompt_id, seq_len, input_ids in eval_inputs:
            good_metrics = compute_metrics_for_input(model, good_baseline_model, input_ids, device)
            bad_metrics = compute_metrics_for_input(model, bad_model, input_ids, device)
            status = classify_bad(good_metrics, bad_metrics)
            print(
                f"mode=BAD prompt_id={prompt_id} seq_len={seq_len} "
                f"max_abs_diff={bad_metrics['max_abs_diff']:.8e} "
                f"mean_abs_diff={bad_metrics['mean_abs_diff']:.8e} "
                f"relative_max_error={bad_metrics['relative_max_error']:.8e} "
                f"top1_match_rate={bad_metrics['top1_match_rate']:.8f} status={status}"
            )
            rows.append(
                {
                    "tag": "bad",
                    "model_name": MODEL_NAME,
                    "tokenizer_name": TOKENIZER_NAME,
                    "layers_arg": args.layers,
                    "heads_arg": args.heads,
                    "target_layers": str(target_layers),
                    "target_heads": str(target_heads),
                    "reordering_mode": "BAD",
                    "negative_control": True,
                    "prompt_id": prompt_id,
                    "seq_len": seq_len,
                    "disable_mem_eff_path": args.disable_mem_eff_path,
                    "max_abs_diff": f"{bad_metrics['max_abs_diff']:.8e}",
                    "mean_abs_diff": f"{bad_metrics['mean_abs_diff']:.8e}",
                    "relative_max_error": f"{bad_metrics['relative_max_error']:.8e}",
                    "top1_match_rate": f"{bad_metrics['top1_match_rate']:.8f}",
                    "output_abs_max": f"{bad_metrics['logits_abs_max']:.8e}",
                    "strict_threshold_max": "",
                    "strict_threshold_mean": "",
                    "strict_threshold_relative": "",
                    "functional_threshold_relative": "",
                    "status": status,
                }
            )

    save_results_csv(rows, output_path)

    summary = summarize_cases(rows)
    print("\nSummary")
    print(f"pass_count / total_count = {summary['pass_count']} / {summary['total_count']}")
    print(
        f"mean/max max_abs_diff = {summary['mean_max_abs_diff']:.8e} / {summary['max_max_abs_diff']:.8e}"
    )
    print(
        f"mean/max relative_max_error = {summary['mean_relative_max_error']:.8e} / {summary['max_relative_max_error']:.8e}"
    )
    print(
        f"mean/min top1_match_rate = {summary['mean_top1_match_rate']:.8f} / {summary['min_top1_match_rate']:.8f}"
    )

    if not args.negative_control:
        if all(row["status"] == "strict_pass" for row in rows):
            print("[PASS] Full-model intra-head reordering meets strict equivalence.")
        elif all(row["status"] in {"strict_pass", "functional_pass"} for row in rows):
            print("[PARTIAL] Full-model intra-head reordering meets functional equivalence only.")
        else:
            print("[FAIL] Full-model intra-head reordering did not meet the requested criteria.")
    elif all(row["status"] == "expected_fail" for row in rows):
        print("[PASS] Negative control clearly breaks full-model output as expected.")
    else:
        print("[FAIL] Negative control did not fail clearly enough.")


if __name__ == "__main__":
    main()
