"""
Tier 3 - frozen large-model intervention diagnostic.

Nothing is trained here. A pretrained open-weight causal LM is loaded through
Hugging Face and interrogated with the same question as Tier 1:

    Does structural overlap predict behavioral contribution at scale?

What an "edge" means here
-------------------------
Exactly what it means elsewhere: ``(layer, head, query position, key position)``
in the frozen model's own attention. Intervening on an edge sets that single
pre-softmax score to ``-inf`` for that head in that layer, so the surviving
entries of the row renormalise. No CRPA mask is imposed on the frozen model;
the partitioning is used only to define the neighbourhood P(i) over which
structural overlap is computed.

Why eager attention is forced
-----------------------------
FlashAttention and SDPA never materialise the probability matrix, so neither
extraction nor a per-head intervention is possible through them. This module
sets ``attn_implementation="eager"`` and refuses to proceed if that fails,
rather than quietly reporting numbers from a pass where no intervention
occurred. A 4D ``attention_mask`` cannot express a per-head edit either - it
broadcasts across heads - so the target layer's forward is patched directly.

Memory
------
Attention for a 7B model at 2k context is ~275 GB if every layer is retained.
Only the layers under study are instrumented, and their probabilities are
reduced to statistics immediately.

    # tiny smoke test - runs on CPU, no large download
    python -m experiments.large_model_diagnostic --smoke \\
        --model_id hf-internal-testing/tiny-random-LlamaForCausalLM

    # real diagnostic
    python -m experiments.large_model_diagnostic \\
        --model_id meta-llama/Meta-Llama-3-8B --dtype bfloat16 \\
        --context_length 1024 --layers 8 16 24
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from crpa.metrics import correlations, edge_structural_overlap, top_p_support_mask
from crpa.runmeta import RunRecord, Status, environment, save_record, write_csv, write_json
from crpa.seeding import set_seed
from experiments.common import print_header, resolve_device

EXPERIMENT = "large_model_diagnostic"


class InstrumentationError(RuntimeError):
    """Raised when attention cannot be observed or edited on this model."""


def load_model(
    model_id: str,
    dtype: str = "bfloat16",
    device_map: Optional[str] = None,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    trust_remote_code: bool = False,
):
    """Load a frozen causal LM with eager attention.

    Heavy dependencies stay optional: this module is only imported by Tier 3,
    so Tier 1 needs neither accelerate nor bitsandbytes.
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "Tier 3 requires transformers: pip install transformers"
        ) from exc

    torch_dtype = {
        "bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32,
    }[dtype]

    kwargs: Dict[str, object] = {
        "attn_implementation": "eager",   # required for extraction + intervention
        "trust_remote_code": trust_remote_code,
    }
    if load_in_8bit or load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as exc:
            raise ImportError(
                "quantised loading requires bitsandbytes and accelerate"
            ) from exc
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=load_in_8bit, load_in_4bit=load_in_4bit
        )
    else:
        # transformers renamed this argument: 4.x takes torch_dtype, 5.x takes
        # dtype and warns on the old name. Pick by signature rather than by
        # version string, so the probe works on whichever is installed.
        import inspect

        from transformers import AutoModelForCausalLM as _AMC

        try:
            params = inspect.signature(_AMC.from_pretrained).parameters
            key = "dtype" if "dtype" in params else "torch_dtype"
        except (TypeError, ValueError):
            key = "torch_dtype"
        kwargs[key] = torch_dtype
    if device_map:
        kwargs["device_map"] = device_map

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=trust_remote_code
    )
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.eval()
    return model, tokenizer


def decoder_layers(model) -> List[torch.nn.Module]:
    """Locate the decoder layer list across common architectures."""
    for path in ("model.layers", "transformer.h", "model.decoder.layers",
                 "gpt_neox.layers", "transformer.blocks"):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
        except AttributeError:
            continue
        if isinstance(obj, (torch.nn.ModuleList, list)):
            return list(obj)
    raise InstrumentationError(
        "could not locate decoder layers on {}; supported layouts are "
        "model.layers, transformer.h, model.decoder.layers, gpt_neox.layers, "
        "transformer.blocks".format(type(model).__name__)
    )


class AttentionProbe:
    """Capture attention probabilities and optionally edit one layer's scores.

    Implemented by patching the attention module's forward rather than by
    passing a 4D mask, because a mask broadcasts across heads and this study
    needs per-head edges.
    """

    def __init__(self, model, layer_idx: int) -> None:
        self.layer_idx = layer_idx
        layers = decoder_layers(model)
        if not 0 <= layer_idx < len(layers):
            raise IndexError(
                "layer {} out of range; model has {} layers".format(layer_idx, len(layers))
            )
        self.layer = layers[layer_idx]
        self.attn = self._find_attention(self.layer)
        self.captured: Optional[torch.Tensor] = None
        self.edges: List[Tuple[Optional[int], int, int]] = []
        self.n_intervened = 0
        self._orig_forward = None
        self._patched = None
        # True only while the instrumented layer's forward is executing, so
        # other layers' softmax calls are neither captured nor edited.
        self._active = False

    @staticmethod
    def _find_attention(layer: torch.nn.Module) -> torch.nn.Module:
        for name in ("self_attn", "attn", "attention", "self_attention"):
            if hasattr(layer, name):
                return getattr(layer, name)
        raise InstrumentationError(
            "no attention submodule found on {}".format(type(layer).__name__)
        )

    def __enter__(self) -> "AttentionProbe":
        probe = self
        original_softmax = F.softmax

        def patched_softmax(input, dim=-1, *args, **kwargs):
            """Capture, and optionally edit, the target layer's attention.

            Gated on ``probe._active`` so only the layer under study is
            observed. Without that gate a later layer's softmax overwrites the
            capture from the edited layer, and an intervention that did take
            effect looks inert.
            """
            is_attention = input.dim() == 4 and dim in (-1, 3)
            if not (probe._active and is_attention):
                return original_softmax(input, dim=dim, *args, **kwargs)

            scores = input
            if probe.edges:
                scores = input.clone()
                for head, q, k in probe.edges:
                    if not (0 <= q < scores.shape[-2] and 0 <= k < scores.shape[-1]):
                        continue
                    if head is None:
                        scores[:, :, q, k] = float("-inf")
                        probe.n_intervened += scores.shape[1]
                    else:
                        scores[:, head, q, k] = float("-inf")
                        probe.n_intervened += 1

            out = original_softmax(scores, dim=dim, *args, **kwargs)
            probe.captured = out.detach()
            return out

        self._orig_forward = self.attn.forward

        def scoped_forward(*args, **kwargs):
            probe._active = True
            try:
                return probe._orig_forward(*args, **kwargs)
            finally:
                probe._active = False

        self.attn.forward = scoped_forward
        F.softmax = patched_softmax
        self._patched = original_softmax
        return self

    def __exit__(self, *exc) -> None:
        if self._patched is not None:
            F.softmax = self._patched
        if self._orig_forward is not None:
            self.attn.forward = self._orig_forward
        self._active = False

    def reset(self) -> None:
        self.captured = None
        self.n_intervened = 0


@torch.no_grad()
def verify_instrumentation(model, probe: AttentionProbe, x: torch.Tensor) -> Dict[str, object]:
    """Prove that capture and intervention both actually work.

    Runs a baseline, checks the captured rows are probability distributions,
    then removes a real edge and checks that its mass is gone and the row still
    sums to 1. Raises rather than letting a silently-inert intervention be
    reported as a null result.
    """
    probe.reset()
    probe.edges = []
    model(x)
    if probe.captured is None:
        raise InstrumentationError(
            "no attention probabilities were captured. The model is probably "
            "not using eager attention; re-load with attn_implementation='eager'."
        )
    A = probe.captured
    row_sums = A.sum(dim=-1)
    if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-2):
        raise InstrumentationError(
            "captured tensor rows do not sum to 1 (min {:.4f}, max {:.4f}); "
            "the captured tensor is not an attention probability matrix".format(
                float(row_sums.min()), float(row_sums.max()))
        )

    T = A.shape[-1]
    # Not T - 1. A causal language-model loss compares logits[..., :-1, :] with
    # labels[..., 1:], so the final position's logits are discarded and an
    # intervention there cannot move the loss no matter how much attention mass
    # it removes. Verifying on that position would confirm the edit and prove
    # nothing about measurability.
    q = T - 2
    head = 0
    k = int(A[0, head, q, :q + 1].argmax().item())
    before = float(A[0, head, q, k].item())

    base_out = model(x, labels=x)
    baseline_loss = float(base_out.loss.item())
    baseline_logits = base_out.logits.detach().clone()

    probe.reset()
    probe.edges = [(head, q, k)]
    out = model(x, labels=x)
    intervened_loss = float(out.loss.item())
    logit_shift = float((out.logits - baseline_logits).abs().max().item())
    after = float(probe.captured[0, head, q, k].item())
    after_row_sum = float(probe.captured[0, head, q].sum().item())
    removed = probe.n_intervened
    probe.edges = []

    if removed == 0 or after > 1e-6:
        raise InstrumentationError(
            "intervention did not take effect: edge probability went from "
            "{:.6f} to {:.6f} with {} edits applied".format(before, after, removed)
        )
    if not math.isclose(after_row_sum, 1.0, abs_tol=1e-2):
        raise InstrumentationError(
            "row failed to renormalise after intervention: sum={:.6f}".format(after_row_sum)
        )
    # Two separate properties, and conflating them causes false alarms.
    #
    # Correctness: the edit must reach the objective's inputs. If the logits do
    # not move at all, the intervention is not connected to the output and
    # nothing downstream means anything.
    if logit_shift == 0.0:
        raise InstrumentationError(
            "removing the largest attention weight at query {} left the logits "
            "bit-identical. The edit reached the probabilities but not the "
            "output, so any delta measured this way would be structurally zero "
            "rather than small.".format(q)
        )

    # Resolvability: whether the loss can *represent* that change is a
    # different question, and a real limit rather than a bug. A model with a
    # tiny hidden state, or a long sequence averaging the effect away, can
    # propagate the edit correctly while the loss stays bit-identical because
    # the shift falls below one float ULP.
    loss_resolves = intervened_loss != baseline_loss
    return {
        "verified": True,
        "probe_layer": probe.layer_idx,
        "probe_query": q,
        "edge_prob_before": before,
        "edge_prob_after": after,
        "row_sum_after": after_row_sum,
        "baseline_loss": baseline_loss,
        "intervened_loss": intervened_loss,
        "loss_delta": intervened_loss - baseline_loss,
        "max_logit_shift": logit_shift,
        "loss_resolves_intervention": loss_resolves,
        "n_edits_applied": removed,
        "attention_shape": list(A.shape),
    }


@torch.no_grad()
def run_layer(
    model,
    layer_idx: int,
    x: torch.Tensor,
    labels: torch.Tensor,
    partition_size: int,
    rho: float,
    n_candidates: int,
    rng: np.random.Generator,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    """Score candidate edges in one layer of the frozen model."""
    rows: List[Dict[str, object]] = []
    with AttentionProbe(model, layer_idx) as probe:
        verification = verify_instrumentation(model, probe, x)

        probe.reset()
        probe.edges = []
        out = model(x, labels=labels)
        baseline = float(out.loss.item())
        A = probe.captured[0].float()          # (H, T, T)
        H, T, _ = A.shape

        # Reduce to statistics immediately: retaining full attention for every
        # layer of a 7B model at 2k context would be hundreds of gigabytes.
        supports = [top_p_support_mask(A[h], rho) for h in range(H)]

        picks: List[Tuple[int, int, int, float]] = []
        for _ in range(n_candidates * 8):
            h = int(rng.integers(0, H))
            # Later half for real context, but never T - 1: the shifted causal
            # loss discards that position's logits, so its delta is zero by
            # construction rather than by measurement.
            i = int(rng.integers(max(T // 2, 1), max(T - 1, 2)))
            row = supports[h][i].nonzero(as_tuple=True)[0]
            row = row[row <= i]
            if row.numel() == 0:
                continue
            j = int(row[int(rng.integers(0, row.numel()))])
            ov = edge_structural_overlap(supports[h], i, j, partition_size)
            picks.append((h, i, j, ov))
        picks.sort(key=lambda t: -t[3])
        picks = picks[:n_candidates]

        for h, i, j, ov in picks:
            probe.reset()
            probe.edges = [(h, i, j)]
            out_m = model(x, labels=labels)
            removed = probe.n_intervened
            probe.edges = []
            if removed == 0:
                continue
            intervened = float(out_m.loss.item())
            rows.append({
                "layer": layer_idx, "head": h, "query": i, "key": j,
                "overlap": ov, "baseline_loss": baseline,
                "intervened_loss": intervened,
                "delta_loss": intervened - baseline,
                "n_intervened": removed,
                "context_length": T,
            })
        del A, supports
    return rows, verification


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--dtype", default="float32",
                        choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--device_map", default=None,
                        help="e.g. 'auto' to shard across GPUs / offload to CPU")
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true",
                        help="off by default; only enable for models you trust")
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--layers", type=int, nargs="*", default=None,
                        help="layer indices to instrument; default is a spread of 3")
    parser.add_argument("--n_candidates", type=int, default=24)
    parser.add_argument("--partition_size", type=int, default=64)
    parser.add_argument("--rho", type=float, default=0.60)
    parser.add_argument("--matched_budget", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results_dir", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args(argv)

    device = resolve_device(args.device)
    results_dir = (Path(args.results_dir) if args.results_dir
                   else Path(__file__).resolve().parent.parent / "results" / "tier3")
    results_dir.mkdir(parents=True, exist_ok=True)

    print_header("Tier 3 - frozen large-model diagnostic")
    print("model_id={}  device={}  dtype={}".format(args.model_id, device, args.dtype))
    print("context_length={}  n_candidates={}".format(args.context_length, args.n_candidates))

    if args.dry_run:
        print("\nWould load {} and instrument layers {}".format(args.model_id, args.layers))
        return 0

    set_seed(args.seed)
    model, tokenizer = load_model(
        args.model_id, dtype=args.dtype, device_map=args.device_map,
        load_in_8bit=args.load_in_8bit, load_in_4bit=args.load_in_4bit,
        trust_remote_code=args.trust_remote_code,
    )
    if args.device_map is None:
        model = model.to(device)

    n_layers = len(decoder_layers(model))
    layers = args.layers or sorted({0, n_layers // 2, n_layers - 1})
    context_length = min(args.context_length, getattr(model.config, "max_position_embeddings", 4096))
    print("model has {} layers; instrumenting {}".format(n_layers, layers))

    vocab = int(getattr(model.config, "vocab_size", tokenizer.vocab_size))
    gen = torch.Generator().manual_seed(args.seed)
    x = torch.randint(0, vocab, (1, context_length), generator=gen).to(
        next(model.parameters()).device
    )
    labels = x.clone()

    rng = np.random.default_rng(args.seed)
    all_rows: List[Dict[str, object]] = []
    verifications: Dict[str, object] = {}

    status = Status.SMOKE if args.smoke else Status.COMPLETED
    record = RunRecord(
        run_id="tier3_{}_{}".format(
            args.model_id.replace("/", "_")[:40], context_length),
        experiment=EXPERIMENT, status=status,
        config={"model_id": args.model_id, "dtype": args.dtype,
                "context_length": context_length, "layers": layers,
                "n_candidates": args.n_candidates,
                "partition_size": args.partition_size, "rho": args.rho},
        seed=args.seed, context_length=context_length, dtype=args.dtype,
    )

    for layer_idx in layers:
        rows, verification = run_layer(
            model, layer_idx, x, labels, args.partition_size,
            args.rho, args.n_candidates, rng,
        )
        verifications["layer_{}".format(layer_idx)] = verification
        if not verification.get("loss_resolves_intervention", True):
            print("  layer {:<3} warning: the edit propagates (logits move by "
                  "{:.2e}) but the loss is bit-identical. Deltas from this "
                  "model are below float resolution and are not "
                  "measurements.".format(
                      layer_idx, verification.get("max_logit_shift", float("nan"))))
        all_rows.extend(rows)
        deltas = [r["delta_loss"] for r in rows]
        print("  layer {:<3} scored {:>3} edges | delta mean={:+.3e} max={:+.3e}".format(
            layer_idx, len(rows),
            float(np.mean(deltas)) if deltas else float("nan"),
            float(np.max(deltas)) if deltas else float("nan")))

    if not all_rows:
        raise SystemExit(
            "no edges were scored; refusing to write an empty Tier 3 dataset"
        )

    overlaps = [r["overlap"] for r in all_rows]
    deltas = [r["delta_loss"] for r in all_rows]
    corr = correlations(overlaps, deltas)

    # Analysis B: structurally similar, behaviorally different?
    ov_thr = float(np.quantile(overlaps, 0.75))
    hi = [r for r in all_rows if r["overlap"] >= ov_thr]
    hi_deltas = [r["delta_loss"] for r in hi]
    lo_thr = float(np.quantile(hi_deltas, 0.25)) if hi_deltas else float("nan")
    hi_thr = float(np.quantile(hi_deltas, 0.75)) if hi_deltas else float("nan")
    group_a = [r for r in hi if r["delta_loss"] <= lo_thr]
    group_b = [r for r in hi if r["delta_loss"] >= hi_thr]

    # Analysis C: at a matched removal budget, which selection hurts more?
    budget = min(args.matched_budget, len(all_rows))
    naive_pick = sorted(all_rows, key=lambda r: -r["overlap"])[:budget]
    contrib_pick = sorted(all_rows, key=lambda r: r["delta_loss"])[:budget]

    def group_loss(rows_sel: List[Dict[str, object]]) -> float:
        by_layer: Dict[int, List[Tuple[int, int, int]]] = {}
        for r in rows_sel:
            by_layer.setdefault(int(r["layer"]), []).append(
                (int(r["head"]), int(r["query"]), int(r["key"]))
            )
        # Only one layer can be probed at a time, so apply per layer and sum
        # the individual effects; this is an additive approximation and is
        # labelled as such in the results.
        total = 0.0
        for layer_idx, edges in by_layer.items():
            with AttentionProbe(model, layer_idx) as probe:
                probe.reset(); probe.edges = []
                base = float(model(x, labels=labels).loss.item())
                probe.reset(); probe.edges = edges
                out = model(x, labels=labels)
                if probe.n_intervened == 0:
                    raise InstrumentationError(
                        "group intervention on layer {} removed nothing".format(layer_idx)
                    )
                total += float(out.loss.item()) - base
                probe.edges = []
        return total

    summary = {
        "model_id": args.model_id,
        "n_model_layers": n_layers,
        "layers_instrumented": layers,
        "context_length": context_length,
        "n_edges": len(all_rows),
        "correlation": corr,
        "instrumentation_verification": verifications,
        "group_thresholds": {"overlap": ov_thr,
                             "low_contribution": lo_thr,
                             "high_contribution": hi_thr},
        "high_overlap_low_contribution": {
            "n": len(group_a),
            "mean_delta": float(np.mean([r["delta_loss"] for r in group_a])) if group_a else float("nan"),
            "mean_overlap": float(np.mean([r["overlap"] for r in group_a])) if group_a else float("nan"),
        },
        "high_overlap_high_contribution": {
            "n": len(group_b),
            "mean_delta": float(np.mean([r["delta_loss"] for r in group_b])) if group_b else float("nan"),
            "mean_overlap": float(np.mean([r["overlap"] for r in group_b])) if group_b else float("nan"),
        },
        "matched_budget_comparison": {
            "budget": budget,
            "naive_total_delta": group_loss(naive_pick),
            "contribution_total_delta": group_loss(contrib_pick),
            "note": "per-layer effects summed; an additive approximation, "
                    "since only one layer is probed at a time",
        },
        "interpretation_note": (
            "A weak correlation shows overlap predicts contribution poorly on "
            "this sample. It is not evidence of independence."
        ),
    }
    record.metrics = summary
    save_record(results_dir, record)

    tag = args.model_id.replace("/", "_")
    write_csv(results_dir / "edges_{}.csv".format(tag), all_rows)
    write_json(results_dir / "diagnostic_{}.json".format(tag), {
        "experiment": EXPERIMENT, "environment": environment(),
        "summary": summary, "rows": all_rows,
    })

    print_header("Results")
    print("edges scored          : {}".format(len(all_rows)))
    print("pearson r             : {:.3f} (p={:.3g}, n={})".format(
        corr.get("pearson_r", float("nan")), corr.get("pearson_p", float("nan")),
        corr.get("n", 0)))
    print("spearman r            : {:.3f}".format(corr.get("spearman_r", float("nan"))))
    print("high-ov / low-contrib : n={} mean delta={:+.3e}".format(
        len(group_a), summary["high_overlap_low_contribution"]["mean_delta"]))
    print("high-ov / high-contrib: n={} mean delta={:+.3e}".format(
        len(group_b), summary["high_overlap_high_contribution"]["mean_delta"]))
    print("matched budget {}      : naive={:+.4e}  contribution={:+.4e}".format(
        budget, summary["matched_budget_comparison"]["naive_total_delta"],
        summary["matched_budget_comparison"]["contribution_total_delta"]))
    print("\nWrote {}".format(results_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
