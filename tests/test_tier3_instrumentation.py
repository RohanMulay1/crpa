"""
Tier 3 instrumentation, validated across attention layouts.

The large-model diagnostic patches into a frozen model's attention to read
probabilities and edit individual scores. That is the most fragile part of this
repository: it depends on internals Hugging Face is free to change, and on
every architecture implementing eager attention the same way.

These tests run the real probe against four genuinely different tiny models:

  Llama      standard multi-head attention
  Mistral    grouped-query attention, 4 query heads over 2 KV heads, which is
             what Llama-3-8B uses and the case most likely to break per-head
             targeting
  GPT-NeoX   a different module layout and 5 layers
  GPT-2      Conv1D projections and an `attn` submodule rather than `self_attn`

They are marked slow because they download weights. They are small: the whole
class is a few megabytes.

What is deliberately *not* claimed: that this works on a 7B or 8B model. These
establish that the mechanism is correct across layouts, not that it survives
scale. Section "What was not run" in the README says so.
"""

from __future__ import annotations

import math

import pytest
import torch

pytestmark = pytest.mark.slow

MODELS = [
    "hf-internal-testing/tiny-random-LlamaForCausalLM",
    "hf-internal-testing/tiny-random-MistralForCausalLM",
    "hf-internal-testing/tiny-random-GPTNeoXForCausalLM",
    "sshleifer/tiny-gpt2",
]

GQA_MODEL = "hf-internal-testing/tiny-random-MistralForCausalLM"


def _load(model_id, impl="eager"):
    """Load a tiny model, skipping the test if it cannot be fetched."""
    pytest.importorskip("transformers")
    from transformers import AutoModelForCausalLM

    try:
        return AutoModelForCausalLM.from_pretrained(
            model_id, attn_implementation=impl, dtype=torch.float32
        ).eval()
    except OSError as exc:
        pytest.skip("{} unavailable (offline?): {}".format(model_id, exc))


def _tokens(model, length=32):
    gen = torch.Generator().manual_seed(0)
    return torch.randint(0, int(model.config.vocab_size), (1, length), generator=gen)


class TestAcrossArchitectures:
    @pytest.mark.parametrize("model_id", MODELS)
    def test_layers_are_found(self, model_id):
        from experiments.large_model_diagnostic import decoder_layers

        model = _load(model_id)
        layers = decoder_layers(model)
        assert len(layers) >= 2

    @pytest.mark.parametrize("model_id", MODELS)
    def test_instrumentation_verifies(self, model_id):
        """Capture works, an edge is removed, and the row renormalises."""
        from experiments.large_model_diagnostic import AttentionProbe, verify_instrumentation

        model = _load(model_id)
        x = _tokens(model)
        with AttentionProbe(model, 0) as probe:
            result = verify_instrumentation(model, probe, x)

        assert result["verified"] is True
        assert result["edge_prob_before"] > 0, "test targeted an edge with no mass"
        assert result["edge_prob_after"] < 1e-6
        assert result["row_sum_after"] == pytest.approx(1.0, abs=1e-3)
        assert result["n_edits_applied"] == 1
        assert len(result["attention_shape"]) == 4

    @pytest.mark.parametrize("model_id", MODELS)
    def test_only_the_targeted_head_changes(self, model_id):
        """Per-head targeting must not leak across heads."""
        from experiments.large_model_diagnostic import AttentionProbe

        model = _load(model_id)
        x = _tokens(model)
        with AttentionProbe(model, 0) as probe, torch.no_grad():
            probe.reset()
            probe.edges = []
            model(x)
            before = probe.captured.clone()

            n_heads = before.shape[1]
            if n_heads < 2:
                pytest.skip("model has a single head; nothing to isolate")
            query = before.shape[-2] - 2
            key = int(before[0, 0, query, :query + 1].argmax())

            probe.reset()
            probe.edges = [(0, query, key)]
            model(x)
            after = probe.captured

        assert float(after[0, 0, query, key]) == pytest.approx(0.0, abs=1e-7)
        for head in range(1, n_heads):
            assert torch.allclose(before[:, head], after[:, head], atol=1e-6), (
                "editing head 0 changed head {}".format(head)
            )

    @pytest.mark.parametrize("model_id", MODELS)
    def test_the_intervention_reaches_the_output(self, model_id):
        """The edit must propagate to the logits.

        This is the correctness property, and it is distinct from whether the
        *loss* can represent the change. sshleifer/tiny-gpt2 has a 2-dimensional
        hidden state, so it propagates the edit correctly while the loss stays
        bit-identical: the shift is below one float32 ULP. That is a
        resolvability limit, not a broken probe, and the two are asserted
        separately.
        """
        from experiments.large_model_diagnostic import AttentionProbe

        model = _load(model_id)
        x = _tokens(model)
        with AttentionProbe(model, 0) as probe, torch.no_grad():
            probe.reset()
            probe.edges = []
            out = model(x, labels=x)
            base_logits = out.logits.clone()
            captured = probe.captured
            query = captured.shape[-2] - 2
            key = int(captured[0, 0, query, :query + 1].argmax())

            probe.reset()
            probe.edges = [(0, query, key)]
            cut = model(x, labels=x)
            removed = probe.n_intervened
            shift = (cut.logits - base_logits).abs()

        assert removed == 1
        assert float(shift.max()) > 0.0, "the edit never reached the logits"

        # Causality: positions before the intervened query cannot be affected.
        # Positions after it can be, and generally are, because later layers
        # let them attend to the position whose representation changed - so
        # asserting that exactly one position moves would be wrong for any
        # model with more than one layer.
        per_position = shift.sum(dim=-1)[0]
        assert float(per_position[:query].max()) == 0.0, (
            "an intervention at query {} changed an earlier position, which "
            "violates causality".format(query)
        )
        assert float(per_position[query]) > 0.0

    @pytest.mark.parametrize("model_id", MODELS)
    def test_verification_reports_whether_the_loss_can_resolve_it(self, model_id):
        """Correctness and resolvability are reported as separate facts."""
        from experiments.large_model_diagnostic import AttentionProbe, verify_instrumentation

        model = _load(model_id)
        x = _tokens(model)
        with AttentionProbe(model, 0) as probe:
            result = verify_instrumentation(model, probe, x)

        assert result["max_logit_shift"] > 0.0
        assert isinstance(result["loss_resolves_intervention"], bool)
        if result["loss_resolves_intervention"]:
            assert result["loss_delta"] != 0.0
        else:
            # A degenerate model: propagates correctly, unmeasurable in loss.
            assert result["loss_delta"] == 0.0
            assert model.config.hidden_size <= 8 or x.shape[1] >= 1024


class TestGroupedQueryAttention:
    """GQA is what Llama-3-8B uses, so it is the layout that matters most."""

    def test_kv_heads_really_are_fewer(self):
        model = _load(GQA_MODEL)
        cfg = model.config
        assert cfg.num_key_value_heads < cfg.num_attention_heads, (
            "this model is not actually GQA; the test would prove nothing"
        )

    def test_attention_is_over_query_heads_not_kv_heads(self):
        """Per-head edits must address the expanded query-head axis.

        GQA repeats KV heads up to the query-head count before the softmax, so
        the probability tensor has num_attention_heads rows. Targeting the KV
        axis instead would silently hit the wrong head.
        """
        from experiments.large_model_diagnostic import AttentionProbe

        model = _load(GQA_MODEL)
        x = _tokens(model)
        with AttentionProbe(model, 0) as probe, torch.no_grad():
            probe.reset()
            probe.edges = []
            model(x)
            shape = probe.captured.shape

        assert shape[1] == model.config.num_attention_heads
        assert shape[1] != model.config.num_key_value_heads

    def test_every_query_head_is_independently_addressable(self):
        from experiments.large_model_diagnostic import AttentionProbe

        model = _load(GQA_MODEL)
        x = _tokens(model)
        n_heads = model.config.num_attention_heads
        with AttentionProbe(model, 0) as probe, torch.no_grad():
            probe.reset()
            probe.edges = []
            model(x)
            base = probe.captured.clone()
            query = base.shape[-2] - 2

            for head in range(n_heads):
                key = int(base[0, head, query, :query + 1].argmax())
                probe.reset()
                probe.edges = [(head, query, key)]
                model(x)
                assert float(probe.captured[0, head, query, key]) == pytest.approx(
                    0.0, abs=1e-7
                ), "head {} was not editable".format(head)
                assert probe.n_intervened == 1


class TestRefusals:
    """The diagnostic must fail loudly rather than report a silent null."""

    def test_non_eager_attention_is_refused(self):
        """SDPA never materialises probabilities, so nothing can be measured."""
        from experiments.large_model_diagnostic import (
            AttentionProbe,
            InstrumentationError,
            verify_instrumentation,
        )

        model = _load(MODELS[0], impl="sdpa")
        x = _tokens(model)
        with pytest.raises(InstrumentationError, match="eager"):
            with AttentionProbe(model, 0) as probe:
                verify_instrumentation(model, probe, x)

    def test_an_out_of_range_edge_counts_as_zero_edits(self):
        from experiments.large_model_diagnostic import AttentionProbe

        model = _load(MODELS[0])
        x = _tokens(model)
        with AttentionProbe(model, 0) as probe, torch.no_grad():
            probe.reset()
            probe.edges = [(0, 9999, 9999)]
            model(x)
            assert probe.n_intervened == 0

    def test_an_out_of_range_layer_is_rejected(self):
        from experiments.large_model_diagnostic import AttentionProbe

        model = _load(MODELS[0])
        with pytest.raises(IndexError, match="out of range"):
            AttentionProbe(model, 999)

    def test_capture_does_not_leak_across_layers(self):
        """A later layer's softmax must not overwrite the edited layer's.

        This is a regression test: the probe originally patched softmax
        globally, so layer 1 overwrote layer 0's capture and an intervention
        that had taken effect looked inert.
        """
        from experiments.large_model_diagnostic import AttentionProbe, decoder_layers

        model = _load(MODELS[2])   # GPT-NeoX, 5 layers
        assert len(decoder_layers(model)) >= 3
        x = _tokens(model)
        with AttentionProbe(model, 0) as probe, torch.no_grad():
            probe.reset()
            probe.edges = [(0, 31, 5)]
            model(x)
            assert float(probe.captured[0, 0, 31, 5]) == pytest.approx(0.0, abs=1e-7)
            assert float(probe.captured[0, 0, 31].sum()) == pytest.approx(1.0, abs=1e-3)

    def test_the_probe_restores_forward_and_softmax_on_exit(self):
        """A leaked patch would corrupt every later measurement in the process."""
        import torch.nn.functional as F

        from experiments.large_model_diagnostic import AttentionProbe

        model = _load(MODELS[0])
        original_softmax = F.softmax
        attn_forward = AttentionProbe(model, 0).attn.forward

        with AttentionProbe(model, 0):
            assert F.softmax is not original_softmax
        assert F.softmax is original_softmax
        assert AttentionProbe(model, 0).attn.forward == attn_forward
