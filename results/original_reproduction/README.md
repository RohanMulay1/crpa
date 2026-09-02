# Reproducing the original implementation

`run_l40s.log` is the full output of the original code at commit 7474c77, run
on this branch's hardware with exactly one source change: the dataset id in
`data.py`, because newer `huggingface_hub` rejects the bare `wikitext` repo
name and the script cannot start without it.

```diff
-    ds = load_dataset('wikitext', 'wikitext-2-raw-v1')
+    ds = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1')
```

## Result

| variant | published | reproduced | |
|---|---|---|---|
| Dense Transformer | 50.9% | 55.6% | reproduces |
| Sliding Window | 51.9% | 52.5% | reproduces |
| CRPA no reg. | 8.4% | 10.9% | reproduces |
| CRPA naive reg. | 5.3% | 5.3% | reproduces exactly |
| CRPA causal reg. | **32.8%** | **7.5%** | does not reproduce |

Chance accuracy on this task is 5.0%. The reproduced value for the paper's
headline configuration is barely above chance and below its own
no-regularization baseline. The original's own verification code reports this:

```
S3 Naive reduces overlap most: 0.256 < 0.237  -> FAILED
S3 But naive hurts retrieval:  5.3% < 7.5%    -> VERIFIED
S5 Causal beats no-reg:        7.5% > 10.9%   -> FAILED
```

## Environment

L40S 46GB, torch 2.4.1+cu124, Python 3.11, bf16 autocast, single seed (42), as
published. The original README names an RTX A6000 with torch 2.10.0+cu124 and
Python 3.10. That difference cannot be excluded in principle, but four of the
five rows reproduce on this stack, so it is an unlikely explanation for the
fifth.

## Why it was never a measurement

See F1 and F2 in the main README. Candidates were pairs of query rows scored by
support Jaccard, but the intervention masked the edge `i -> j`, a different
object. Because the mask is causal and the two indices were drawn
independently, every sample with `i < j` masked an already-absent entry: a
no-op with delta identically zero, which the `delta <= eps` rule then filed as
redundant. The threshold compounded it, sitting about four orders of magnitude
above the observed delta scale, so it would have admitted those no-ops even had
they been real.
