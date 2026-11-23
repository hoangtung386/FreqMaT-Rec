# FreqMaT architecture

```mermaid
flowchart TD
    IN["Item sequence (max_seq_len items)"] --> EMB["Item Embedding (n_items x D)"]
    EMB --> LN0["Dropout + LayerNorm, zero padding"]
    LN0 --> L1

    subgraph L1["FreqMaTLayer (x n_layers)"]
        direction TB
        PRE["FreqFilter (freq_position=pre)<br/>residual + LayerNorm"] --> SPLIT(( ))
        SPLIT --> M["Mamba-1 / Mamba-3 (SSM)<br/>long-range, O(N)"]
        SPLIT --> A["RoPE causal attention<br/>local interactions"]
        SPLIT -. "freq_position=branch" .-> FB["FreqFilter.filter (parallel branch)"]
        M --> G["GatedFusion<br/>2 branches: sigmoid gate<br/>3 branches: softmax gate"]
        A --> G
        FB -.-> G
        G --> R1["Add + LayerNorm"]
        R1 --> FFN["MoEFFN (8 experts, top-2) or dense FFN"] --> R2["Add + LayerNorm"]
    end

    L1 --> LAST["State at the last real item"]
    LAST --> SCORE["Dot product with all item embeddings (shared weights)"]
    SCORE --> OUT["Scores for every item -> Top-K"]
```

## Options (`configs/model/freqmat.yaml`, validated by `FreqMaTConfig`)

| Option | Values | Meaning |
|---|---|---|
| `use_freq`, `freq_type` | `none`, `learned`, `bsa`, `both`, `gft` | FMLP-style filter / BSARec rescaler / cascade / time-weighted graph Fourier |
| `freq_position` | `pre`, `branch` | sequential denoiser vs. parallel branch fused by the gate |
| `use_mamba`, `mamba_type` | `mamba1`, `mamba3` | SSM branch; `mamba3` needs CUDA + `mamba-ssm` |
| `use_attn`, `attn_window` | `0` = full causal | attention branch |
| `ffn_type` | `moe`, `dense` | MoE replaces the FFN (param-matched by default) |
| `loss_type` | `CE`, `BPR` | BPR needs negative sampling configured |

## Source map

| Concept | File |
|---|---|
| Model (RecBole `SequentialRecommender`) | `src/freqmat/models/freqmat.py` |
| One layer | `src/freqmat/models/layer.py` |
| Hyper-parameters | `src/freqmat/models/config.py` |
| Frequency filter / graph basis | `models/modules/frequency.py`, `graph_fourier.py` |
| Mamba / attention / fusion / FFN+MoE | `models/modules/mamba.py`, `attention.py`, `fusion.py`, `ffn.py` |

Parameter counts depend on the dataset (embedding is `n_items x hidden_size`) and variant; read them from
`metrics.json` (`params`) of a run instead of this document.
