# Baselines

We compare LinkRank against four representative issue–commit linking baselines.
The baseline implementations are **not** redistributed in this repository; they are
available from their respective papers and replication packages listed below.

| Baseline | Reference |
|---|---|
| **EALink** | Zhang, C., Wang, Y., Wei, Z., Xu, Y., Wang, J., Li, H., Ji, R., 2023. *EALink: An Efficient and Accurate Pre-trained Framework for Issue-Commit Link Recovery.* ASE 2023. |
| **MPLinker** | Wang, B., Deng, Y., Luo, R., Liang, P., Bi, T., 2025. *MPLinker: Multi-template Prompt-tuning with Adversarial Training for Issue-Commit Link Recovery.* Journal of Systems and Software. |
| **EasyLink** | Huang, H., Widyasari, R., Zhang, T., Irsan, I.C., Shi, J., Ang, H.W., Liauw, F., Ouh, E.L., Shar, L.K., Kang, H.J., et al., 2025. *Back to the Basics: Rethinking Issue-Commit Linking with LLM-Assisted Retrieval.* arXiv:2507.09199. |
| **LinkAnchor** | Akhavan, A., Hosseinpour, A., Heydarnoori, A., Keshani, M., 2025. *LinkAnchor: An Autonomous LLM-Based Agent for Issue-to-Commit Link Recovery.* arXiv:2508.12232. (Adapted to the one-to-many setting; evaluated with GPT-OSS-120B as described in Section 4.3 of our paper.) |

## Evaluation protocol

All baselines were evaluated on the same K ≤ 7 datasets (`k7_dataset/`) with **identical
5-fold stratified splits** (stratified by K, seed 42) as LinkRank. Each baseline produces
per-candidate scores or ranked outputs, from which we derive the same set-based
(Precision / Recall / F1 under Known-K, ABS, REL) and ranking (MRR, NDCG@K, P@K)
metrics used for LinkRank.

The per-dataset baseline results are provided in:

```
results/ealink_k7_5fold_{dataset}/
results/easylink_k7_5fold_{dataset}/
results/mplinker_k7_5fold_{dataset}/
results/gptoss_baseline_{dataset}/     # LinkAnchor-style GPT-OSS-120B evaluation
```

where `{dataset}` ∈ {pytorch, beam, datafusion, dubbo, iceberg, mxnet}.
