# Dataset

The full LinkRank dataset (six projects, K ≤ 7) is hosted on **Zenodo**:

> **https://zenodo.org/records/21197786?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjU0ZjcxMzNmLTNjY2QtNDg4Zi04ZDU0LWZlNDI3YzI4YzczNyIsImRhdGEiOnt9LCJyYW5kb20iOiJhZTkwMzg4MGNhMWEwZjc2N2Q4M2JhZTVkNjVmYjI3NiJ9.Vh7c9vyDwirHDAX-WN9Q02lMwZxk_xBJfODDlYeOuu9LbmaPCLVFBcEKhwLuZZx_cfQqKyH0LX48VES-gxOjAA**

<!-- TODO: after the Zenodo record is published, replace the tokenized link above
     with the public DOI link: https://zenodo.org/records/21197786 -->

Download the archive and extract the six project folders into this `Dataset/`
directory so the layout looks like:

```
Dataset/
├── beam/
├── datafusion/
├── dubbo/
├── iceberg/
├── mxnet/
└── pytorch/
```

## Files per project

| File | Description |
|---|---|
| `rds_issues.csv` | Issues: `Issue ID`, `Issue Date`, `Title`, `Description`, `Labels`, `Comments` |
| `rds_commits.csv` | Commits: `Commit ID`, `Commit Date`, `Message`, `Diff Summary`, `File Changes`, `Full Diff` |
| `rds_links.csv` | Candidate pairs: `Issue ID`, `Commit ID`, `Output` (1 = true link, 0 = candidate) — the RDS candidate pools used for training and evaluation |
| `issues.csv` / `commits.csv` / `links.csv` | Raw issue–commit link data before RDS pool construction (where available) |

## Dataset statistics (K = 1..7)

| Project | Issues | Commits | Avg. Candidate Pool | Avg K |
|---|---|---|---|---|
| Apache Beam | 671 | 1,625 | ~590 | 2.42 |
| Apache DataFusion | 738 | 2,270 | ~1,264 | 3.08 |
| Apache Dubbo | 469 | 938 | ~133 | 2.00 |
| Apache Iceberg | 551 | 1,357 | ~193 | 2.46 |
| Apache MXNet | 383 | 903 | ~427 | 2.36 |
| PyTorch | 291 | 595 | ~439 | 2.04 |
| **Total** | **3,103** | **7,688** | | **2.48** |

Issue and commit counts refer to issues with at least one true link inside the
RDS candidate pool (`rds_links.csv`, `Output = 1`) — i.e., exactly the data used
in the paper's experiments.

## Construction

Candidate pools are built with the **Relative Date Span (RDS)** strategy: for each
issue, all commits within the time span of its ground-truth commits extended by a
±365-day margin are included as candidates (see Section 3.1 of the paper).
