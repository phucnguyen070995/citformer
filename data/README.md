# Data

The datasets are not included in the repository. Place the files below in this directory (they may sit
in sub-directories; a `.gz` version of any file is also accepted), or pass another directory with
`--data-dir` or the `CITFORMER_DATA` environment variable.

| Dataset | File | C | Rows | Step | Period | Target | MD5 (uncompressed) |
|---|---|---|---|---|---|---|---|
| ETTh1 | `ETTh1.csv` | 7 | 17,420 | 1 h | 24 | `OT` | `8381763947c85f4be6ac456c508460d6` |
| ETTm1 | `ETTm1.csv` | 7 | 69,680 | 15 min | 96 | `OT` | `82d6bd89109c63d075d99c1077b33f38` |
| Exchange | `exchange_rate.csv` | 8 | 7,588 | 1 day | 7 | last column (`8`) | `abac0026bf1345d44598eced8f2214e2` |
| Weather | `weather.csv` | 21 | 52,696 | 10 min | 144 | `OT` | `2e79e5e4c921d3542684c4a34c6748b4` |
| Covid | `covid.csv` | 55 | 335 | 1 day | 7 | last column (`55`) | `8260bd7a461caccb10f474acd6a547f5` |
| Solar | `solar_AL.txt` or `solar_AL.txt.gz` | 137 | 52,560 | 10 min | 144 | last column | `de1d4810540a960549c2ce0f05c42974` |
| ECL | `electricity.csv` | 321 | 26,304 | 1 h | 24 | `OT` | `a1973ba4f4bed84136013ffa1ca27dc8` |
| Traffic | `traffic.csv` | 862 | 17,544 | 1 h | 24 | `OT` | `a62d8f2cd2c6acaaaa6f7aa819e378c0` |

C is the number of variables (the `date` column is dropped). `solar_AL.txt` has no header row; every
other file is a CSV with a `date` column first. The target is always the last column. "Period" is the
seasonal period in rows used by the EDA and the Seasonal-Naive baseline.

## Sources

- **ETTh1, ETTm1** — Electricity Transformer Temperature, 2016-07 to 2018-06, from
  [zhouhaoyi/ETDataset](https://github.com/zhouhaoyi/ETDataset) (`ETT-small/`).
  Zhou et al., *Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting*,
  AAAI 2021.
- **Exchange, Solar, ECL, Traffic** — introduced by LSTNet and distributed in
  [laiguokun/multivariate-time-series-data](https://github.com/laiguokun/multivariate-time-series-data).
  Lai et al., *Modeling Long- and Short-Term Temporal Patterns with Deep Neural Networks*, SIGIR 2018.
  Solar is used as published there (`solar-energy/solar_AL.txt.gz`: 137 photovoltaic plants in
  Alabama, 2006). For Exchange, ECL and Traffic the CSV versions with a `date` column (and `OT` as the
  last column of ECL and Traffic) are used; these are the files distributed with Autoformer and
  Time-Series-Library (dataset links in the READMEs of
  [thuml/Autoformer](https://github.com/thuml/Autoformer) and
  [thuml/Time-Series-Library](https://github.com/thuml/Time-Series-Library)).
- **Weather** — 21 meteorological indicators every 10 minutes during 2020, recorded at the weather
  station of the Max Planck Institute for Biogeochemistry, Jena
  ([bgc-jena.mpg.de/wetter](https://www.bgc-jena.mpg.de/wetter/)); the `weather.csv` file distributed with
  Autoformer / Time-Series-Library. Wu et al., *Autoformer: Decomposition Transformers with
  Auto-Correlation for Long-Term Series Forecasting*, NeurIPS 2021.
  The file contains the missing-value code -9999 in 81 cells; `citformer/data.py` replaces these by
  linear interpolation along time.
- **Covid** — `dataset/covid.csv` of [aikunyi/FreTS](https://github.com/aikunyi/FreTS)
  (raw: <https://raw.githubusercontent.com/aikunyi/FreTS/main/dataset/covid.csv>; the file used here is
  byte-identical, git blob `546335ef2174293d6c7a7d6598b2135a908805b8`). The dataset was introduced by
  FourierGNN (Yi et al., *FourierGNN: Rethinking Multivariate Time Series Forecasting from a Pure Graph
  Perspective*, NeurIPS 2023), whose appendix describes it as COVID-19 data for California from
  2020-02-01 to 2020-12-31, daily, built from the Johns Hopkins University CSSE repository
  ([CSSEGISandData/COVID-19](https://github.com/CSSEGISandData/COVID-19)). It is also used by FreTS
  (Yi et al., *Frequency-domain MLPs are More Effective Learners in Time Series Forecasting*,
  NeurIPS 2023). The 55 columns are unnamed integer IDs (`1` to `55`) with non-negative integer daily
  counts; all values are 0 before 2020-04-01. The column-to-region mapping is not documented by the
  source. This dataset is unrelated to, and much smaller than, the Covid-19 dataset of the TFB benchmark
  used in the DUET paper.
