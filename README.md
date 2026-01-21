# FOMC Press Conference Analytics

This project implements a modular toolkit for studying the words Federal Reserve
Chair Jerome Powell uses in his press conferences and comparing them to
contracts offered in markets such as Kalshi.  It provides tools for
parsing transcripts, extracting Powell‑only remarks, counting occurrences
of contract phrases, estimating mention probabilities, and simulating
trades against historical Kalshi price data.

The goal of this toolkit is to help researchers and traders build a
systematic view of which phrases are genuinely under‑ or over‑priced in
FOMC mention markets.  By using a reproducible codebase, you can
avoid ad‑hoc guesses and instead rely on data gathered from multiple
press conferences.

## Features

* **Transcript ingestion** – Load press conference transcripts from PDF
  or plain text.  The included parser uses
  [PyMuPDF](https://pymupdf.readthedocs.io/) to read PDF files and
  converts them into clean text.  Additional loaders can be added
  easily.
* **Chair-only extraction** – Identify lines spoken by the FOMC chair (e.g., `CHAIR POWELL`) and
  ignore questions from reporters or remarks from moderators.  The
  parser uses simple regular expressions to detect speaker changes.
* **Phrase mapping** – Maintain a configurable mapping from market
  contract names to lists of phrases (synonyms) that count as a
  "mention".  For example, the contract `AI / Artificial Intelligence`
  maps to the variants `"AI"` and `"artificial intelligence"`.
* **Recency‑weighted probabilities** – Estimate the probability of a
  phrase appearing using historical pressers.  The default
  implementation uses an exponentially weighted moving average (EWMA)
  of binary mention events, but you can plug in your own models (e.g.
  Beta–Binomial or logistic regression).
* **Kalshi API integration** – A small wrapper around the
  [Kalshi API](https://docs.kalshi.com/) to fetch market data and
  trade.  You must provide your own API credentials; the code does
  not hard‑code any secrets.
* **Backtester** – Simulate trades based on your model’s output.
  The backtester applies configurable edge thresholds, position sizing
  rules (e.g. fractional Kelly or fixed fraction of capital), and
  realistic bid/ask spreads.
* **Extensible architecture** – Each component (loader, parser,
  model, backtester) is a separate module.  You can swap in your own
  implementations or extend functionality without modifying the rest
  of the codebase.

## Quick Start

### Prerequisites

1. **Python 3.10 or higher.**  Older versions may work but are
   unsupported.
2. **[`uv`](https://github.com/astral-sh/uv)** – a fast drop‑in
   replacement for `pip` that caches wheels and builds.  You can
   install it globally via:

   ```sh
   pip install uv
   ```

### Installation

Clone the repository and change into its directory:

```sh
git clone <url-of-this-repo> fomc-analysis
cd fomc-analysis
```

Create a virtual environment (optional but recommended):

```sh
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies using `uv`:

```sh
uv sync --extra dev
```

### Adding Transcripts

Place all PDF or text transcripts in a directory of your choice,
for example `data/transcripts`.  The parser will read each file,
convert PDF to text automatically, and extract Powell‑only text in
memory.  You can store the parsed transcripts or counts via the
subcommands described below.  There is no intermediate cache by
default; the parser reads files directly each time.

### Running the Parser

Once dependencies are installed, you can extract Powell‑only
remarks and compute phrase counts.  Use the `count` subcommand of
the CLI to load transcripts and count mentions.  For example:

```sh
python -m powell_analysis.main count \
    --transcripts-dir data/transcripts \
    --contract-mapping configs/contract_mapping.yaml \
    --output counts.csv
```

This command parses all transcripts in `data/transcripts`, counts
how many times each contract phrase appears in Powell’s remarks,
and writes the counts to `counts.csv` (one row per presser, one
column per contract).  The contract mapping file (`.yaml` or
`.json`) defines which phrases map to which contracts.

### Estimating Mention Probabilities

The `model.py` module includes simple estimators and you can use
the CLI to compute probabilities.  To estimate mention
probabilities from the counts file, use the `estimate` subcommand:

```sh
python -m fomc_analysis.main estimate \
    --counts-file counts.csv \
    --model ewma \
    --alpha 0.5 \
    --output estimates.csv
```

The `--model` argument selects which estimator to use: `ewma` for
exponential smoothing, `beta` for a Beta–Binomial estimator, or
`logistic` for a logistic regression model.  For EWMA, `--alpha`
controls recency weighting; a higher value weights recent pressers
more heavily.  For the Beta–Binomial estimator, you can specify
`--alpha-prior`, `--beta-prior` and optionally `--half-life` to
define the prior and decay rate.  Logistic regression uses lagged
events as features by default.

### Backtesting Trades

To evaluate whether your probabilities translate into profitable
trades, use the `backtest` subcommand.  You need historical price
data, which you can download via the Kalshi API or from a CSV file:

```sh
python -m fomc_analysis.main backtest \
    --price-file data/prices/kxfedmention-26jan-day.csv \
    --predictions estimates.csv \
    --edge-threshold 0.05 \
    --initial-capital 1000.0 \
    --output backtest_results.json
```

The backtester compares your model’s probabilities with market
probabilities (implied from prices), applies an edge threshold
(e.g. only bet when your edge exceeds 5 percentage points), sizes
positions according to a fractional‑Kelly rule, and outputs final
capital and a list of trades in JSON format.

### Using the Kalshi API

The `kalshi_api.py` module shows how to authenticate with Kalshi,
fetch market prices, and submit orders.  **You must supply your own
API key and secret.**  Store these in a `.env` file or environment
variables (`KALSHI_API_KEY`, `KALSHI_API_SECRET`).  See
`kalshi_api.py` for details.

### Directory Layout

```
powell_analysis/
├── README.md            – this guide
├── pyproject.toml        – project metadata used by uv/PEP 621
├── requirements.txt      – optional requirements file for uv
├── configs/
│   └── contract_mapping.yaml – default mapping of contracts to phrases
├── src/
│   └── powell_analysis/
│       ├── __init__.py
│       ├── data_loader.py
│       ├── contract_mapping.py
│       ├── feature_extraction.py
│       ├── model.py
│       ├── backtester.py
│       ├── kalshi_api.py
│       ├── utils.py
│       └── main.py
└── tests/                 – (optional) unit tests
```

### Extending the Toolkit

You can extend this project by implementing additional models or
features.  Here are a few ideas:

* **Natural language processing:**  Use a large language model or
  embeddings to capture semantic meaning instead of exact phrase
  matches.  For example, you could cluster similar questions and
  answers and map them to contract categories.
* **Bayesian updating:**  Use a Beta–Binomial model to update
  mention probabilities with uncertainty estimates.  This can help
  decide when an observed mention frequency is statistically
  significant.
* **Topic modelling:**  Analyze transcripts for high‑level themes
  (e.g. inflation, labor market, AI) and build cross‑contract
  correlations.  Use these features to adjust probabilities when
  multiple contracts are related.
* **Advanced backtesting:**  Incorporate order book depth, latency,
  and realistic transaction costs.  Use walk‑forward cross
  validation to avoid lookahead bias.

## Notes on Accuracy and Responsibility

This codebase is intended for educational and research purposes.  It
does not provide financial advice.  Always understand the rules of
the market you are trading, ensure that your code replicates the
resolution criteria precisely (e.g. does the contract count
questions from reporters?), and backtest thoroughly before risking
capital.

If you find bugs or have ideas for improvement, feel free to open an
issue or submit a pull request.  Happy hacking!
