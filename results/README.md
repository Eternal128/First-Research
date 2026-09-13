# results/

**Nothing in this directory is tracked by git, and nothing in it is a finding.**

It is written by the scripts in `../scripts/` and is regenerable in minutes:

```bash
bash scripts/run_all.sh
```

## What lands here

```
results/
├── environment.json              what was installed when the pipeline ran
├── data_availability.json        which sources are present and reachable
├── validation/                   instrument-validation checks — read this first
├── dataset/<source>/             the arrivals table and preprocessing provenance
├── main/<source>/                model metrics, comparisons, predictions, leakage audit
├── subgroups/<source>/           per-stratum calibration, spatial maps
├── selection/<source>/           density-ratio diagnostics, strata, sensitivity bounds
├── downstream/<source>/          EPV shifts, decision displacement, decision curves
├── ablations/<source>/           the assumption sweeps and their spreads
└── power/                        corpus-size design guide
```

Every directory carries a `manifest.json` recording the script, the UTC
timestamp, the git revision, **whether the working tree was dirty**, the Python
and platform versions, the installed package versions, the full configuration
and its SHA-256. Two result sets with different config hashes were produced under
different settings and must not be compared.

## The default source is the simulator

Unless a real corpus has been obtained and a provider adapter completed, every
number here comes from `pcc.data.synthetic` and is tagged
`tracking_source = "simulated"`. The simulator's football is crude: Gaussian
formations, no offside, no possession structure, no fatigue.

**Simulated outputs describe the simulator, not football.** They exist to show
that the pipeline is complete and that the calibration estimator behaves as
specified. They must not appear in a results section.

## Where to look first

`validation/tables/instrument_validation.csv`. If any check there has failed,
nothing else in this directory means anything.
