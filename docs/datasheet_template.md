# Datasheet — `<corpus name>`

Complete one per corpus actually used. Following Gebru et al. (2021), trimmed to
the questions that matter for this study.

## Motivation
- Why was the dataset created, and by whom?
- Who funded it?

## Composition
- What does an instance represent (a frame? an event? a match?)
- How many matches, competitions, seasons, arrivals?
- **Which of the study's required fields are present?** *(complete the table in
  `docs/data_sources.md` §1 and record the answers here)*
- **Is velocity supplied, or derived here?**
- **Is a frame-level possession label supplied, or derived from touch events?**
- **Is ball height present?**
- Is any data missing, and how is missingness represented (absent rows, nulls,
  an explicit visibility flag)?
- Does the dataset contain personal data? *(positional tracking of identifiable
  individuals does.)*

## Collection
- How was the data acquired (optical multi-camera, broadcast video, manual
  annotation)?
- Sampling rate, and is it constant across matches?
- What is the stated positional accuracy? **Has it been independently checked?**
- Over what time frame was it collected?

## Preprocessing by the provider
- What smoothing, interpolation or identity resolution has the provider already
  applied? *(this compounds with the study's own preprocessing and is often
  undocumented — record what is knowable and flag what is not)*
- Is the raw data available, or only the processed form?

## Preprocessing by this study
- Reference the provenance table written by `scripts/02_build_dataset.py`.
- Retention rate through each filter.
- Estimated synchronisation offset per match, and the residual.
- Fraction of arrivals with imputed flight time.
- Frame-completeness distribution.

## Uses
- What has the dataset been used for previously?
- What tasks is it unsuitable for?
- **Does anything about its composition bias this study's estimand?** *(a single
  tournament is a single competition; a broadcast corpus systematically omits
  off-camera players)*

## Distribution
- Licence, and who holds it.
- May derived per-arrival forecasts and outcomes be published? *(this is what
  would let a third party recompute every calibration statistic — worth asking
  explicitly)*
- May aggregated results be published? *(assumed yes; confirm)*

## Maintenance
- Who maintains it; is it versioned; will it be updated?
- The exact version/snapshot used here, and its checksum if available.
