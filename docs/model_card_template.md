# Model card — `<model name>`

Complete one per model actually reported. Following Mitchell et al. (2019).
The point of this template is that the questions it asks are ones a
pitch-control model is not usually asked.

## Model details
- **Name and version:**
- **Type:** unfitted geometric / unfitted physical / fitted statistical / neural
- **Implementation:** `src/pcc/models/…`, git revision
- **Free parameters, and how each was set:** *(distinguish assumed from
  estimated for every one — this is the central question for a physics model)*
- **Inputs:** *(exactly which state variables; note anything the other models
  in the comparison do not receive)*
- **Output:** scalar in `[0, 1]`, interpreted as ___
- **Training data, if fitted:** matches, competitions, date range, n arrivals
- **Compute:**

## Intended use
- **Intended:** forecasting the probability that team A controls the ball at a
  realised arrival point, at a stated horizon.
- **Out of scope:** *(state explicitly)* — untargeted regions of the pitch;
  set pieces, if excluded; competitions and player populations not represented
  in the evaluation; any use of the raw value as a probability if the
  calibration results below say it is not one.

## Evaluation
- **Corpus, split and outcome definition:**
- **Brier / log loss / spherical:**
- **CORP MCB (cross-fitted), DSC, UNC:**
- **Calibration slope and intercept, with intervals:**
- **AUC and average precision:**
- **Forecast dispersion:**
- **Reliability curve:** *(figure reference)*

## Calibration status — the section that matters here
- **Is the raw output a calibrated probability?** yes / no / only after
  recalibration / only within stated strata
- **Where it is not calibrated:** *(zones, pass types, tracking sources)*
- **Recommended recalibration, if any:** map, where it was fitted, whether it
  transfers across competitions
- **Residual miscalibration after recalibration:**

## Limitations
- **Assumptions that could fail:**
- **Known failure modes:**
- **Sensitivity to assumed parameters:** *(quote the ablation range for the
  headline statistic)*
- **Behaviour under distribution shift:**
- **Behaviour with incomplete tracking:**

## Ethical and practical considerations
- **Individual-level inference:** is any player-level claim being made, and is
  that appropriate given the uncertainty?
- **Deployment risk:** what decision would be made wrongly if the calibration
  result above were ignored?
