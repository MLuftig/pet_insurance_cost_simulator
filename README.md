# Pet Insurance Claims Prediction: Breed Predisposition, Pricing, and Portfolio Risk Simulation

## Executive Summary
This project asks a real underwriting question: **using only what's known about a pet at the moment of enrollment — species, breed, age — how well can we predict their future insurance claims?** It builds a breed-level disease predisposition feature set from real, cited veterinary research, tests whether it meaningfully improves claims prediction, and — after finding real, honest limits to that approach — pivots to where the underlying model actually performs well: **portfolio-level Monte Carlo risk simulation**, validated to within 0.04% of real observed data, plus a deployed individual-pet cost simulator.

* **Business Goal:** Move beyond flat demographic pricing (species/breed/age alone) toward risk-adjusted pricing informed by real clinical knowledge, and give both a reserving team (portfolio view) and an individual underwriter (per-pet view) a genuine cost-distribution forecast rather than a single point estimate.
* **Key Result:** Breed-specific disease predisposition, engineered from real veterinary literature, is a real but modest signal for *individual* claims prediction (AUC 0.57-0.58 — barely above chance) — age and species dominate almost entirely. That same model, applied at the **portfolio level** via Monte Carlo simulation, produces a well-calibrated, validated cost-distribution forecast that closely matches real historical totals. At the **individual level**, the model's own output reveals something worth stating explicitly: for most pet profiles, the single most likely first-year outcome is genuinely **$0** — which isn't a weak result, it's the actual statistical shape of why insurance exists.
* **Actionable Recommendation:** Use demographic + predisposition features for portfolio-level reserving and stress-testing (where they work well), not as the primary driver of individual policy pricing (where their predictive power is limited) — a finding consistent with how the insurance industry actually operates, leaning on claims history and experience rating rather than pure demographics.

## Live App
**[Pet Insurance Cost Simulator](https://pet-insurance-cost-simulator.streamlit.app/)** — enter a pet's species, breed, and age, and see a real Monte Carlo-simulated first-year cost range: probability of any claim, typical/worst-case/extreme-worst-case cost scenarios, and how that range shifts across the pet's lifetime.

## The Data
* **Source:** `PetData.csv` (50,000 pets, enrolled 2018) and `ClaimData.csv` (210,235 real claims, 2018-2020), obtained from a public GitHub repository ([stevenrhart/predicting-claims](https://github.com/stevenrhart/predicting-claims)), itself sourced from an unnamed national US pet insurance provider.
* **Provenance caveat:** the original source names no specific insurer and states no explicit data license for the raw files. Used here for a personal portfolio project on that basis; not represented as a licensed commercial dataset.
* **Species:** 41,976 dogs / 8,024 cats (5:1 ratio). **377 distinct breed strings**, following a power-law distribution (top 10 breeds cover 44.9% of all pets; top 30 cover 67.2%).
* **Claims:** 67.8% of pets filed at least one claim across the full 3-year window; 43.2% filed a claim within their first policy year specifically. Claim amounts are heavily right-skewed (median $199 per claim; mean $476.87; max $45,084.99).

## Breed Predisposition Feature Engineering
A lookup table (`breed_predisposition_lookup.py`) maps real breed strings to nine disease-predisposition flags (cancer, orthopedic, patellar luxation, cardiac, brachycephalic airway syndrome, neurological, drug sensitivity, eye disease, metabolic), built from real, individually cited veterinary sources — not general "breed is prone to X" folk knowledge. Coverage prioritized breeds by actual frequency in this dataset (the top ~13 dog breeds, covering ~51% of all dogs), rather than attempting all 377 strings.

**Every flag traces to a specific, checkable source**, for example:
- Golden Retriever — cancer risk (60-65% lifetime cancer mortality, Morris Animal Foundation's Golden Retriever Lifetime Study, 3,044+ dogs)
- German Shepherd — degenerative myelopathy (specific `SOD1:c.118G>A` gene mutation, confirmed via UK/Japan referral population studies)
- Cavalier King Charles Spaniel — mitral valve disease (100% prevalence by age 8+, echocardiographically confirmed) and syringomyelia (25-70% prevalence depending on age)
- Australian Shepherd — MDR1/ABCB1 drug sensitivity (~50% carrier frequency, UC Davis Veterinary Genetics Laboratory)
- Labrador Retriever — obesity linked to a specific POMC gene mutation (~25% carrier rate), plus a distinct Exercise-Induced Collapse gene (DNM1)

Unresearched breeds are explicitly flagged `NaN` (not `False`) via a `research_coverage` column, so "no known predisposition" is never confused with "we didn't look into this breed." Generic mixed-breed size categories (22% of the dataset) are handled as their own rule-based category — real veterinary literature supports *lower* predisposition risk for mixed breeds via hybrid vigor, not an unknown one. Cat breed categories (Domestic Shorthair, etc.) are intentionally left breed-agnostic — these are generic non-pedigreed population categories, and the real feline risk driver in this domain is age, not breed genetics.

**Coverage achieved: 49.4% of the full dataset** carries a real, sourced predisposition profile.

## Finding 1: Predisposition Flags Have Real, But Limited, Predictive Power for Individual Claims

A hurdle model (classifier for "will this pet claim," regressor for "how much, given a claim") was tested against two targets:

| Target | Classifier AUC | Predisposition flags' combined feature importance | Age + species combined importance |
|---|---|---|---|
| First-year claims | 0.583 | ~8% | ~83% |
| Lifetime (3-year) claims | 0.574 | ~14.6% | ~79% |

**Honest interpretation:** predisposition flags' relative contribution nearly doubled moving from a first-year to a lifetime target — directionally consistent with the underlying clinical hypothesis, since most flagged conditions (cancer, hip dysplasia, mitral valve disease) are late-onset and chronic, and wouldn't be expected to generate claims within a pet's *first* policy year specifically. But overall model accuracy barely moved either way (AUC 0.57-0.58, both effectively weak). **Demographic and predisposition data alone have a real, fundamental ceiling for predicting any one specific pet's claims** — most of what drives whether an individual pet claims (an injury, an infection, individual variation) simply isn't knowable from static enrollment-time data. This mirrors why real insurers lean heavily on claims history and experience rating for renewal pricing rather than demographics alone.

## Model Correction Note
An earlier version of the classifier (`class_weight='balanced'`, uncalibrated) produced `predict_proba()` output that overstated the true claims rate by a factor of **1.155x** (mean predicted probability 49.8% vs. true base rate 43.1%) — the same class of calibration issue found and corrected earlier in this portfolio's [animal shelter recidivism model](https://github.com/MLuftig/animal-shelter-recidivism-prediction). This directly propagated into the Monte Carlo simulation below, initially overstating total portfolio cost by the same ~15.5%. Corrected via `CalibratedClassifierCV` with isotonic regression, calibrated on a held-out split — bringing predicted probability to within 0.01 percentage points of the true base rate.

## Finding 2: The Same Model Performs Well at the Portfolio Level
Individual-pet prediction has a real ceiling, but a Monte Carlo simulation doesn't need individual-level accuracy to be useful — errors partially cancel out across a large portfolio (law of large numbers), and the actual business question ("what's our expected total payout, and what's our tail risk?") is different from "can I predict this one pet."

**Method:** for each of the 50,000 pets, draw whether they claim (calibrated probability) and, if so, bootstrap a real observed claim amount from the empirical severity distribution (not a parametric fit — see Limitations). Repeated 5,000 times to build a full distribution of possible portfolio-wide annual costs.

**Validation:** simulated mean total cost landed within **0.04%** of the real observed historical total ($40,498,629 simulated vs. $40,515,072 actual).

**Output, for a 50,000-pet portfolio:**
| Percentile | Total Annual Cost |
|---|---|
| 5th | $39,717,055 |
| 50th (median) | $40,500,064 |
| 95th | $41,318,808 |
| 99th ("bad year") | $41,642,553 |

This is the actual deliverable of a reserving simulation — not a single forecast, but a defensible range for capital reserving and stress-test scenarios.

## Finding 3: At the Individual Level, "$0" Is Often the Correct Answer — And That's the Point
The same calibrated model and Monte Carlo method were repurposed to simulate an *individual* pet's plausible first-year cost range (the deployed app, above). For most pet profiles, the median simulated outcome is genuinely **$0** — not a modeling failure, but the correct reflection of how insurance risk actually works: protection against a comparatively rare but potentially expensive event, not a routine payout.

This connects directly back to Finding 1 rather than sitting apart from it: the same underlying fact — that a given pet, in a given year, most likely files no claim at all — is *why* individual-level demographic prediction has a real accuracy ceiling, and *why* risk pooling (Finding 2's portfolio simulation) is the tool that actually works well for this kind of risk. The app makes this explicit and self-explanatory: a `$0` median is paired with a real, quantified tail (e.g., a 3-year-old Golden Retriever: $0 median, but a 95th-percentile cost of $3,569 and a 99th-percentile cost of $10,825) rather than presented as an unexplained or discouraging number. The app's median outcome only turns positive for higher-risk profiles where a claim genuinely becomes more likely than not (e.g., an 8-year-old Cavalier King Charles Spaniel — four active predisposition flags plus age — shows a $140 positive median, crossing the >50%-claim-probability threshold).

## Technical Note: Handling "Unknown" Categorical Inputs Correctly
The deployed app lets a user select an unknown enrollment channel rather than forcing a guess. This required care: the model's one-hot encoding (`drop_first=True`) has no genuine "blank" state — setting both `EnrollPath_Phone` and `EnrollPath_Web` to 0 doesn't represent ignorance, it silently and specifically represents the third real category (`EB`, "Employee Benefit" channel). A naive "Unknown" option would have quietly misrepresented every ambiguous input as a specific answer. The correct fix, requiring no retraining: **marginalize the prediction across the real observed channel distribution** (Web 51.0% / Phone 47.1% / EB 2.0% of real enrollments), producing a genuine probability-weighted average rather than a disguised, incorrect default.

## Limitations
- **Claim-cause attribution is inferred, not observed.** `ClaimData.csv` has no diagnosis/condition field — there is no way to link a specific claim to a specific predisposed condition. Predisposition flags are used as population-level risk-profile features, tested for statistical association with claims behavior, not as individual disease attribution.
- **Claim severity is modeled via empirical bootstrap resampling, not a parametric distribution.** A lognormal fit was tested first and matches the real data well through the 90th percentile, but underestimates the true 99th-percentile tail by ~25% ($3,666.87 fitted vs. $4,916.82 real) — a well-documented "fat tail" pattern in real insurance claims data. Empirical resampling was used instead specifically to avoid smoothing over this real tail risk.
- **`PetAge` is a coarse categorical bucket, not a continuous value.** One single bucket (`"8 weeks to 12 months old"`) covers 62.75% of the entire dataset, meaningfully limiting age's true discriminating power despite its high measured feature importance.
- **Breed predisposition coverage is real but partial** (49.4% of the dataset overall; 11 individual dog breeds directly available in the deployed app) — prioritized by actual frequency in this data, not exhaustive across all 377 breed strings. Full source citations available on request.
- **Data provenance is not fully verifiable** — see Data section above.

## Tech Stack
`Python`, `pandas`, `NumPy`, `scikit-learn` (Random Forest classification/regression, isotonic calibration), `SciPy` (distribution fitting/testing), `Streamlit` (deployment), `Matplotlib` (visualization)

## Repository Structure
```text
├── data/
│   ├── PetData.csv
│   ├── ClaimData.csv
│   └── breed_predisposition_lookup.py         # Real, cited breed-to-condition mapping
├── models/
│   ├── pet_insurance_claims_classifier.pkl     # Calibrated Random Forest classifier
│   └── real_severity_pool.npy                  # Empirical claim-severity data for bootstrap sampling
├── src/
│   ├── 01-breed-predisposition-lookup.ipynb
│   ├── 02-hurdle-model-year1-vs-lifetime.ipynb
│   └── 03-monte-carlo-portfolio-simulation.ipynb
├── app.py                                       # Deployed individual-pet cost simulator (Streamlit)
├── requirements.txt
└── README.md
```

## Related Projects
This project applies the same core techniques used throughout this portfolio's animal shelter work — Monte Carlo simulation ([Shelter Overflow Risk Forecaster](https://github.com/MLuftig/shelter-overflow-forecaster), [Shelter Medical Supply Forecaster](https://github.com/MLuftig/shelter-supply-forecaster)), probability calibration correction ([Animal Shelter Recidivism Prediction](https://github.com/MLuftig/animal-shelter-recidivism-prediction)), and the same real-data-verification discipline throughout — applied here to a different domain (insurance pricing/reserving) to demonstrate the same skill set is transferable, not shelter-specific.

## Getting Started & Installation
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
```
