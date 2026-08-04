# Pet Insurance Cost Simulator

An interactive Streamlit app that estimates a real, Monte Carlo-simulated range of plausible first-year veterinary insurance costs for an individual pet — not a single point estimate, but a genuine best-case-to-worst-case picture grounded in real historical claims data.

**Live App:** [pet-insurance-cost-simulator.streamlit.app](https://pet-insurance-cost-simulator.streamlit.app/)

**Full methodology, model validation, and findings:** [Pet Insurance Risk & Pricing Analysis](https://github.com/MLuftig/pet-insurance-risk-and-pricing-analysis) — this repo is the deployed app only; the underlying research (breed predisposition sourcing, model calibration, portfolio validation, and the pricing fairness audit) lives there.

## What It Does

Enter a pet's species, breed, and age, and the app:
- Predicts the probability of a first-year insurance claim using a calibrated Random Forest model
- Runs a Monte Carlo simulation (using real, empirical historical claim amounts, not an assumed distribution) to generate a genuine cost-range forecast: typical, worst-case (95th percentile), and extreme-worst-case (99th percentile) outcomes
- Shows how that cost range shifts across the pet's full lifetime (0-15 years), not just at the selected age
- Displays the specific, real, cited veterinary predisposition flags behind the prediction for the selected breed (e.g., a Cavalier King Charles Spaniel shows cardiac, neurological, brachycephalic, and patellar luxation risk — each traceable to a real source)

## A Note on the "$0" Result
For most pet profiles, the single most likely first-year outcome the app shows is genuinely **$0**. That isn't a weak or broken result — it's the correct reflection of how insurance risk actually works: protection against a comparatively rare but potentially expensive event, not a routine payout. The app pairs that $0 median with the real, quantified tail (the worst-case and extreme-worst-case numbers) so the actual value of coverage is visible alongside it, not hidden by it.

## Handling Unknown Inputs Correctly
The "Enrollment Channel" field includes an explicit "Unknown" option. Because the underlying model's categorical encoding has no genuine blank/null state, selecting "Unknown" doesn't guess a default — it marginalizes the prediction across the real observed distribution of enrollment channels (Web, Phone, and Employee Benefit), producing a genuine probability-weighted estimate rather than a silently incorrect one.

## Running Locally
```bash
git clone https://github.com/MLuftig/pet_insurance_cost_simulator.git
cd pet_insurance_cost_simulator
pip install -r requirements.txt
streamlit run app.py
```

## Tech Stack
`Python`, `Streamlit`, `Scikit-Learn` (calibrated Random Forest classifier), `Joblib` (model persistence), `Matplotlib` (visualization), `NumPy`/`pandas`

## Repository Structure
```text
├── app.py                                     # Streamlit application
├── pet_insurance_claims_classifier.pkl         # Pre-trained, calibrated Random Forest classifier
├── real_severity_pool.npy                      # Empirical claim-severity data for Monte Carlo bootstrap sampling
├── requirements.txt
└── README.md
```

Both the classifier and severity pool are produced and validated in the companion [analysis repository](https://github.com/MLuftig/pet-insurance-risk-and-pricing-analysis) — see that repo for the calibration methodology, portfolio-level validation (accurate to within 0.04% of real historical totals), and the full breed predisposition research with citations.
