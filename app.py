import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Pet Insurance Cost Simulator", page_icon="🐾", layout="centered")

# ============================================================
# Load validated artifacts
# ============================================================
@st.cache_resource
def load_artifacts():
    clf = joblib.load("pet_insurance_claims_classifier.pkl")
    severity_pool = np.load("real_severity_pool.npy")
    return clf, severity_pool

clf, severity_pool = load_artifacts()

FEATURE_COLS = ['age_years', 'risk_cancer', 'risk_orthopedic', 'risk_patellar_luxation',
                'risk_cardiac', 'risk_brachycephalic', 'risk_neurological',
                'risk_drug_sensitivity', 'risk_eye_disease', 'risk_metabolic',
                'research_coverage', 'Species_Dog', 'EnrollPath_Phone', 'EnrollPath_Web']

FLAG_COLS = ['risk_cancer', 'risk_orthopedic', 'risk_patellar_luxation', 'risk_cardiac',
             'risk_brachycephalic', 'risk_neurological', 'risk_drug_sensitivity',
             'risk_eye_disease', 'risk_metabolic']

FLAG_LABELS = {
    'risk_cancer': 'Cancer', 'risk_orthopedic': 'Hip/Elbow Dysplasia',
    'risk_patellar_luxation': 'Patellar Luxation', 'risk_cardiac': 'Cardiac Disease',
    'risk_brachycephalic': 'Brachycephalic Airway Syndrome', 'risk_neurological': 'Neurological Disease',
    'risk_drug_sensitivity': 'Multidrug Sensitivity (MDR1)', 'risk_eye_disease': 'Hereditary Eye Disease',
    'risk_metabolic': 'Obesity/Metabolic Predisposition',
}

# Real, sourced predisposition flags -- see breed_predisposition_lookup.py
# for full citations. Breeds not listed here are treated as unresearched
# (all flags null, research_coverage=False), matching the source table
# exactly -- no guessing is introduced at the app layer.
BREED_PREDISPOSITIONS = {
    'Golden Retriever': {'risk_cancer': 1, 'risk_orthopedic': 1, 'risk_patellar_luxation': 0, 'risk_cardiac': 0, 'risk_brachycephalic': 0, 'risk_neurological': 0, 'risk_drug_sensitivity': 0, 'risk_eye_disease': 0, 'risk_metabolic': 0},
    'German Shepherd': {'risk_cancer': 1, 'risk_orthopedic': 1, 'risk_patellar_luxation': 0, 'risk_cardiac': 0, 'risk_brachycephalic': 0, 'risk_neurological': 1, 'risk_drug_sensitivity': 0, 'risk_eye_disease': 0, 'risk_metabolic': 0},
    'French Bulldog': {'risk_cancer': 0, 'risk_orthopedic': 0, 'risk_patellar_luxation': 1, 'risk_cardiac': 0, 'risk_brachycephalic': 1, 'risk_neurological': 0, 'risk_drug_sensitivity': 0, 'risk_eye_disease': 0, 'risk_metabolic': 0},
    'English Bulldog': {'risk_cancer': 0, 'risk_orthopedic': 0, 'risk_patellar_luxation': 1, 'risk_cardiac': 0, 'risk_brachycephalic': 1, 'risk_neurological': 0, 'risk_drug_sensitivity': 0, 'risk_eye_disease': 0, 'risk_metabolic': 0},
    'Boston Terrier': {'risk_cancer': 0, 'risk_orthopedic': 0, 'risk_patellar_luxation': 1, 'risk_cardiac': 0, 'risk_brachycephalic': 1, 'risk_neurological': 0, 'risk_drug_sensitivity': 0, 'risk_eye_disease': 0, 'risk_metabolic': 0},
    'Yorkshire Terrier': {'risk_cancer': 0, 'risk_orthopedic': 0, 'risk_patellar_luxation': 1, 'risk_cardiac': 0, 'risk_brachycephalic': 0, 'risk_neurological': 0, 'risk_drug_sensitivity': 0, 'risk_eye_disease': 0, 'risk_metabolic': 0},
    'Chihuahua': {'risk_cancer': 0, 'risk_orthopedic': 0, 'risk_patellar_luxation': 1, 'risk_cardiac': 0, 'risk_brachycephalic': 1, 'risk_neurological': 0, 'risk_drug_sensitivity': 0, 'risk_eye_disease': 0, 'risk_metabolic': 0},
    'Cavalier King Charles Spaniel': {'risk_cancer': 0, 'risk_orthopedic': 0, 'risk_patellar_luxation': 1, 'risk_cardiac': 1, 'risk_brachycephalic': 1, 'risk_neurological': 1, 'risk_drug_sensitivity': 0, 'risk_eye_disease': 0, 'risk_metabolic': 0},
    'Labrador Retriever': {'risk_cancer': 0, 'risk_orthopedic': 1, 'risk_patellar_luxation': 0, 'risk_cardiac': 0, 'risk_brachycephalic': 0, 'risk_neurological': 0, 'risk_drug_sensitivity': 0, 'risk_eye_disease': 0, 'risk_metabolic': 1},
    'Australian Shepherd': {'risk_cancer': 0, 'risk_orthopedic': 0, 'risk_patellar_luxation': 0, 'risk_cardiac': 0, 'risk_brachycephalic': 0, 'risk_neurological': 0, 'risk_drug_sensitivity': 1, 'risk_eye_disease': 0, 'risk_metabolic': 0},
    'Siberian Husky': {'risk_cancer': 0, 'risk_orthopedic': 0, 'risk_patellar_luxation': 0, 'risk_cardiac': 0, 'risk_brachycephalic': 0, 'risk_neurological': 0, 'risk_drug_sensitivity': 0, 'risk_eye_disease': 1, 'risk_metabolic': 0},
    'Shih Tzu': {'risk_cancer': 0, 'risk_orthopedic': 0, 'risk_patellar_luxation': 1, 'risk_cardiac': 0, 'risk_brachycephalic': 1, 'risk_neurological': 0, 'risk_drug_sensitivity': 0, 'risk_eye_disease': 0, 'risk_metabolic': 0},
}


# Real observed enrollment-channel frequencies in the training data (Web:
# 25,486 / Phone: 23,525 / EB: 989 of 50,000 pets) -- used to marginalize
# over the real channel distribution when the user doesn't know/specify
# their channel, rather than silently defaulting to one specific category.
# NOTE: one-hot encoding with drop_first=True has no true "blank" state --
# EnrollPath_Phone=0, EnrollPath_Web=0 already means "EB" to the model, not
# "unknown." A genuine unknown has to be handled by averaging predictions
# across the real categories, not by zeroing out the dummy flags.
_ENROLLPATH_COUNTS = {'Web': 25486, 'Phone': 23525, 'EB': 989}
_ENROLLPATH_TOTAL = sum(_ENROLLPATH_COUNTS.values())
ENROLLPATH_WEIGHTS = {k: v / _ENROLLPATH_TOTAL for k, v in _ENROLLPATH_COUNTS.items()}


def build_feature_row(species, breed, age_years, enroll_path):
    is_researched = breed in BREED_PREDISPOSITIONS
    flags = BREED_PREDISPOSITIONS.get(breed, {k: 0 for k in FLAG_COLS})
    row = {'age_years': age_years}
    row.update(flags)
    row['research_coverage'] = int(is_researched)
    row['Species_Dog'] = int(species == 'Dog')
    row['EnrollPath_Phone'] = int(enroll_path == 'Phone')
    row['EnrollPath_Web'] = int(enroll_path == 'Web')
    return pd.DataFrame([row], columns=FEATURE_COLS), is_researched, flags


def predict_claim_prob(species, breed, age_years, enroll_path):
    """Handles 'Unknown' by marginalizing over the real observed channel
    distribution (weighted average across Web/Phone/EB), rather than
    guessing a single specific channel."""
    if enroll_path == 'Unknown':
        weighted_prob = 0.0
        flags_out, researched_out = None, None
        for channel, weight in ENROLLPATH_WEIGHTS.items():
            X, is_researched, flags = build_feature_row(species, breed, age_years, channel)
            weighted_prob += weight * clf.predict_proba(X)[0][1]
            flags_out, researched_out = flags, is_researched  # same regardless of channel
        return weighted_prob, researched_out, flags_out
    else:
        X, is_researched, flags = build_feature_row(species, breed, age_years, enroll_path)
        return clf.predict_proba(X)[0][1], is_researched, flags


def simulate_individual_pet(claim_prob, n_sims, seed=None):
    rng = np.random.default_rng(seed)
    claims_occur = rng.random(n_sims) < claim_prob
    outcomes = np.zeros(n_sims)
    n_claims = claims_occur.sum()
    if n_claims > 0:
        outcomes[claims_occur] = rng.choice(severity_pool, size=n_claims, replace=True)
    return outcomes


def build_age_range_chart(species, breed, enroll_path, age_sims=3000):
    """Sweeps age 0-15 for the given species/breed, returning the 5th-95th
    percentile cost range plus median at each age -- shows how the plausible
    cost band shifts as a pet ages, not just a single age's outcome."""
    ages = np.arange(0, 15.5, 1.0)
    p5s, medians, p95s = [], [], []
    for age in ages:
        prob, _, _ = predict_claim_prob(species, breed, age, enroll_path)
        outcomes = simulate_individual_pet(prob, age_sims, seed=42)
        p5, med, p95 = np.percentile(outcomes, [5, 50, 95])
        p5s.append(p5); medians.append(med); p95s.append(p95)
    return ages, np.array(p5s), np.array(medians), np.array(p95s)


# ============================================================
# UI
# ============================================================
st.title("🐾 Pet Insurance Cost Simulator")
st.markdown(
    "Estimates a **range of plausible first-year veterinary claim costs** for an "
    "individual pet, using a calibrated Random Forest model and Monte Carlo "
    "simulation grounded in real historical claims data (50,000 real pets, "
    "210,000+ real claims)."
)
st.info(
    "⚠️ This estimates one *individual* pet's outcome range, not a portfolio "
    "average -- most simulated years show no claim at all, with a smaller "
    "chance of a real, sometimes large, veterinary expense. That's an honest "
    "reflection of how individual insurance risk actually works, not a flaw "
    "in the model."
)

st.divider()

col1, col2 = st.columns(2)
with col1:
    species = st.selectbox("Species", ["Dog", "Cat"])
    age_years = st.slider("Age (years)", 0.0, 15.0, 3.0, step=0.5)
with col2:
    if species == "Dog":
        breed_options = list(BREED_PREDISPOSITIONS.keys()) + ["Other / Mixed Breed"]
    else:
        breed_options = ["Other / Mixed Breed"]  # cat predisposition is age-driven, not breed-driven -- see README
    breed = st.selectbox("Breed", breed_options)
    enroll_path = st.selectbox("Enrollment Channel", ["Unknown", "Web", "Phone", "EB"])
    st.caption("EB = Employee Benefit (workplace enrollment)")

n_sims = st.slider("Number of simulations", 500, 10000, 5000, step=500)

if st.button("Run Simulation", type="primary"):
    claim_prob, is_researched, flags = predict_claim_prob(species, breed, age_years, enroll_path)

    with st.spinner("Running Monte Carlo simulation..."):
        outcomes = simulate_individual_pet(claim_prob, n_sims)

    st.divider()
    st.subheader("Results")

    m1, m2, m3 = st.columns(3)
    m1.metric("Probability of a claim (year 1)", f"{claim_prob:.1%}")
    m2.metric("No-claim probability", f"{(outcomes == 0).mean():.1%}")
    m3.metric("Median outcome", f"${np.median(outcomes):,.0f}")

    median_outcome = np.median(outcomes)
    if median_outcome == 0:
        st.caption(
            "💡 A $0 median means this profile's single most likely outcome "
            "in any given year is **no claim at all** -- this isn't a "
            "limitation of the model, it's the actual nature of insurance "
            "risk. The real value shows up in the tail below: a smaller "
            "chance of a real, sometimes large, expense."
        )
    else:
        st.caption(
            "💡 This profile's median outcome is **above $0** -- meaning a "
            "claim is more likely than not in a given year for this "
            "specific pet, not just a tail-risk possibility. Worth pricing "
            "and reserving for accordingly."
        )

    with st.expander("How to read these numbers"):
        st.markdown(
            """
            - **Probability of a claim** — the model's estimate of how likely
              *any* claim is for this pet in its first year, based on
              species, age, and breed predisposition.
            - **Median outcome** — the single most likely result across
              thousands of simulated years. Often $0 (see the note above,
              specific to this result) -- that's expected, not an error.
            - **75th / 95th / 99th percentile** — how bad a "somewhat
              unlucky," "worst case," and "extreme worst case" year could
              realistically look, based on real historical claim amounts,
              not a guess.
            - **Median cost, given a claim happens** — a separate number
              from the overall median: if a claim *does* occur, this is the
              typical size of that specific claim.
            """
        )

    st.markdown("**Cost range across simulated years:**")
    p50, p75, p95, p99 = np.percentile(outcomes, [50, 75, 95, 99])
    range_df = pd.DataFrame({
        "Scenario": ["Typical (median)", "75th percentile", "Worst case (95th percentile)", "Extreme worst case (99th percentile)"],
        "Estimated Cost": [f"${p50:,.0f}", f"${p75:,.0f}", f"${p95:,.0f}", f"${p99:,.0f}"],
    })
    st.table(range_df.set_index("Scenario"))

    if (outcomes > 0).any():
        st.caption(f"Median cost *given a claim actually happens*: ${np.median(outcomes[outcomes > 0]):,.2f}")

    st.divider()
    st.subheader("Cost Range by Age")
    st.caption(
        f"How the plausible cost range shifts across a {breed}'s lifetime "
        "(5th-95th percentile band, shaded). The current age is marked with "
        "a vertical line."
    )
    with st.spinner("Building age range chart..."):
        ages, p5s, medians, p95s = build_age_range_chart(species, breed, enroll_path)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.fill_between(ages, p5s, p95s, alpha=0.25, color="#FF6B6B", label="5th-95th percentile range")
    ax.plot(ages, medians, color="#FF6B6B", linewidth=2, label="Median outcome")
    ax.axvline(age_years, color="gray", linestyle="--", linewidth=1, label=f"Selected age ({age_years})")
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Estimated First-Year Claim Cost ($)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    st.pyplot(fig)

    st.divider()
    if is_researched:
        active_flags = [FLAG_LABELS[k] for k, v in flags.items() if v == 1]
        if active_flags:
            st.markdown(f"**Known predisposition(s) for {breed}:** " + ", ".join(active_flags))
        else:
            st.markdown(f"**{breed}** has been individually researched; no major predisposition flags were found in the sources reviewed.")
    else:
        st.markdown(
            f"**{breed}** has not been individually researched for this tool -- "
            "predisposition flags are not applied, and predictions rely on species/age alone."
        )

st.divider()
with st.expander("About this model"):
    st.markdown(
        """
        This tool combines two pieces of analysis:

        1. **A calibrated Random Forest classifier** predicting the probability
           of a first-year insurance claim, trained on 50,000 real pet
           insurance enrollments. Age and species are the dominant predictors;
           breed-specific disease predisposition (drawn from real, cited
           veterinary research) adds real but modest additional signal --
           individual-level prediction has a real ceiling, since most of what
           determines whether *this specific* pet claims (an injury, an
           acute illness) isn't knowable from static enrollment-time data.
        2. **Monte Carlo simulation** using the calibrated probability plus
           real historical claim amounts (not an assumed distribution) to
           show a genuine range of plausible outcomes, not a single guess.

        **Note:** this individual-pet view is intentionally different from a
        portfolio-level forecast. At the portfolio level (many pets averaged
        together), this same model's total-cost predictions are validated to
        within 0.04% of real historical totals -- but any *one* pet's outcome
        is inherently much less predictable, which is why the results above
        show a real range rather than a single confident number.
        """
    )
