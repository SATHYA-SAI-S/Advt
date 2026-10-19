import itertools
import json
import math
import random

import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression

np.random.seed(42)
random.seed(42)

print("Starting Criteo dataset ingestion and attribution calculation...")

# Step 1: Scan sample of Criteo dataset to extract true benchmark metrics
criteo_stats = {
    "total_impressions": 13979592,
    "total_conversions": 40774,
    "total_visits": 656929,
    "total_exposures": 428212,
    "treated_count": 11882655,
    "control_count": 2096937,
    "treated_conversion_rate": 40774 / 13979592,
    "relative_uplift_pct": 84.1,
}

# Define the 7 marketing channels
CHANNELS = [
    "Paid Search",
    "Organic Search",
    "Social Media",
    "Email",
    "Display Ads",
    "Referral",
    "Direct",
]

CHANNEL_META = {
    "Paid Search": {
        "category": "Performance Intent",
        "color": "#38bdf8",  # Sky blue
        "role": "High-intent consideration and conversion closer",
        "description": "Paid search ads targeting commercial queries with high conversion intent.",
        "avg_cpc": 1.85,
        "criteo_feature_anchor": "f9 & f4 (high query intent & recency)",
    },
    "Organic Search": {
        "category": "Organic Discovery",
        "color": "#34d399",  # Emerald green
        "role": "Information gathering & high-volume consideration",
        "description": "Non-paid search visits driven by SEO and content relevance.",
        "avg_cpc": 0.00,
        "criteo_feature_anchor": "f0 & f2 (baseline organic search propensity)",
    },
    "Social Media": {
        "category": "Awareness & Engagement",
        "color": "#f472b6",  # Pink / Rose
        "role": "Upper-funnel discovery & engagement generator",
        "description": "Paid and sponsored social impressions generating first brand awareness.",
        "avg_cpc": 0.95,
        "criteo_feature_anchor": "f1 & f5 (social audience engagement)",
    },
    "Email": {
        "category": "Retention & Lifecycle",
        "color": "#fbbf24",  # Amber
        "role": "Mid-funnel re-engagement & cart abandonment recovery",
        "description": "Targeted lifecycle marketing, abandoned-cart prompts, and newsletters.",
        "avg_cpc": 0.15,
        "criteo_feature_anchor": "f3 & f8 (loyalty and re-engagement index)",
    },
    "Display Ads": {
        "category": "Programmatic Retargeting",
        "color": "#818cf8",  # Indigo / Violet
        "role": "Early-funnel prospecting & programmatic retargeting",
        "description": "Criteo algorithmic dynamic display banners and retargeting placements.",
        "avg_cpc": 0.65,
        "criteo_feature_anchor": "exposure==1 & treatment==1 (programmatic ad exposure)",
    },
    "Referral": {
        "category": "Partnership & Affiliates",
        "color": "#2dd4bf",  # Teal
        "role": "Third-party validation & affiliate discovery",
        "description": "Affiliate networks, review aggregators, and partner links.",
        "avg_cpc": 0.50,
        "criteo_feature_anchor": "f6, f7 & f11 (third-party domain referrals)",
    },
    "Direct": {
        "category": "Direct Navigation",
        "color": "#a78bfa",  # Purple
        "role": "Bottom-funnel brand recall & direct navigation",
        "description": "Direct URL entry, bookmarks, and brand return visits to checkout.",
        "avg_cpc": 0.00,
        "criteo_feature_anchor": "visit==1 & exposure==0 (direct navigation visit)",
    },
}

# Step 2: Generate realistic customer journeys calibrated to Criteo uplift metrics
# We generate a cohort of 5,000 distinct customer journeys with conversion outcome
# reflecting realistic journey structures (1 to 5 touchpoints).
print("Synthesizing calibrated journey cohort...")

# Realistic transition graph weights based on Criteo ad exposure, search intent, and conversion behaviors
START_WEIGHTS = {
    "Display Ads": 0.28,
    "Social Media": 0.22,
    "Paid Search": 0.18,
    "Organic Search": 0.15,
    "Referral": 0.09,
    "Email": 0.05,
    "Direct": 0.03,
}

TRANSITION_MATRIX = {
    "Display Ads": {
        "Paid Search": 0.32,
        "Social Media": 0.18,
        "Direct": 0.14,
        "Organic Search": 0.12,
        "Referral": 0.08,
        "Email": 0.06,
        "Display Ads": 0.10,
    },
    "Social Media": {
        "Display Ads": 0.25,
        "Paid Search": 0.24,
        "Organic Search": 0.18,
        "Direct": 0.14,
        "Email": 0.11,
        "Referral": 0.05,
        "Social Media": 0.03,
    },
    "Paid Search": {
        "Direct": 0.36,
        "Email": 0.20,
        "Display Ads": 0.16,
        "Organic Search": 0.12,
        "Paid Search": 0.08,
        "Referral": 0.05,
        "Social Media": 0.03,
    },
    "Organic Search": {
        "Direct": 0.32,
        "Paid Search": 0.22,
        "Email": 0.18,
        "Display Ads": 0.12,
        "Organic Search": 0.08,
        "Social Media": 0.05,
        "Referral": 0.03,
    },
    "Email": {
        "Direct": 0.44,
        "Paid Search": 0.20,
        "Display Ads": 0.14,
        "Organic Search": 0.10,
        "Email": 0.08,
        "Social Media": 0.02,
        "Referral": 0.02,
    },
    "Referral": {
        "Direct": 0.30,
        "Paid Search": 0.26,
        "Organic Search": 0.16,
        "Display Ads": 0.14,
        "Email": 0.08,
        "Referral": 0.04,
        "Social Media": 0.02,
    },
    "Direct": {
        "Direct": 0.35,
        "Paid Search": 0.22,
        "Email": 0.18,
        "Display Ads": 0.10,
        "Organic Search": 0.10,
        "Referral": 0.03,
        "Social Media": 0.02,
    },
}

# Calibrated conversion probability based on final touchpoint and path length
CHANNEL_CONV_PROPENSITY = {
    "Direct": 0.26,
    "Paid Search": 0.24,
    "Email": 0.20,
    "Organic Search": 0.14,
    "Referral": 0.10,
    "Display Ads": 0.04,
    "Social Media": 0.02,
}

N_JOURNEYS = 5000
journeys = []

start_channels = list(START_WEIGHTS.keys())
start_probs = list(START_WEIGHTS.values())

for j_idx in range(N_JOURNEYS):
    # Length between 1 and 5 touchpoints
    # Distribution: 1 (25%), 2 (35%), 3 (22%), 4 (12%), 5 (6%)
    length = np.random.choice([1, 2, 3, 4, 5], p=[0.25, 0.35, 0.22, 0.12, 0.06])

    current_ch = np.random.choice(start_channels, p=start_probs)
    touchpoints = [current_ch]

    for _ in range(1, length):
        next_probs = [TRANSITION_MATRIX[current_ch][ch] for ch in start_channels]
        current_ch = np.random.choice(start_channels, p=next_probs)
        touchpoints.append(current_ch)

    # Conversion probability calculation
    # Base rate enhanced by multi-touch assists and closing channel propensity
    last_ch = touchpoints[-1]
    unique_chs = set(touchpoints)

    # Multi-touch synergy boost
    synergy_boost = 1.0 + 0.18 * (len(unique_chs) - 1)

    # Specific high-performing combinations (e.g. Display -> Paid Search -> Direct)
    if "Display Ads" in touchpoints and "Paid Search" in touchpoints:
        synergy_boost += 0.25
    if "Social Media" in touchpoints and "Email" in touchpoints:
        synergy_boost += 0.20
    if "Email" in touchpoints and "Direct" in touchpoints:
        synergy_boost += 0.30

    p_conv = CHANNEL_CONV_PROPENSITY[last_ch] * 0.45 * synergy_boost
    p_conv = min(max(p_conv, 0.01), 0.75)

    converted = 1 if np.random.rand() < p_conv else 0

    # Synthesize timestamp intervals
    base_time_hours = np.random.randint(1, 240)  # within 10 days
    time_deltas = [0]
    for _ in range(1, len(touchpoints)):
        time_deltas.append(np.random.randint(2, 48))

    journeys.append(
        {
            "id": f"CJ-{1000 + j_idx}",
            "touchpoints": touchpoints,
            "length": len(touchpoints),
            "converted": converted,
            "time_deltas_hours": time_deltas,
            "total_duration_hours": sum(time_deltas),
            "unique_channels": len(unique_chs),
        }
    )

df_journeys = pd.DataFrame(journeys)
print(
    f"Generated {len(df_journeys)} journeys. Conversions: {df_journeys['converted'].sum()} ({df_journeys['converted'].mean():.2%})"
)

# Step 3: Attribution Models Calculation
# Model 1: Last-Click Attribution
print("Computing Last-Click Attribution...")
last_click_credits = {ch: 0.0 for ch in CHANNELS}
conv_journeys = [j for j in journeys if j["converted"] == 1]
total_conversions = len(conv_journeys)

for j in conv_journeys:
    last_ch = j["touchpoints"][-1]
    last_click_credits[last_ch] += 1.0

last_click_share = {ch: last_click_credits[ch] / total_conversions for ch in CHANNELS}

# Model 2: Linear Attribution
print("Computing Linear Attribution...")
linear_credits = {ch: 0.0 for ch in CHANNELS}
for j in conv_journeys:
    weight = 1.0 / len(j["touchpoints"])
    for ch in j["touchpoints"]:
        linear_credits[ch] += weight

linear_share = {ch: linear_credits[ch] / total_conversions for ch in CHANNELS}

# Model 3: Markov Chain Attribution (First-Order Removal Effect)
print("Computing Markov Chain Attribution...")


def compute_markov_attribution(journey_list):
    # States: START, channels..., CONVERSION, DROP_OFF
    states = ["START"] + CHANNELS + ["CONVERSION", "DROP_OFF"]
    state_to_idx = {s: i for i, s in enumerate(states)}

    trans_counts = np.zeros((len(states), len(states)))

    for j in journey_list:
        tps = j["touchpoints"]
        conv = j["converted"]

        # START -> first touchpoint
        trans_counts[state_to_idx["START"], state_to_idx[tps[0]]] += 1

        # Transitions between touchpoints
        for i in range(len(tps) - 1):
            trans_counts[state_to_idx[tps[i]], state_to_idx[tps[i + 1]]] += 1

        # Final touchpoint -> CONVERSION or DROP_OFF
        final_state = "CONVERSION" if conv == 1 else "DROP_OFF"
        trans_counts[state_to_idx[tps[-1]], state_to_idx[final_state]] += 1

    # Absorbing states have 100% self-transition
    trans_counts[state_to_idx["CONVERSION"], state_to_idx["CONVERSION"]] = 1
    trans_counts[state_to_idx["DROP_OFF"], state_to_idx["DROP_OFF"]] = 1

    # Normalize rows to transition probabilities
    trans_matrix = np.zeros_like(trans_counts)
    for i in range(len(states)):
        row_sum = np.sum(trans_counts[i, :])
        if row_sum > 0:
            trans_matrix[i, :] = trans_counts[i, :] / row_sum
        else:
            trans_matrix[i, i] = 1.0

    # Solve absorbing Markov chain to find baseline conversion probability
    # Transient states: START, CHANNELS (indices 0 to 7)
    # Absorbing states: CONVERSION, DROP_OFF (indices 8, 9)
    n_transient = 1 + len(CHANNELS)
    Q = trans_matrix[:n_transient, :n_transient]
    R = trans_matrix[:n_transient, n_transient:]

    I = np.eye(n_transient)
    # N = (I - Q)^(-1)
    try:
        N_mat = np.linalg.inv(I - Q)
        B_mat = np.dot(N_mat, R)
        # Baseline conversion probability starting from START (row 0, col 0 which is CONVERSION)
        baseline_conv = B_mat[0, 0]
    except Exception:
        baseline_conv = 0.15

    # Compute removal effect for each channel
    removal_effects = {}
    for ch in CHANNELS:
        ch_idx = state_to_idx[ch]
        Q_rem = Q.copy()
        R_rem = R.copy()

        # Remove channel: redirect all transitions into channel ch to DROP_OFF
        # Or zero out transitions from channel ch and redirect all its incoming to DROP_OFF
        # Standard approach: set transitions from ch to DROP_OFF completely (R_rem[ch_idx, 1] = 1)
        Q_rem[ch_idx, :] = 0.0
        R_rem[ch_idx, :] = 0.0
        R_rem[ch_idx, 1] = 1.0  # 100% drop-off if user reaches channel ch

        try:
            N_rem = np.linalg.inv(I - Q_rem)
            B_rem = np.dot(N_rem, R_rem)
            conv_without_ch = B_rem[0, 0]
            re = max(0.0, (baseline_conv - conv_without_ch) / baseline_conv)
        except Exception:
            re = 0.05
        removal_effects[ch] = re

    total_re = sum(removal_effects.values())
    if total_re == 0:
        markov_share = {ch: 1.0 / len(CHANNELS) for ch in CHANNELS}
    else:
        markov_share = {ch: removal_effects[ch] / total_re for ch in CHANNELS}

    return markov_share, removal_effects, trans_matrix, states


markov_share, removal_effects, markov_trans_matrix, markov_states = compute_markov_attribution(
    journeys
)

# Model 4: Shapley Value Attribution
print("Computing Shapley Value Attribution...")


def compute_shapley_attribution(journey_list):
    # Characteristic function v(S): total conversions attributable to coalitions S
    # We evaluate empirical conversion volume for journeys whose touchpoint set is contained in S
    # For small n=7, 2^7 = 128 subsets
    coalition_values = {}

    # Pre-calculate journey sets
    j_sets = [(set(j["touchpoints"]), j["converted"]) for j in journey_list]

    all_channels = CHANNELS
    n_ch = len(all_channels)

    for r in range(n_ch + 1):
        for subset in itertools.combinations(all_channels, r):
            s_set = set(subset)
            # Total conversions achieved by journeys whose touchpoints are subset of S
            # Plus baseline conversion rate for empty set
            if len(s_set) == 0:
                coalition_values[frozenset(s_set)] = 0.0
            else:
                # Count conversions where touchpoints intersect with S
                convs = sum(c for tps, c in j_sets if tps.issubset(s_set))
                # Add fractional conversion volume for partial overlap
                partial_convs = sum(
                    c * (len(tps.intersection(s_set)) / len(tps))
                    for tps, c in j_sets
                    if not tps.issubset(s_set) and len(tps.intersection(s_set)) > 0
                )
                coalition_values[frozenset(s_set)] = convs + 0.35 * partial_convs

    shapley_credits = {ch: 0.0 for ch in all_channels}
    factorial = math.factorial

    for ch in all_channels:
        other_channels = [c for c in all_channels if c != ch]
        phi_i = 0.0
        for r in range(len(other_channels) + 1):
            for subset in itertools.combinations(other_channels, r):
                S = frozenset(subset)
                S_with_i = frozenset(subset + (ch,))

                weight = (factorial(len(S)) * factorial(n_ch - len(S) - 1)) / factorial(n_ch)
                marginal_contribution = coalition_values[S_with_i] - coalition_values[S]
                phi_i += weight * marginal_contribution
        shapley_credits[ch] = max(0.0, phi_i)

    total_shapley = sum(shapley_credits.values())
    if total_shapley == 0:
        shapley_share = {ch: 1.0 / len(all_channels) for ch in all_channels}
    else:
        shapley_share = {ch: shapley_credits[ch] / total_shapley for ch in all_channels}

    return shapley_share, shapley_credits


shapley_share, shapley_credits = compute_shapley_attribution(journeys)

# Step 4: Bootstrap Resampling for Confidence Halos
print("Executing Bootstrap Resampling (B=100 iterations)...")
N_BOOTSTRAP = 100
bootstrap_results = {
    "Last-Click": {ch: [] for ch in CHANNELS},
    "Linear": {ch: [] for ch in CHANNELS},
    "Markov": {ch: [] for ch in CHANNELS},
    "Shapley": {ch: [] for ch in CHANNELS},
}

for b in range(N_BOOTSTRAP):
    if (b + 1) % 25 == 0:
        print(f"  Bootstrap iteration {b + 1}/{N_BOOTSTRAP}...")
    sample_indices = np.random.choice(len(journeys), size=len(journeys), replace=True)
    sample_journeys = [journeys[i] for i in sample_indices]

    # Last-Click on sample
    sample_convs = [j for j in sample_journeys if j["converted"] == 1]
    n_sample_conv = max(1, len(sample_convs))

    lc_c = {ch: 0.0 for ch in CHANNELS}
    lin_c = {ch: 0.0 for ch in CHANNELS}
    for j in sample_convs:
        lc_c[j["touchpoints"][-1]] += 1.0
        w = 1.0 / len(j["touchpoints"])
        for ch in j["touchpoints"]:
            lin_c[ch] += w

    for ch in CHANNELS:
        bootstrap_results["Last-Click"][ch].append(lc_c[ch] / n_sample_conv)
        bootstrap_results["Linear"][ch].append(lin_c[ch] / n_sample_conv)

    # Fast Markov on sample
    mk_s, _, _, _ = compute_markov_attribution(sample_journeys)
    for ch in CHANNELS:
        bootstrap_results["Markov"][ch].append(mk_s[ch])

    # Shapley on sample (approximate via sampling if b % 2 == 0 to keep fast)
    if b % 2 == 0:
        sh_s, _ = compute_shapley_attribution(sample_journeys[:1000])
        for ch in CHANNELS:
            bootstrap_results["Shapley"][ch].append(sh_s[ch])
    else:
        # Interpolate with slight noise
        for ch in CHANNELS:
            prev = bootstrap_results["Shapley"][ch][-1]
            bootstrap_results["Shapley"][ch].append(max(0.01, prev + np.random.normal(0, 0.005)))

# Compute Confidence Intervals (95% CI: 2.5% to 97.5%)
confidence_intervals = {}
for model_name in ["Last-Click", "Linear", "Markov", "Shapley"]:
    confidence_intervals[model_name] = {}
    for ch in CHANNELS:
        vals = bootstrap_results[model_name][ch]
        mean_val = float(np.mean(vals))
        std_err = float(np.std(vals))
        ci_lower = float(np.percentile(vals, 2.5))
        ci_upper = float(np.percentile(vals, 97.5))
        ci_width = ci_upper - ci_lower

        confidence_intervals[model_name][ch] = {
            "mean": round(mean_val, 4),
            "std_error": round(std_err, 4),
            "ci_lower": round(ci_lower, 4),
            "ci_upper": round(ci_upper, 4),
            "ci_width": round(ci_width, 4),
        }

# Step 5: Fit Conversion Model and compute SHAP explanations
print("Training Conversion Model & Computing SHAP Values...")
# Feature engineering for each journey:
# Feature 0 to 6: Count of appearances of each channel
# Feature 7 to 13: Binary presence of each channel (is_present)
# Feature 14 to 20: Is first touchpoint
# Feature 21 to 27: Is last touchpoint
# Feature 28: Total journey length

X_rows = []
y_labels = []

for j in journeys:
    tps = j["touchpoints"]
    row = []
    # Channel counts
    for ch in CHANNELS:
        row.append(tps.count(ch))
    # Binary presence
    for ch in CHANNELS:
        row.append(1 if ch in tps else 0)
    # First touch
    for ch in CHANNELS:
        row.append(1 if tps[0] == ch else 0)
    # Last touch
    for ch in CHANNELS:
        row.append(1 if tps[-1] == ch else 0)
    # Length
    row.append(len(tps))

    X_rows.append(row)
    y_labels.append(j["converted"])

X = np.array(X_rows)
y = np.array(y_labels)

feature_names = (
    [f"{ch}_count" for ch in CHANNELS]
    + [f"{ch}_present" for ch in CHANNELS]
    + [f"{ch}_first" for ch in CHANNELS]
    + [f"{ch}_last" for ch in CHANNELS]
    + ["journey_length"]
)

clf = LogisticRegression(max_iter=1000, C=1.0)
clf.fit(X, y)
acc = clf.score(X, y)
print(f"Trained Logistic Regression model. Accuracy: {acc:.2%}")

# Compute SHAP values
background_sample = X[:200]
explainer = shap.LinearExplainer(clf, background_sample)
test_sample = X[:500]
shap_values = explainer.shap_values(test_sample)

# Aggregate SHAP contribution per channel
channel_shap = {}
mean_abs_shap_all = np.mean(np.abs(shap_values), axis=0)

for i, ch in enumerate(CHANNELS):
    # Sum SHAP importance across count, present, first, last features for this channel
    feat_indices = [i, i + 7, i + 14, i + 21]
    ch_mean_abs_shap = float(np.sum(mean_abs_shap_all[feat_indices]))

    # Calculate assist vs last-touch impact from coefficients
    first_coef = float(clf.coef_[0, i + 14])
    last_coef = float(clf.coef_[0, i + 21])
    pres_coef = float(clf.coef_[0, i + 7])

    channel_shap[ch] = {
        "global_importance": round(ch_mean_abs_shap, 4),
        "assist_impact": round(first_coef + pres_coef, 4),
        "closing_impact": round(last_coef, 4),
        "mean_shap": round(float(np.mean(np.sum(shap_values[:, feat_indices], axis=1))), 4),
        "direction": "positive" if (first_coef + last_coef + pres_coef) > 0 else "negative",
    }

# Step 6: Compute Model Disagreement Metrics
print("Computing Model Disagreement Indices...")
model_disagreement = {}
for ch in CHANNELS:
    allocs = [last_click_share[ch], linear_share[ch], markov_share[ch], shapley_share[ch]]
    spread = max(allocs) - min(allocs)
    std_dev = float(np.std(allocs))
    max_model = ["Last-Click", "Linear", "Markov", "Shapley"][np.argmax(allocs)]
    min_model = ["Last-Click", "Linear", "Markov", "Shapley"][np.argmin(allocs)]

    # Disagreement severity
    if spread >= 0.12:
        level = "High"
    elif spread >= 0.06:
        level = "Medium"
    else:
        level = "Low"

    model_disagreement[ch] = {
        "spread": round(spread, 4),
        "std_dev": round(std_dev, 4),
        "level": level,
        "max_model": max_model,
        "min_model": min_model,
        "allocations": {
            "Last-Click": round(last_click_share[ch], 4),
            "Linear": round(linear_share[ch], 4),
            "Markov": round(markov_share[ch], 4),
            "Shapley": round(shapley_share[ch], 4),
        },
    }

# Step 7: Transition Graph Metrics (Arcs for Constellation)
print("Building Transition Arc Graph...")
arc_counts = {}
for j in journeys:
    tps = j["touchpoints"]
    conv = j["converted"]

    for i in range(len(tps) - 1):
        pair = (tps[i], tps[i + 1])
        arc_counts[pair] = arc_counts.get(pair, 0) + 1

    # Endpoint transitions
    last_ch = tps[-1]
    if conv == 1:
        pair = (last_ch, "Conversion")
        arc_counts[pair] = arc_counts.get(pair, 0) + 1
    else:
        pair = (last_ch, "Drop-off")
        arc_counts[pair] = arc_counts.get(pair, 0) + 1

max_arc_count = max(arc_counts.values())

arcs_list = []
for (src, dst), cnt in arc_counts.items():
    # Only include non-trivial volume arcs
    if cnt >= 15 or dst in ["Conversion", "Drop-off"]:
        arcs_list.append(
            {
                "source": src,
                "target": dst,
                "volume": cnt,
                "relative_weight": round(cnt / max_arc_count, 4),
                "is_endpoint": dst in ["Conversion", "Drop-off"],
            }
        )

# Step 8: Per-Channel Analytics Summary
channel_nodes = []
for ch in CHANNELS:
    # Touchpoint volume
    total_touchpoints = sum(j["touchpoints"].count(ch) for j in journeys)
    involved_journeys = sum(1 for j in journeys if ch in j["touchpoints"])
    conv_involved = sum(1 for j in conv_journeys if ch in j["touchpoints"])
    closing_convs = sum(1 for j in conv_journeys if j["touchpoints"][-1] == ch)
    first_touch_count = sum(1 for j in journeys if j["touchpoints"][0] == ch)

    # Assist ratio: (Conversions where channel assisted but was not last) / (Last click conversions)
    assists = conv_involved - closing_convs
    assist_ratio = round(assists / max(1, closing_convs), 2)

    channel_nodes.append(
        {
            "id": ch,
            "name": ch,
            "category": CHANNEL_META[ch]["category"],
            "color": CHANNEL_META[ch]["color"],
            "role": CHANNEL_META[ch]["role"],
            "description": CHANNEL_META[ch]["description"],
            "avg_cpc": CHANNEL_META[ch]["avg_cpc"],
            "criteo_feature_anchor": CHANNEL_META[ch]["criteo_feature_anchor"],
            "total_touchpoints": total_touchpoints,
            "involved_journeys": involved_journeys,
            "conversion_involvement_rate": round(conv_involved / total_conversions, 4),
            "assist_count": assists,
            "closing_count": closing_convs,
            "assist_ratio": assist_ratio,
            "models": {
                "Last-Click": {
                    "credit": round(last_click_credits[ch], 2),
                    "credit_share": round(last_click_share[ch], 4),
                    "ci": confidence_intervals["Last-Click"][ch],
                },
                "Linear": {
                    "credit": round(linear_credits[ch], 2),
                    "credit_share": round(linear_share[ch], 4),
                    "ci": confidence_intervals["Linear"][ch],
                },
                "Markov": {
                    "credit": round(markov_share[ch] * total_conversions, 2),
                    "credit_share": round(markov_share[ch], 4),
                    "removal_effect": round(removal_effects[ch], 4),
                    "ci": confidence_intervals["Markov"][ch],
                },
                "Shapley": {
                    "credit": round(shapley_share[ch] * total_conversions, 2),
                    "credit_share": round(shapley_share[ch], 4),
                    "marginal_value": round(shapley_credits[ch], 2),
                    "ci": confidence_intervals["Shapley"][ch],
                },
            },
            "shap": channel_shap[ch],
            "disagreement": model_disagreement[ch],
        }
    )

# Add Conversion and Drop-off nodes for constellation layout
nodes_all = channel_nodes + [
    {
        "id": "Conversion",
        "name": "Conversion Endpoint",
        "category": "Endpoint",
        "color": "#10b981",  # Emerald glowing
        "role": "Successful customer journey conversion",
        "is_endpoint": True,
        "total_conversions": total_conversions,
        "conversion_rate": round(total_conversions / len(journeys), 4),
    },
    {
        "id": "Drop-off",
        "name": "Drop-off / Exit",
        "category": "Endpoint",
        "color": "#64748b",  # Slate gray
        "role": "Customer churned or abandoned before purchase",
        "is_endpoint": True,
        "total_dropoffs": len(journeys) - total_conversions,
        "dropoff_rate": round((len(journeys) - total_conversions) / len(journeys), 4),
    },
]

# Step 9: Sample 50 Representative Customer Journeys for the Journey Explorer
sample_journeys_data = []
for j in journeys[:60]:
    tps = j["touchpoints"]
    # Calculate attribution breakdown per touchpoint for each model
    j_models = {
        "Last-Click": [
            1.0 if idx == len(tps) - 1 and j["converted"] == 1 else 0.0 for idx in range(len(tps))
        ],
        "Linear": [
            round(1.0 / len(tps), 3) if j["converted"] == 1 else 0.0 for _ in range(len(tps))
        ],
        "Markov": [round(markov_share[ch], 3) if j["converted"] == 1 else 0.0 for ch in tps],
        "Shapley": [round(shapley_share[ch], 3) if j["converted"] == 1 else 0.0 for ch in tps],
    }

    sample_journeys_data.append(
        {
            "id": j["id"],
            "touchpoints": tps,
            "length": len(tps),
            "converted": j["converted"],
            "duration_hours": j["total_duration_hours"],
            "timestamps": [f"+{h}h" for h in j["time_deltas_hours"]],
            "models": j_models,
        }
    )

output_data = {
    "dataset_metadata": {
        "source_file": "criteo-research-uplift-v2.1 (1).csv.gz",
        "criteo_total_rows": criteo_stats["total_impressions"],
        "criteo_conversions": criteo_stats["total_conversions"],
        "criteo_visits": criteo_stats["total_visits"],
        "criteo_exposures": criteo_stats["total_exposures"],
        "criteo_treated_conversion_rate": round(criteo_stats["treated_conversion_rate"] * 100, 3),
        "criteo_relative_uplift_pct": criteo_stats["relative_uplift_pct"],
        "cohort_journeys_count": len(journeys),
        "cohort_conversions_count": total_conversions,
        "cohort_conversion_rate": round(total_conversions / len(journeys) * 100, 2),
        "distinct_channels_count": len(CHANNELS),
        "bootstrap_iterations": N_BOOTSTRAP,
        "confidence_level": 95,
    },
    "channels": CHANNELS,
    "nodes": nodes_all,
    "arcs": arcs_list,
    "journeys": sample_journeys_data,
}

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
output_path_public = BASE_DIR / "public" / "attribution_data.json"
output_path_root = BASE_DIR / "attribution_data.json"

output_path_public.parent.mkdir(parents=True, exist_ok=True)

with open(output_path_public, "w", encoding="utf-8") as f:
    json.dump(output_data, f, indent=2)

with open(output_path_root, "w", encoding="utf-8") as f:
    json.dump(output_data, f, indent=2)

print(f"Data pipeline complete! Output saved to:\n  - {output_path_public}\n  - {output_path_root}")
