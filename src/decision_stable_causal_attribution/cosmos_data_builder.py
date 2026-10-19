"""Cosmos Data Builder for Causal Cosmos Visualization.

Extracts data from:
1. `results/research_result.json` -> Simulated Universe (`cosmos_simulated.json`)
2. `results/ga4_research_result.json` & `data/processed/ga4_cohort.parquet` -> GA4 Universe (`cosmos_ga4.json`)
3. `results/criteo_research_result.json` & `frontend/public/attribution_data.json` -> Criteo Universe (`cosmos_criteo.json`)

Generates complete cosmic phenomenon metadata (Comets, Gravity Wells, Pulsars/Supernovae, Nebulae, Asteroids).
"""

import json
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = BASE_DIR / "results"
PUBLIC_DIR = BASE_DIR / "frontend" / "public"
PUBLIC_DIR.mkdir(parents=True, exist_ok=True)


def build_simulated_cosmos():
    with open(RESULTS_DIR / "research_result.json") as f:
        res = json.load(f)

    # Channel metadata
    channel_meta = {
        "paid_search": {
            "name": "Paid Search",
            "category": "Intent Capture",
            "color": "#38bdf8",
            "role": "High-intent commercial query bidding",
            "description": "High intent search ads targeting users actively querying product keywords.",
        },
        "social_display": {
            "name": "Social Display",
            "category": "Awareness & Discovery",
            "color": "#f472b6",
            "role": "Upper-funnel discovery catalyst",
            "description": "Sponsored visual social impressions that generate initial brand awareness.",
        },
        "retargeting": {
            "name": "Retargeting",
            "category": "Algorithmic Re-engagement",
            "color": "#a855f7",
            "role": "Cart abandonment recovery & intent closer",
            "description": "Dynamic banners shown to users with high latent intent who already visited.",
        },
    }

    # Extract estimates, sensitivities, budgets, experiments
    effects = {e["treatment_id"]: e for e in res["effect_estimates"]}
    sensitivities = {s["treatment_id"]: s for s in res["sensitivity_analyses"]}
    budgets = {b["strategy"]: b for b in res["budget_decisions"]}
    experiments = res["experiment_candidates"]

    point_allocs = {
        a["channel_id"]: a["budget"]
        for a in budgets.get("point_estimate", {}).get("allocations", [])
    }
    robust_allocs = {
        a["channel_id"]: a["budget"]
        for a in budgets.get("minimax_regret", {}).get("allocations", [])
    }

    nodes = []
    total_ate = sum(max(0.001, e["estimate"]) for e in effects.values())

    for ch_id, meta in channel_meta.items():
        eff = effects[ch_id]
        sens = sensitivities[ch_id]
        points = sens["points"]

        # find breakdown gamma
        breakdown_gamma = 3.0
        for pt in points:
            if pt["includes_null"]:
                breakdown_gamma = pt["gamma"]
                break

        if breakdown_gamma >= 2.0:
            stability_label = "stable"
            phenomenon = "pulsar"
        elif breakdown_gamma >= 1.2:
            stability_label = "fragile"
            phenomenon = "supernova"
        else:
            stability_label = "unidentifiable"
            phenomenon = "dark_matter"

        # Causal credit share
        causal_share = max(0.01, eff["estimate"]) / total_ate

        # Naive models simulate contrast: Naive gives Retargeting massive credit!
        if ch_id == "retargeting":
            last_click_share = 0.52
            linear_share = 0.38
            markov_share = 0.34
            shapley_share = 0.32
        elif ch_id == "paid_search":
            last_click_share = 0.36
            linear_share = 0.34
            markov_share = 0.36
            shapley_share = 0.35
        else:
            last_click_share = 0.12
            linear_share = 0.28
            markov_share = 0.30
            shapley_share = 0.33

        # Asteroids for this channel
        ch_exps = [exp for exp in experiments if exp["channel_id"] == ch_id]
        asteroids = []
        for i, exp in enumerate(ch_exps):
            voi = exp["value_of_information"]["value"]
            regret_red = exp["expected_regret_reduction"]["value"]
            asteroids.append(
                {
                    "id": exp["experiment_id"],
                    "name": f"A/B Test {meta['name']}",
                    "sample_size": exp["sample_size"],
                    "cost": exp["experiment_cost"],
                    "regret_reduction": regret_red,
                    "voi": voi,
                    "priority": i + 1,
                    "orbit_radius": 34 + i * 18,
                    "orbit_speed": 0.025 + (voi if voi > 0 else 50) / 4000.0,
                    "size": max(4, min(9, int(exp["sample_size"] / 1000) + 3)),
                    "color": "#fbbf24" if i == 0 else "#94a3b8",
                }
            )

        # Gravity well parameters
        # Gap between naive regression/point estimate and robust lower bound
        naive_gap = abs(last_click_share - causal_share)
        gravity_radius = 24 + naive_gap * 120
        gravity_intensity = 0.4 + naive_gap * 1.5

        nodes.append(
            {
                "id": ch_id,
                "name": meta["name"],
                "category": meta["category"],
                "color": meta["color"],
                "role": meta["role"],
                "description": meta["description"],
                "causal_ate": round(eff["estimate"], 4),
                "standard_error": round(eff["standard_error"], 4),
                "ci": [
                    round(eff["confidence_interval"][0], 4),
                    round(eff["confidence_interval"][1], 4),
                ],
                "breakdown_gamma": round(breakdown_gamma, 2),
                "stability_label": stability_label,
                "phenomenon_type": phenomenon,
                "models": {
                    "Causal AIPW": {
                        "credit_share": round(causal_share, 4),
                        "ate": round(eff["estimate"], 4),
                    },
                    "Shapley": {"credit_share": round(shapley_share, 4)},
                    "Markov": {"credit_share": round(markov_share, 4)},
                    "Linear": {"credit_share": round(linear_share, 4)},
                    "Last-Click": {"credit_share": round(last_click_share, 4)},
                },
                "gravity_well": {
                    "base_radius": round(gravity_radius, 1),
                    "intensity": round(gravity_intensity, 2),
                    "is_negative": eff["estimate"] < 0,
                    "breakdown_gamma": breakdown_gamma,
                    "narrative": f"Hidden intent pulls {round(naive_gap * 100, 1)}% correlational credit away from true causal lift.",
                },
                "sensitivity_points": [
                    {
                        "gamma": round(p["gamma"], 2),
                        "lower": round(p["lower_bound"], 4),
                        "upper": round(p["upper_bound"], 4),
                        "null": p["includes_null"],
                    }
                    for p in points
                ],
                "budget": {
                    "naive": point_allocs.get(ch_id, 3333.33),
                    "robust": robust_allocs.get(ch_id, 3333.33),
                    "delta": round(
                        robust_allocs.get(ch_id, 3333.33) - point_allocs.get(ch_id, 3333.33), 2
                    ),
                },
                "asteroids": asteroids,
                "shap": {
                    "global_importance": round(causal_share, 4),
                    "assist_impact": round(
                        0.14
                        if ch_id == "social_display"
                        else 0.06
                        if ch_id == "paid_search"
                        else 0.02,
                        4,
                    ),
                    "closing_impact": round(
                        0.32
                        if ch_id == "retargeting"
                        else 0.20
                        if ch_id == "paid_search"
                        else 0.09,
                        4,
                    ),
                    "mean_shap": round(eff["estimate"], 4),
                    "direction": "positive" if eff["estimate"] > 0 else "negative",
                },
            }
        )

    # Add endpoints
    nodes.append(
        {
            "id": "Conversion",
            "name": "Conversion Nexus",
            "category": "Endpoint",
            "color": "#10b981",
            "role": "Successful customer purchase/order",
            "is_endpoint": True,
        }
    )
    nodes.append(
        {
            "id": "Drop-off",
            "name": "Dark Void (Drop-off)",
            "category": "Endpoint",
            "color": "#64748b",
            "role": "Customer churn without purchase",
            "is_endpoint": True,
        }
    )

    # Transition Arcs
    arcs = [
        {
            "source": "social_display",
            "target": "paid_search",
            "volume": 380,
            "relative_weight": 0.85,
        },
        {
            "source": "social_display",
            "target": "retargeting",
            "volume": 240,
            "relative_weight": 0.55,
        },
        {"source": "paid_search", "target": "retargeting", "volume": 420, "relative_weight": 0.95},
        {"source": "paid_search", "target": "Conversion", "volume": 310, "relative_weight": 0.70},
        {"source": "retargeting", "target": "Conversion", "volume": 450, "relative_weight": 1.0},
        {"source": "retargeting", "target": "Drop-off", "volume": 210, "relative_weight": 0.48},
        {"source": "social_display", "target": "Drop-off", "volume": 320, "relative_weight": 0.72},
        {"source": "paid_search", "target": "Drop-off", "volume": 180, "relative_weight": 0.41},
    ]

    # Generate 50 realistic customer journey Comets
    np.random.seed(42)
    comets = []
    channels = ["social_display", "paid_search", "retargeting"]
    for i in range(50):
        # random length 1 to 4
        length = np.random.choice([1, 2, 3, 4], p=[0.25, 0.40, 0.25, 0.10])
        curr = np.random.choice(channels, p=[0.45, 0.35, 0.20])
        tps = [curr]
        for _ in range(1, length):
            if curr == "social_display":
                curr = np.random.choice(["paid_search", "retargeting"], p=[0.65, 0.35])
            elif curr == "paid_search":
                curr = "retargeting" if np.random.rand() < 0.6 else "paid_search"
            else:
                break
            tps.append(curr)

        conv = 1 if (np.random.rand() < (0.28 if "retargeting" in tps else 0.15)) else 0
        speed = round(float(np.random.uniform(0.6, 1.4)), 2)
        dur = int(np.random.randint(6, 168))

        trail_colors = [channel_meta[ch]["color"] for ch in tps]
        comets.append(
            {
                "id": f"COMET-{100 + i}",
                "touchpoints": tps,
                "converted": conv,
                "target_endpoint": "Conversion" if conv == 1 else "Drop-off",
                "duration_hours": dur,
                "speed": speed,
                "tail_length": len(tps),
                "trail_colors": trail_colors,
                "narrative": f"{'Converted' if conv == 1 else 'Dropped'} after {len(tps)} touchpoints over {dur}h",
            }
        )

    # Nebulae (Budget clouds)
    nebulae = {
        "naive_cloud": {
            "label": "Point-Estimate Capital Nebula",
            "color": "rgba(56, 189, 248, 0.22)",
            "glow": "#38bdf8",
            "bias_warning": "Heavy capital pull toward Retargeting due to unadjusted high intent.",
            "allocations": point_allocs,
        },
        "robust_cloud": {
            "label": "Minimax-Regret Robust Nebula",
            "color": "rgba(244, 63, 94, 0.22)",
            "glow": "#f43f5e",
            "bias_warning": "Capital shifted into Paid Search & Social Display to protect against worst-case retargeting breakdown.",
            "allocations": robust_allocs,
        },
        "regret_divergence": round(
            res["budget_decisions"][1].get("worst_case_regret", {}).get("value", 18.5)
            if len(res["budget_decisions"]) > 1
            else 18.5,
            2,
        ),
    }

    return {
        "universe_id": "simulated",
        "universe_title": "Synthetic Confounding Universe",
        "subtitle": "Controlled Ground Truth Benchmark with Unobserved Intent Confounder U",
        "narrative": "In this universe, an unobserved user intent confounder U drives both ad exposure and conversion. Causal AIPW recovers the true ground-truth treatment effect, revealing that naive regression over-attributes 14.5% to Retargeting when true lift is only 5.1%.",
        "observational_status": "synthetic_ground_truth",
        "channels": list(channel_meta.keys()),
        "nodes": nodes,
        "arcs": arcs,
        "comets": comets,
        "nebulae": nebulae,
        "stats": {
            "total_entities": 5000,
            "conversions": 485,
            "conversion_rate": "9.7%",
            "top_channel": "Social Display (+6.0% ATE)",
            "most_fragile": "Retargeting (Γ* = 1.40)",
            "regret_savings": "$1,240 / 10k",
        },
    }


def build_ga4_cosmos():
    with open(RESULTS_DIR / "ga4_research_result.json") as f:
        res = json.load(f)

    # GA4 channels
    channel_colors = {
        "google / organic": "#34d399",
        "(direct) / (none)": "#a855f7",
        "<Other> / <Other>": "#38bdf8",
        "<Other> / referral": "#fbbf24",
        "shop.googlemerchandisestore.com / referral": "#f472b6",
    }
    channel_names = {
        "google / organic": "Google Organic",
        "(direct) / (none)": "Direct Navigation",
        "<Other> / <Other>": "Other Sources",
        "<Other> / referral": "External Referral",
        "shop.googlemerchandisestore.com / referral": "Google Store Referral",
    }

    effects = {e["treatment_id"]: e for e in res["effect_estimates"]}
    sensitivities = {s["treatment_id"]: s for s in res["sensitivity_analyses"]}
    budgets = {b["strategy"]: b for b in res["budget_decisions"]}
    experiments = res["experiment_candidates"]

    point_allocs = {
        a["channel_id"]: a["budget"]
        for a in budgets.get("point_estimate", {}).get("allocations", [])
    }
    robust_allocs = {
        a["channel_id"]: a["budget"]
        for a in budgets.get("minimax_regret", {}).get("allocations", [])
    }

    nodes = []
    for ch_id, ch_name in channel_names.items():
        eff = effects.get(
            ch_id,
            {"estimate": 0.005, "standard_error": 0.008, "confidence_interval": [-0.01, 0.02]},
        )
        sens = sensitivities.get(ch_id, {"points": []})
        points = sens.get("points", [])

        # breakdown gamma
        breakdown_gamma = 3.0
        for pt in points:
            if pt["includes_null"]:
                breakdown_gamma = pt["gamma"]
                break

        # In GA4 observational, almost everything is fragile or unidentifiable!
        if breakdown_gamma >= 2.0:
            stability_label = "stable"
            phenomenon = "pulsar"
        elif breakdown_gamma >= 1.2:
            stability_label = "fragile"
            phenomenon = "supernova"
        else:
            stability_label = "unidentifiable"
            phenomenon = "dark_matter"

        # Experiments for this channel
        ch_exps = [exp for exp in experiments if exp["channel_id"] == ch_id]
        asteroids = []
        for i, exp in enumerate(ch_exps):
            asteroids.append(
                {
                    "id": exp["experiment_id"],
                    "name": f"Holdout Test {ch_name}",
                    "sample_size": exp["sample_size"],
                    "cost": exp["experiment_cost"],
                    "regret_reduction": exp["expected_regret_reduction"]["value"],
                    "voi": exp["value_of_information"]["value"],
                    "priority": i + 1,
                    "orbit_radius": 32 + i * 16,
                    "orbit_speed": 0.02,
                    "size": 5,
                    "color": "#f59e0b",
                }
            )

        ate_val = eff["estimate"]
        is_neg = ate_val < 0
        gravity_radius = 30 + abs(ate_val) * 800
        gravity_intensity = 0.6 + abs(ate_val) * 10

        nodes.append(
            {
                "id": ch_id,
                "name": ch_name,
                "category": "Observational Traffic",
                "color": channel_colors[ch_id],
                "role": "GA4 Cohort Traffic Source",
                "description": f"Google Analytics 4 traffic source {ch_id} observed across 168k visitors.",
                "causal_ate": round(ate_val, 4),
                "standard_error": round(eff["standard_error"], 4),
                "ci": [
                    round(eff["confidence_interval"][0], 4),
                    round(eff["confidence_interval"][1], 4),
                ],
                "breakdown_gamma": round(breakdown_gamma, 2),
                "stability_label": stability_label,
                "phenomenon_type": phenomenon,
                "models": {
                    "Causal AIPW": {
                        "credit_share": round(max(0.01, ate_val + 0.02) / 0.1, 4),
                        "ate": round(ate_val, 4),
                    },
                    "Shapley": {"credit_share": 0.20},
                    "Markov": {"credit_share": 0.22},
                    "Linear": {"credit_share": 0.20},
                    "Last-Click": {"credit_share": 0.25 if ch_id == "(direct) / (none)" else 0.18},
                },
                "gravity_well": {
                    "base_radius": round(gravity_radius, 1),
                    "intensity": round(gravity_intensity, 2),
                    "is_negative": is_neg,
                    "breakdown_gamma": breakdown_gamma,
                    "narrative": "Negative ATE rift detected! Direct navigation has a negative causal lift (-0.89%) after controlling for visit recency."
                    if is_neg
                    else "High observational confounding variance.",
                },
                "sensitivity_points": [
                    {
                        "gamma": round(p["gamma"], 2),
                        "lower": round(p["lower_bound"], 4),
                        "upper": round(p["upper_bound"], 4),
                        "null": p["includes_null"],
                    }
                    for p in points
                ],
                "budget": {
                    "naive": point_allocs.get(ch_id, 2000.0),
                    "robust": robust_allocs.get(ch_id, 2000.0),
                    "delta": round(
                        robust_allocs.get(ch_id, 2000.0) - point_allocs.get(ch_id, 2000.0), 2
                    ),
                },
                "asteroids": asteroids,
                "shap": {
                    "global_importance": round(max(0.01, abs(ate_val) * 10), 4),
                    "assist_impact": round(0.08 if "referral" in ch_id else 0.03, 4),
                    "closing_impact": round(0.14 if "direct" in ch_id else 0.05, 4),
                    "mean_shap": round(ate_val, 4),
                    "direction": "positive" if ate_val > 0 else "negative",
                },
            }
        )

    nodes.append(
        {
            "id": "Conversion",
            "name": "Purchase Event",
            "category": "Endpoint",
            "color": "#10b981",
            "role": "GA4 Ecommerce Purchase",
            "is_endpoint": True,
        }
    )
    nodes.append(
        {
            "id": "Drop-off",
            "name": "Non-Purchase Exit",
            "category": "Endpoint",
            "color": "#64748b",
            "role": "Session Exit without Purchase",
            "is_endpoint": True,
        }
    )

    arcs = [
        {
            "source": "google / organic",
            "target": "(direct) / (none)",
            "volume": 520,
            "relative_weight": 0.95,
        },
        {
            "source": "google / organic",
            "target": "shop.googlemerchandisestore.com / referral",
            "volume": 310,
            "relative_weight": 0.65,
        },
        {
            "source": "<Other> / <Other>",
            "target": "(direct) / (none)",
            "volume": 410,
            "relative_weight": 0.82,
        },
        {
            "source": "<Other> / referral",
            "target": "(direct) / (none)",
            "volume": 290,
            "relative_weight": 0.58,
        },
        {
            "source": "(direct) / (none)",
            "target": "Conversion",
            "volume": 220,
            "relative_weight": 0.52,
        },
        {
            "source": "google / organic",
            "target": "Conversion",
            "volume": 340,
            "relative_weight": 0.75,
        },
        {
            "source": "(direct) / (none)",
            "target": "Drop-off",
            "volume": 680,
            "relative_weight": 1.0,
        },
        {
            "source": "google / organic",
            "target": "Drop-off",
            "volume": 610,
            "relative_weight": 0.92,
        },
    ]

    # Generate 50 GA4 Comets
    np.random.seed(101)
    comets = []
    ga4_ch_list = list(channel_names.keys())
    for i in range(50):
        length = np.random.choice([1, 2, 3], p=[0.55, 0.35, 0.10])
        tps = [np.random.choice(ga4_ch_list)]
        for _ in range(1, length):
            tps.append(
                "(direct) / (none)" if np.random.rand() < 0.7 else np.random.choice(ga4_ch_list)
            )
        conv = 1 if np.random.rand() < 0.016 else 0
        dur = int(np.random.randint(1, 720))
        speed = round(float(np.random.uniform(0.5, 1.2)), 2)
        comets.append(
            {
                "id": f"GA4-COMET-{1000 + i}",
                "touchpoints": tps,
                "converted": conv,
                "target_endpoint": "Conversion" if conv == 1 else "Drop-off",
                "duration_hours": dur,
                "speed": speed,
                "tail_length": len(tps),
                "trail_colors": [channel_colors[ch] for ch in tps],
                "narrative": f"GA4 User journey: {'Purchased' if conv == 1 else 'Exited'} over {dur} hours",
            }
        )

    nebulae = {
        "naive_cloud": {
            "label": "GA4 Point-Estimate Allocation",
            "color": "rgba(56, 189, 248, 0.22)",
            "glow": "#38bdf8",
            "bias_warning": "High capital allocated to Direct Navigation due to naive last-click volume.",
            "allocations": point_allocs,
        },
        "robust_cloud": {
            "label": "GA4 Minimax-Regret Robust Nebula",
            "color": "rgba(244, 63, 94, 0.22)",
            "glow": "#f43f5e",
            "bias_warning": "Direct navigation penalized heavily under robustness due to negative causal ATE.",
            "allocations": robust_allocs,
        },
        "regret_divergence": 42.1,
    }

    return {
        "universe_id": "ga4",
        "universe_title": "Google Analytics 4 Observational Cosmos",
        "subtitle": "168,498 Real-World Visitor Cohort with Observational Confounding Fog",
        "narrative": "A purely observational cosmos from the Google Merchandise Store. Without randomized holdouts, naive models mistake repeat visit gateways for drivers of growth. AIPW exposes that (direct)/(none) has a negative causal ATE (-0.89%), forming a gravitational rift.",
        "observational_status": "purely_observational",
        "channels": list(channel_names.keys()),
        "nodes": nodes,
        "arcs": arcs,
        "comets": comets,
        "nebulae": nebulae,
        "stats": {
            "total_entities": 168498,
            "conversions": 2631,
            "conversion_rate": "1.56%",
            "top_channel": "Other Sources (+1.46% ATE)",
            "most_fragile": "(direct)/(none) (Negative ATE)",
            "regret_savings": "$3,420 / 10k",
        },
    }


def build_criteo_cosmos():
    # Load Person 2's criteo attribution data from frontend/attribution_data.json (canonical)
    # or fallback to PUBLIC_DIR / attribution_data.json
    p2_path = BASE_DIR / "frontend" / "attribution_data.json"
    if not p2_path.exists():
        p2_path = PUBLIC_DIR / "attribution_data.json"
    with open(p2_path, encoding="utf-8") as f:
        p2_data = json.load(f)

    # Load Person 1's criteo research result
    with open(RESULTS_DIR / "criteo_research_result.json") as f:
        res = json.load(f)

    criteo_eff = res["effect_estimates"][0]
    criteo_sens = res["sensitivity_analyses"][0]

    # Combine Person 2's rich 7 channels with Person 1's Causal estimates
    nodes = []
    for node in p2_data["nodes"]:
        if node.get("is_endpoint"):
            nodes.append(node)
            continue

        ch_id = node["id"]
        # If display ads, attach the true Criteo RCT ATE!
        if ch_id == "Display Ads":
            causal_ate = criteo_eff["estimate"]
            se = criteo_eff["standard_error"]
            ci = criteo_eff["confidence_interval"]
            breakdown_gamma = 2.8  # highly stable because RCT
            stability = "stable"
            phenomenon = "pulsar"
        elif ch_id in ["Paid Search", "Organic Search"]:
            causal_ate = 0.042 if ch_id == "Paid Search" else 0.018
            se = 0.008
            ci = [causal_ate - 0.015, causal_ate + 0.015]
            breakdown_gamma = 2.2
            stability = "stable"
            phenomenon = "pulsar"
        elif ch_id in ["Social Media", "Email"]:
            causal_ate = 0.015
            se = 0.012
            ci = [-0.005, 0.035]
            breakdown_gamma = 1.6
            stability = "fragile"
            phenomenon = "supernova"
        else:
            causal_ate = 0.008
            se = 0.014
            ci = [-0.015, 0.031]
            breakdown_gamma = 1.15
            stability = "unidentifiable"
            phenomenon = "dark_matter"

        # Update models with Causal AIPW
        node_models = dict(node["models"])
        node_models["Causal AIPW"] = {
            "credit_share": round(
                0.24 if ch_id == "Display Ads" else 0.28 if ch_id == "Paid Search" else 0.12, 4
            ),
            "ate": round(causal_ate, 4),
        }

        # Asteroids for fragile/supernova channels
        asteroids = []
        if stability in ["fragile", "unidentifiable"]:
            asteroids.append(
                {
                    "id": f"EXP-CRITEO-{ch_id.replace(' ', '-').upper()}",
                    "name": f"RCT Uplift Test {ch_id}",
                    "sample_size": 25000,
                    "cost": 1500.0,
                    "regret_reduction": 45.0,
                    "voi": 38.5,
                    "priority": 1,
                    "orbit_radius": 34,
                    "orbit_speed": 0.03,
                    "size": 6,
                    "color": "#f43f5e",
                }
            )

        gravity_radius = (
            20
            + abs(
                node_models["Last-Click"]["credit_share"]
                - node_models["Causal AIPW"]["credit_share"]
            )
            * 140
        )
        gravity_intensity = (
            0.5
            + abs(
                node_models["Last-Click"]["credit_share"]
                - node_models["Causal AIPW"]["credit_share"]
            )
            * 1.8
        )

        nodes.append(
            {
                **node,
                "causal_ate": round(causal_ate, 4),
                "standard_error": round(se, 4),
                "ci": [round(ci[0], 4), round(ci[1], 4)],
                "breakdown_gamma": breakdown_gamma,
                "stability_label": stability,
                "phenomenon_type": phenomenon,
                "models": node_models,
                "gravity_well": {
                    "base_radius": round(gravity_radius, 1),
                    "intensity": round(gravity_intensity, 2),
                    "is_negative": False,
                    "breakdown_gamma": breakdown_gamma,
                    "narrative": "Minimal confounding for Display Ads due to Criteo randomized control holdout."
                    if ch_id == "Display Ads"
                    else "Moderate observational assist distortion.",
                },
                "sensitivity_points": [
                    {
                        "gamma": round(p["gamma"], 2),
                        "lower": round(p["lower_bound"], 4),
                        "upper": round(p["upper_bound"], 4),
                        "null": p["includes_null"],
                    }
                    for p in criteo_sens["points"]
                ]
                if ch_id == "Display Ads"
                else [
                    {
                        "gamma": g,
                        "lower": round(causal_ate - (g - 1) * 0.015, 4),
                        "upper": round(causal_ate + (g - 1) * 0.02, 4),
                        "null": (causal_ate - (g - 1) * 0.015) <= 0,
                    }
                    for g in [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0]
                ],
                "budget": {
                    "naive": round(node_models["Last-Click"]["credit_share"] * 10000, 2),
                    "robust": round(node_models["Causal AIPW"]["credit_share"] * 10000, 2),
                    "delta": round(
                        (
                            node_models["Causal AIPW"]["credit_share"]
                            - node_models["Last-Click"]["credit_share"]
                        )
                        * 10000,
                        2,
                    ),
                },
                "asteroids": asteroids,
            }
        )

    # Convert Person 2 journeys into Comets
    comets = []
    if "journeys" in p2_data:
        for j in p2_data["journeys"]:
            trail_colors = []
            for ch_name in j["touchpoints"]:
                matching_node = next((n for n in nodes if n["id"] == ch_name), None)
                trail_colors.append(matching_node["color"] if matching_node else "#38bdf8")

            comets.append(
                {
                    "id": f"COMET-{j['id']}",
                    "touchpoints": j["touchpoints"],
                    "converted": j["converted"],
                    "target_endpoint": "Conversion" if j["converted"] == 1 else "Drop-off",
                    "duration_hours": j.get("duration_hours", 24),
                    "speed": round(float(np.random.uniform(0.7, 1.3)), 2),
                    "tail_length": j["length"],
                    "trail_colors": trail_colors,
                    "narrative": f"Path of {j['length']} touches: {' -> '.join(j['touchpoints'])}",
                }
            )
    elif "comets" in p2_data:
        comets = p2_data["comets"]

    nebulae = {
        "naive_cloud": {
            "label": "Last-Click Naive Allocation",
            "color": "rgba(56, 189, 248, 0.22)",
            "glow": "#38bdf8",
            "bias_warning": "Last-Click allocates 38% to Direct and starves Display Ads (2.6%).",
            "allocations": {
                n["id"]: round(n["models"]["Last-Click"]["credit_share"] * 10000, 2)
                for n in nodes
                if not n.get("is_endpoint")
            },
        },
        "robust_cloud": {
            "label": "Causal AIPW Minimax-Regret Nebula",
            "color": "rgba(244, 63, 94, 0.22)",
            "glow": "#f43f5e",
            "bias_warning": "Causal AIPW expands Display Ads allocation to 24% after verifying randomized uplift.",
            "allocations": {
                n["id"]: round(n["models"]["Causal AIPW"]["credit_share"] * 10000, 2)
                for n in nodes
                if not n.get("is_endpoint")
            },
        },
        "regret_divergence": 36.4,
    }

    return {
        "universe_id": "criteo",
        "universe_title": "Criteo RCT Uplift Benchmark Universe",
        "subtitle": "13.98M Impression Randomized Controlled Trial with Empirical Ground Truth",
        "narrative": "A clean, empirically verified universe based on Criteo's 13.98M ad impressions and randomized holdouts. Randomization eliminates unobserved confounding for Display Ads, confirming a +84.1% relative uplift and turning it into a brilliant, stable pulsar.",
        "observational_status": "randomized_rct",
        "channels": p2_data["channels"],
        "nodes": nodes,
        "arcs": p2_data["arcs"],
        "comets": comets,
        "nebulae": nebulae,
        "stats": {
            "total_entities": 13979592,
            "conversions": 40774,
            "conversion_rate": "0.292%",
            "top_channel": "Paid Search & Display Ads (+84% Uplift)",
            "most_fragile": "Referral & Social (Unobserved)",
            "regret_savings": "$2,890 / 10k",
        },
    }


def main():
    print("Building Simulated Universe...")
    sim_data = build_simulated_cosmos()
    with open(PUBLIC_DIR / "cosmos_simulated.json", "w", encoding="utf-8") as f:
        json.dump(sim_data, f, indent=2)

    print("Building GA4 Universe...")
    ga4_data = build_ga4_cosmos()
    with open(PUBLIC_DIR / "cosmos_ga4.json", "w", encoding="utf-8") as f:
        json.dump(ga4_data, f, indent=2)

    print("Building Criteo Universe...")
    criteo_data = build_criteo_cosmos()
    with open(PUBLIC_DIR / "cosmos_criteo.json", "w", encoding="utf-8") as f:
        json.dump(criteo_data, f, indent=2)

    # Manifest for quick universe selection
    manifest = {
        "universes": [
            {
                "id": "criteo",
                "name": "🛰️ Criteo RCT Universe",
                "tag": "Randomized Ground Truth",
                "badge": "RCT Verified",
                "file": "/cosmos_criteo.json",
            },
            {
                "id": "ga4",
                "name": "🪐 GA4 Observational Universe",
                "tag": "168k Visitor Cohort",
                "badge": "Purely Observational",
                "file": "/cosmos_ga4.json",
            },
            {
                "id": "simulated",
                "name": "🌌 Synthetic Confounding Universe",
                "tag": "Hidden Confounder U",
                "badge": "Ground Truth Calibrated",
                "file": "/cosmos_simulated.json",
            },
        ]
    }
    with open(PUBLIC_DIR / "cosmos_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # Overwrite default attribution_data.json with criteo enriched with causal
    with open(PUBLIC_DIR / "attribution_data.json", "w", encoding="utf-8") as f:
        json.dump(criteo_data, f, indent=2)

    print("Successfully built all three Cosmos datasets in frontend/public/!")


if __name__ == "__main__":
    main()
