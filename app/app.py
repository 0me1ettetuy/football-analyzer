"""Transfer flop-risk screening tool (demo).

Run from the repository root:   streamlit run app/app.py
Needs: models/flop_model.joblib (notebook 03) and data/processed/transfers_model.csv.gz (notebook 01).
"""
from pathlib import Path
import json

import joblib
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "flop_model.joblib"
DATA_PATH = ROOT / "data" / "processed" / "transfers_model.csv.gz"
METRICS_PATH = ROOT / "reports" / "tables" / "03_metrics.json"

st.set_page_config(page_title="Transfer flop-risk screen", page_icon="⚽", layout="wide")

# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model and data ...")
def load():
    bundle = joblib.load(MODEL_PATH)
    d = pd.read_csv(DATA_PATH)
    return bundle, d


bundle, d = load()
FEATURES, MODEL_COLS = bundle["features"], bundle["model_columns"]
NUMERIC = [f for f in FEATURES if f not in ("main_position", "foot")]


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def make_X(df: pd.DataFrame) -> pd.DataFrame:
    """Same model matrix as notebook 03: numeric features as they are, position and foot one-hot with fixed levels."""
    X = df[NUMERIC].copy()
    for lv in bundle["pos_levels"]:
        X["pos_" + lv] = (df["main_position"] == lv).astype(float)
    for lv in bundle["foot_levels"]:
        X["foot_" + lv] = (df["foot"] == lv).astype(float)
    return X[MODEL_COLS]


def predict(df: pd.DataFrame) -> np.ndarray:
    """Calibrated flop probability (Platt scaling fitted on 2020-2021, as in notebook 03)."""
    raw = bundle["model"].predict_proba(make_X(df))[:, 1]
    return bundle["calibrator"].predict_proba(logit(raw).reshape(-1, 1))[:, 1]


@st.cache_data(show_spinner=False)
def score_history():
    out = d.copy()
    out["risk"] = predict(out)
    return out


hist = score_history()
test = hist[hist["season_y"] >= 2022]
FLAG_10 = float(np.quantile(test["risk"], 0.90))      # risk level above which a deal is in the riskiest 10 % of 2022-24
BASE = float(test["y_flop"].mean())

# --------------------------------------------------------------------------------------
# Building one hypothetical transfer from a few inputs
# --------------------------------------------------------------------------------------
KEY = ["age", "log_value", "prev1_share", "dest_level", "orig_level", "is_gk"]
_h = d.assign(is_gk=(d["main_position"] == "Goalkeeper").astype(float))
_KEYDATA = _h[KEY]
_SCALE = _KEYDATA.std().replace(0, 1)


def neighbours(inp: dict, k: int) -> pd.DataFrame:
    """The k historical transfers closest to the user's inputs (standardised distance on the key inputs)."""
    q = pd.Series({c: inp[c] for c in KEY})
    dist = (((_KEYDATA.fillna(_KEYDATA.median()) - q) / _SCALE) ** 2).sum(axis=1) ** 0.5
    return d.loc[dist.nsmallest(k).index]


def build_row(inp: dict) -> pd.DataFrame:
    """Fill the features the user did not set with the median of the 300 most similar past transfers."""
    near = neighbours(inp, 300)
    row = {c: near[c].median() for c in NUMERIC}          # NaN stays NaN: the model handles missing values
    value = float(np.expm1(inp["log_value"]))
    fee = inp["fee_eur"]
    row.update(
        age=inp["age"], log_value=inp["log_value"], log_fee=float(np.log1p(fee)), fee_zero=int(fee == 0),
        fee_to_value=float(min(fee / value, 5.0)) if value > 0 else 0.0, from_no_club=0,
        prev1_share=inp["prev1_share"], prev1_missing=0,
        dest_level=inp["dest_level"], orig_level=inp["orig_level"], tier_change=inp["orig_level"] - inp["dest_level"],
        dest_rank_pct=inp["dest_rank_pct"], dest_regulars_same_pos=inp["regulars"], inj_days_1y=inp["inj_days"],
    )
    # the club's points per game follows from its finish in the table (typical value among similar finishes)
    sim = d[(d["dest_level"] == inp["dest_level"]) & (d["dest_rank_pct"].sub(inp["dest_rank_pct"]).abs() < 0.06)]
    if len(sim) > 30:
        row["dest_ppg"] = float(sim["dest_ppg"].median())
    row["main_position"] = inp["position"]
    row["foot"] = "right"
    return pd.DataFrame([row])[FEATURES]


def risk_of(inp: dict) -> float:
    return float(predict(build_row(inp))[0])


# --------------------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------------------
st.title("⚽ Will this transfer flop?")
st.caption(
    "A screening tool for sporting directors: it ranks permanent summer transfers by the risk that the player plays "
    "**less than 25 % of the new club's matches in the first season**, using only information available before the deal."
)

tab_score, tab_past, tab_about = st.tabs(["Score a transfer", "Check it on past deals", "How to read this"])

# --------------------------------------------------------------------------------------
# Tab 1: what-if
# --------------------------------------------------------------------------------------
with tab_score:
    left, right = st.columns([1, 1.15], gap="large")
    with left:
        st.subheader("The deal")
        c1, c2 = st.columns(2)
        position = c1.selectbox("Position", ["Attack", "Midfield", "Defender", "Goalkeeper"], index=1)
        age = c2.slider("Age", 16, 38, 22)
        c1, c2 = st.columns(2)
        value_m = c1.select_slider("Market value (EUR m)", options=[0.25, 0.5, 1, 2, 3, 5, 8, 12, 20, 30, 50, 80, 120], value=3)
        fee_m = c2.number_input("Fee (EUR m, 0 = free / undisclosed)", min_value=0.0, max_value=300.0, value=3.0, step=0.5)
        share = st.slider("Share of his club's matches played last season", 0, 100, 60, format="%d%%") / 100
        c1, c2 = st.columns(2)
        orig = c1.selectbox("Origin league level", [1, 2, 3, 4], index=1, help="1 = top division of a country, 2 = second division, ...")
        dest = c2.selectbox("Destination league level", [1, 2, 3, 4], index=0)
        finish = st.slider(
            "Destination club's finish last season", 0, 100, 40, format="%d%%",
            help="0 % = champion, 100 % = bottom of its league (position in the table, scaled).",
        ) / 100
        c1, c2 = st.columns(2)
        regulars = c1.slider("Established players in his position at the new club", 0, 8, 3)
        inj = c2.slider("Injury days in the last 12 months", 0, 200, 0)

    inp = dict(position=position, age=float(age), log_value=float(np.log1p(value_m * 1e6)), fee_eur=fee_m * 1e6,
               prev1_share=share, orig_level=float(orig), dest_level=float(dest), dest_rank_pct=finish,
               regulars=float(regulars), inj_days=float(inj), is_gk=float(position == "Goalkeeper"))
    risk = risk_of(inp)
    pct_below = float((test["risk"] < risk).mean())

    with right:
        st.subheader("Result")
        m1, m2, m3 = st.columns(3)
        m1.metric("Flop risk", f"{risk:.0%}", f"{(risk - BASE) * 100:+.0f} pp vs average", delta_color="inverse")
        m2.metric("Average deal (2022-24)", f"{BASE:.0%}")
        m3.metric("Riskier than", f"{pct_below:.0%} of deals")
        if risk >= FLAG_10:
            st.error(f"**Flag for a second opinion.** This is in the riskiest 10 % of 2022-24 deals (risk above {FLAG_10:.0%}).")
        else:
            st.success(f"Not in the riskiest 10 % of 2022-24 deals (that starts at {FLAG_10:.0%}).")

        # what moves this deal: put one input at a typical value and see how the risk changes
        typical = dict(inp)
        changes = {
            "Age": dict(age=26.0),
            "Position": dict(position="Midfield", is_gk=0.0),
            "Market value": dict(log_value=float(np.log1p(1e6))),
            "Fee": dict(fee_eur=0.0),
            "Last-season playing time": dict(prev1_share=0.75),
            "Tier move": dict(orig_level=float(dest)),
            "Destination club finish": dict(dest_rank_pct=0.5),
            "Established players in position": dict(regulars=3.0),
            "Injury days": dict(inj_days=0.0),
        }
        rows = []
        for label, change in changes.items():
            variant = {**typical, **change}
            rows.append({"input": label, "effect on risk (pp)": round((risk - risk_of(variant)) * 100, 1)})
        eff = pd.DataFrame(rows).sort_values("effect on risk (pp)")
        st.markdown("**What moves this deal's risk?** Percentage points of risk added (+) or removed (−) compared with a "
                    "typical value of each input, one at a time.")
        eff["direction"] = np.where(eff["effect on risk (pp)"] >= 0, "adds risk", "lowers risk")
        chart = (
            alt.Chart(eff).mark_bar(size=16)
            .encode(
                x=alt.X("effect on risk (pp):Q", title="percentage points of risk"),
                y=alt.Y("input:N", sort=alt.EncodingSortField("effect on risk (pp)", order="descending"), title=None,
                        axis=alt.Axis(labelLimit=400)),
                color=alt.Color("direction:N", legend=None,
                                scale=alt.Scale(domain=["adds risk", "lowers risk"], range=["#eb6834", "#2a78d6"])),
                tooltip=["input", "effect on risk (pp)"],
            ).properties(height=300)
        )
        st.altair_chart(chart, width="stretch")
        st.caption("Compared with: age 26, a midfielder, market value EUR 1 m, no fee, 75 % of matches played, no tier move, "
                   "a mid-table destination, 3 established players in the position, no injuries. One-at-a-time changes do "
                   "not add up exactly, because the model combines inputs. Read it as a guide.")

    st.divider()
    st.subheader("What happened to similar deals?")
    near = neighbours(inp, 200)
    near_h = hist.loc[near.index]
    a, b = st.columns([1, 2.4])
    a.metric("Flop rate of the 200 most similar past deals", f"{near['y_flop'].mean():.0%}")
    a.caption("Similar = close in age, position group, market value, last-season playing time and league levels.")
    show = near_h.sort_values("transfer_date", ascending=False).head(12)
    b.dataframe(
        pd.DataFrame({
            "player": show["player_name"], "season": show["season_y"], "from": show["from_team_name"], "to": show["to_team_name"],
            "age": show["age"].round(1), "played %": (show["share_after"] * 100).round(0), "flop": show["y_flop"].map({1: "yes", 0: "no"}),
        }),
        hide_index=True, width="stretch",
    )

# --------------------------------------------------------------------------------------
# Tab 2: past deals
# --------------------------------------------------------------------------------------
with tab_past:
    st.subheader("How did the model do on deals it never saw?")
    st.write(
        "The model was trained on transfers up to 2019 and calibrated on 2020-21. The 2022-24 deals below were used "
        "**once**, for the final test."
    )
    try:
        met = json.load(open(METRICS_PATH))
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("AUC, gradient boosting", f"{met['test_auc_gradient_boosting']:.3f}")
        k2.metric("AUC, logistic regression", f"{met['test_auc_logistic_regression']:.3f}")
        k3.metric("AUC, age rule only", f"{met['test_auc_age_rule']:.3f}")
        k4.metric("Test deals", f"{met['n_test']:,}")
    except FileNotFoundError:
        st.info("Run notebook 03 to create the metrics file.")

    f1, f2, f3 = st.columns(3)
    seasons = f1.multiselect("Season", sorted(test["season_y"].unique()), default=sorted(test["season_y"].unique()))
    q = f2.text_input("Search player or club")
    only_flag = f3.checkbox("Only the riskiest 10 %")
    view = test[test["season_y"].isin(seasons)]
    if q:
        mask = view[["player_name", "from_team_name", "to_team_name"]].apply(lambda s: s.str.contains(q, case=False, na=False)).any(axis=1)
        view = view[mask]
    if only_flag:
        view = view[view["risk"] >= FLAG_10]
    if len(view):
        st.write(f"{len(view):,} deals, of which {view['y_flop'].mean():.0%} flopped; average predicted risk {view['risk'].mean():.0%}.")
    view = view.sort_values("risk", ascending=False).head(300)
    st.dataframe(
        pd.DataFrame({
            "player": view["player_name"], "season": view["season_y"], "from": view["from_team_name"], "to": view["to_team_name"],
            "position": view["main_position"], "age": view["age"].round(1), "predicted risk": (view["risk"] * 100).round(0),
            "played % of matches": (view["share_after"] * 100).round(0), "flop": view["y_flop"].map({1: "yes", 0: "no"}),
        }),
        hide_index=True, width="stretch", height=480,
        column_config={"predicted risk": st.column_config.ProgressColumn("predicted risk %", min_value=0, max_value=100, format="%d")},
    )
    st.caption("Showing the 300 highest-risk deals of the selection. The 'flop' column is what really happened.")
    fig_path = ROOT / "reports" / "figures" / "03_calibration.png"
    if fig_path.exists():
        st.image(str(fig_path), caption="Do predicted risks match reality? Recalibrated predictions sit close to the diagonal.", width=520)

# --------------------------------------------------------------------------------------
# Tab 3: about
# --------------------------------------------------------------------------------------
with tab_about:
    st.subheader("What the number means")
    st.markdown(
        """
- **Flop** = the player played less than 25 % of the new club's league matches in the season after a permanent summer
  transfer. It does not distinguish a bad signing from an injury, a loan-like squad-depth deal or a coach change.
- **Risk %** is the model's estimated chance of that outcome. Averaged over all deals it matches the observed rate, but
  one deal's risk is an estimate, not a prediction of what will happen.
- **It is a screen, not a verdict.** Flagging the riskiest 10 % of deals catches about 28 % of all flops; 45 % of the
  flagged deals flop, against 16 % overall. Use it to decide where a second opinion is worth the time.
- **Association, not cause.** The model says what usually goes with a flop (young, big step up, strong squad, little
  recent playing time, goalkeepers), not what a club could change to avoid one.

**Scope and limits.** Permanent summer transfers 2012-2024 from one data source (Transfermarkt), market value or fee
known, moves inside one club excluded. Not meant for loans, winter windows or leagues outside the data. Fields you do not
set (height, career length, goals, ...) are filled with the typical values of similar past transfers. Weaker for
expensive deals than for cheap ones. Results are about 0.69 AUC: useful, far from perfect.
        """
    )
