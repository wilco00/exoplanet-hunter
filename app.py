"""Milestone 7: Exoplanet Hunter dashboard.

Enter a KOI, KIC or Kepler name; see what both models think, and how much that verdict
is worth given whether the models trained on it. All the science lives in inference.py;
this file is only the page.

Usage:
    streamlit run app.py
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from extract_features import vetting_figure
from figures import fold_figure, light_curve_figure
from inference import KoiResult, analyze, feature_context, find_kois, training_kois

DATASET_INFO = {  # badge color, label, what it means for trusting the verdict
    "train": ("gray", "Training set",
              "The models learned from this KOI, so a correct verdict here proves little."),
    "test": ("green", "Test set",
             "Held out from training: one of the 196 KOIs behind the reported test scores."),
    "unseen": ("violet", "Unseen",
               "A labeled KOI that wasn't among the 1,000 we sampled for training: "
               "a fresh out-of-sample check."),
    "candidate": ("yellow", "Candidate",
                  "NASA hasn't decided on this one yet, so there's no true label to compare with."),
}
FEATURE_INFO = {  # feature -> (plain name, what it tells you)
    "shape_ratio": ("Transit shape", "Edge depth ÷ center depth. Near 1 = flat-bottomed U (planet-like); "
                    "well below 1 = V, typical of a grazing eclipsing binary."),
    "duration_frac": ("Duration ÷ period", "Share of the orbit spent in transit. Large values suggest "
                      "two big stars eclipsing each other."),
    "depth_snr": ("Signal-to-noise", "How far the dip stands out from the noise."),
    "n_transits": ("Number of transits", "How many transits the data covers (mostly reflects the period)."),
    "secondary_ratio": ("Secondary ÷ primary", "Depth of any dip half an orbit later, relative to the main "
                        "one. A real dip there means the object glows: likely a star."),
    "noise_ppm": ("Noise", "Scatter of the out-of-transit data, in ppm."),
    "secondary_snr": ("Secondary significance", "Signal-to-noise of the half-orbit dip."),
    "duration_hours": ("Duration (h)", "How long each transit lasts."),
    "depth_ppm": ("Depth (ppm)", "How much light is blocked, in parts per million."),
    "odd_even_rel": ("Odd/even difference", "|odd − even depth| ÷ depth. A binary at half its true period "
                     "alternates deep and shallow eclipses."),
    "harmonic_dll": ("Sine-wave fit", "How much better a smooth sine fits than a transit. Positive = "
                     "probably stellar variability."),
    "period_days": ("Period (days)", "Time between transits."),
    "odd_even_sigma": ("Odd/even significance", "Statistical significance of the odd/even difference."),
    "max_single_frac": ("Single-event share", "Share of the signal from the strongest single transit. "
                        "Near 1 = one glitch faking a periodic signal."),
}
TOP_FEATURES = 6
MODELS = [  # (result attribute, card title, one-line description)
    ("prob_classic", "Classic", "Gradient-boosted trees on 14 vetting features"),
    ("prob_cnn", "CNN", "1D neural network on the folded light curve"),
    ("prob_ensemble", "Ensemble", "Average of the two"),
]


@st.cache_resource(show_spinner=False)
def cached_analyze(name: str) -> KoiResult:
    return analyze(name)


def verdict_card(r: KoiResult, attr: str, title: str, description: str) -> None:
    prob = getattr(r, attr)
    says_planet = prob >= 0.5
    with st.container(border=True):
        st.metric(f"{title}: P(planet)", f"{prob:.1%}")
        verdict = ":blue-badge[Planet]" if says_planet else ":orange-badge[False positive]"
        if r.label is None:
            st.markdown(verdict)
        else:
            right = says_planet == bool(r.label)
            st.markdown(f"{verdict} {'✅ matches NASA' if right else '❌ NASA disagrees'}")
        st.caption(description)  # last, so differing lengths don't misalign the numbers


def header(r: KoiResult) -> None:
    name = f"{r.kepoi_name} · {r.kepler_name}" if r.kepler_name else r.kepoi_name
    st.subheader(name)
    st.caption(f"KIC {r.kepid} · period {r.period:.4f} d · duration {r.duration * 24:.2f} h · "
               f"{r.features['n_transits']} transits · depth {r.features['depth_ppm']:.0f} ppm")

    color, label, meaning = DATASET_INFO[r.dataset]
    st.markdown(f":{color}-badge[{label}] &nbsp; NASA disposition: **{r.disposition}**")
    st.caption(meaning)

    # An unseen/candidate KOI on a training star isn't fully independent: the CNN and the trees
    # never saw this KOI, but they did see the same star's light curve (via another KOI).
    star_split = training_kois()[1].get(r.kepid)
    if r.dataset in ("unseen", "candidate") and star_split == "train":
        st.caption("Note: another KOI on this star was in the training set, "
                   "so the models have seen this star's light curve before.")

    for note in r.warnings:
        st.warning(note)


def main() -> None:
    st.set_page_config(page_title="Exoplanet Hunter", page_icon="🔭", layout="wide")
    st.title("Exoplanet Hunter")
    st.markdown("Is a Kepler transit signal a real planet or a false positive? "
                "Two models trained on 1,000 labeled signals give their verdicts.")

    query = st.text_input("KOI, KIC or Kepler name", value="K00072.01",
                          help="Examples: K00072.01, KOI-72, KIC 11904151, Kepler-10 b")
    names = find_kois(query) if query.strip() else []
    if not names:
        st.info(f"No KOI in the catalog matches '{query}'." if query.strip() else
                "Enter a KOI to begin.")
        return
    name = names[0] if len(names) == 1 else st.radio(
        f"This star has {len(names)} KOIs", names, horizontal=True)

    try:
        with st.spinner(f"Loading and cleaning {name}'s light curve (a new star downloads from MAST)..."):
            r = cached_analyze(name)
    except (LookupError, TimeoutError, ValueError, ConnectionError) as e:
        st.error(f"Couldn't analyze {name}: {e}")
        return

    header(r)
    for col, (attr, title, description) in zip(st.columns(3), MODELS):
        with col:
            verdict_card(r, attr, title, description)
    plots(r)
    why(r)
    vetting(r)


def fmt(x: float) -> str:
    if not np.isfinite(x):
        return "n/a"
    return f"{x:,.0f}" if abs(x) >= 1000 else f"{x:.3g}"


def leaning(share: float) -> str:
    if not np.isfinite(share):
        return "Not measurable"
    return "Planet-like" if share >= 0.7 else "False-positive-like" if share <= 0.3 else "Mixed"


def why(r: KoiResult) -> None:
    st.subheader("Why this verdict")
    st.caption("Each row looks at one feature on its own: among the 50 training KOIs with the closest "
               "value, what share were real planets? (The training set is about half planets, so 50% "
               "means the feature says nothing.) Rows are sorted by how much the classic model relies "
               "on each feature. The model weighs all 14 together, so this shows what the evidence "
               "says, not exactly how the model combined it.")
    ctx = feature_context(r)
    table = pd.DataFrame({
        "Feature": [FEATURE_INFO[f][0] for f in ctx["feature"]],
        "This KOI": ctx["value"].map(fmt),
        "Typical planet": ctx["planet_median"].map(fmt),
        "Typical false positive": ctx["fp_median"].map(fmt),
        "Similar KOIs that were planets": ctx["planet_share"],
        "Reads as": ctx["planet_share"].map(leaning),
        "Model reliance": ctx["importance"].clip(lower=0),
    })
    config = {
        "Similar KOIs that were planets": st.column_config.ProgressColumn(
            format="percent", min_value=0, max_value=1),
        "Model reliance": st.column_config.ProgressColumn(
            format="%.3f", min_value=0, max_value=float(table["Model reliance"].max()),
            help="Drop in test-set ROC AUC when this feature is shuffled (permutation importance)"),
    }
    st.dataframe(table.head(TOP_FEATURES), column_config=config, hide_index=True, width="stretch")
    with st.expander(f"All {len(table)} features"):
        st.dataframe(table, column_config=config, hide_index=True, width="stretch")
    with st.expander("What the features mean"):
        st.markdown("\n".join(f"- **{FEATURE_INFO[f][0]}**: {FEATURE_INFO[f][1]}" for f in ctx["feature"]))


def vetting(r: KoiResult) -> None:
    st.subheader("Vetting checks")
    st.caption("Three classic tests for impostors. Left: the transit itself. Middle: odd-numbered vs. "
               "even-numbered transits; if they differ, it's likely two stars eclipsing each other, "
               "seen at half the true period. Right: half an orbit after the transit; a dip there means "
               "the companion gives off its own light, so it's a star, not a planet.")
    fig = vetting_figure(r.t, r.y, r.period, r.t0, r.duration, r.kepoi_name)
    st.pyplot(fig)
    plt.close(fig)


def plots(r: KoiResult) -> None:
    theme = "dark" if st.context.theme.type == "dark" else "light"
    config = {"displaylogo": False}

    st.subheader("Light curve")
    note = ""
    if r.features["depth_ppm"] < r.features["noise_ppm"]:
        note = (f" This dip ({r.features['depth_ppm']:.0f} ppm) is smaller than the scatter of a single "
                f"measurement ({r.features['noise_ppm']:.0f} ppm), so it can't be seen here; "
                "the folded view below reveals it.")
    st.caption("The star's brightness over Kepler's four years, after removing slow drifts. "
               "Highlighted points fall inside this KOI's predicted transits; drag to zoom, "
               "double-click to reset." + note)
    st.plotly_chart(light_curve_figure(r, theme), theme=None, config=config, width="stretch")

    st.subheader("Folded transit")
    st.caption(f"All {r.features['n_transits']} transits stacked on top of each other using the "
               f"catalog period ({r.period:.4f} d), so the dip stands out from the noise. "
               "A planet usually gives a flat-bottomed U; a grazing eclipsing binary gives a V.")
    st.plotly_chart(fold_figure(r, theme), theme=None, config=config, width="stretch")


main()
