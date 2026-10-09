"""Milestone 7: Exoplanet Hunter dashboard.

Enter a KOI, KIC or Kepler name; see what both models think, and how much that verdict
is worth given whether the models trained on it. All the science lives in inference.py;
this file is only the page.

Usage:
    streamlit run app.py
"""
import streamlit as st

from inference import KoiResult, analyze, find_kois, training_kois

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


main()
