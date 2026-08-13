"""
app.py
Streamlit UI for StackMatch, a skill-based career path recommender.
"""

import streamlit as st
import plotly.graph_objects as go
from PIL import Image
from src.data_loader import load_skills_data
from src.recommender import build_vectorizer, recommend

favicon = Image.open("assets/icon.png")
st.set_page_config(page_title="StackMatch", page_icon=favicon, layout="wide")

LOGO_SVG = '<svg width="28" height="28" viewBox="0 0 40 40" xmlns="http://www.w3.org/2000/svg"><polygon points="20,2 35,11 35,29 20,38 5,29 5,11" fill="#14161A"/><polygon points="20,9 29,14.5 29,25.5 20,31 11,25.5 11,14.5" fill="#0E7C7B"/><circle cx="20" cy="20" r="4" fill="#FAFAF8"/></svg>'

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    #MainMenu, footer, header {visibility: hidden;}

    .navbar {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        padding: 0.4rem 0 1.4rem 0;
        border-bottom: 1px solid #E5E6EA;
        margin-bottom: 1.6rem;
    }
    .navbar-name {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        font-size: 1.25rem;
        color: #14161A;
        letter-spacing: -0.01em;
    }

    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 1.2rem;
    }
    .sidebar-brand-name {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        font-size: 1.1rem;
        color: #14161A;
    }

    .hero-empty {
        position: relative;
        height: 320px;
        border-radius: 10px;
        overflow: hidden;
        background: #FAFAF8;
        border: 1px solid #E5E6EA;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
    }
    .hero-empty::before {
        content: "";
        position: absolute;
        inset: 0;
        background:
            radial-gradient(circle at 20% 30%, rgba(14,124,123,0.10), transparent 40%),
            radial-gradient(circle at 80% 70%, rgba(14,124,123,0.08), transparent 45%);
        animation: drift 8s ease-in-out infinite alternate;
    }
    @keyframes drift {
        0% { transform: translate(0, 0) scale(1); }
        100% { transform: translate(15px, -10px) scale(1.05); }
    }
    .hero-empty-text {
        position: relative;
        z-index: 1;
        color: #8A8D96;
        font-size: 0.95rem;
        margin-top: 0.8rem;
    }
    .hero-empty-title {
        position: relative;
        z-index: 1;
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 600;
        color: #3A3D46;
        font-size: 1.05rem;
    }

    .skill-chip {
        display: inline-block;
        padding: 0.25rem 0.7rem;
        border-radius: 20px;
        font-size: 0.82rem;
        margin: 0.2rem 0.3rem 0.2rem 0;
        font-weight: 500;
    }
    .skill-chip-known { background: #E4F3F1; color: #0E7C7B; border: 1px solid #BEE0DC; }
    .skill-chip-unknown { background: #F5F5F3; color: #9A9DA6; border: 1px solid #E5E6EA; text-decoration: line-through; }

    section[data-testid="stSidebar"] {
        background-color: #F6F5F1;
        border-right: 1px solid #E5E6EA;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Navbar
# ---------------------------------------------------------------------------
st.markdown(
    f'<div class="navbar">{LOGO_SVG}<span class="navbar-name">StackMatch</span></div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        f'<div class="sidebar-brand">{LOGO_SVG}<span class="sidebar-brand-name">StackMatch</span></div>',
        unsafe_allow_html=True,
    )

    st.markdown("### About this project")
    st.write(
        "StackMatch maps your skills to real job roles using TF-IDF "
        "vectorization and cosine similarity, the same content-based "
        "filtering approach used in production recommendation systems."
    )

    st.markdown("**How it works**")
    st.markdown(
        "- Each job role is represented as a skill profile\n"
        "- Your input is vectorized into the same space\n"
        "- Cosine similarity ranks the closest role matches"
    )

    st.markdown("**Stack**")
    st.markdown("`Python` · `scikit-learn` · `pandas` · `Streamlit` · `Plotly`")

    st.markdown("**Limitations**")
    st.markdown(
        "- Fixed dataset of 18 predefined roles, not exhaustive\n"
        "- Matching is keyword-based (TF-IDF), not semantic. Synonyms "
        "(e.g. \"ML\" vs \"Machine Learning\") may not match\n"
        "- Built for demonstration, not production hiring decisions"
    )

    st.markdown("---")
    st.markdown("**Author**")
    st.write("Nida Sheraz")
    st.caption("Part of the DecodeLabs AI Engineering curriculum")

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
@st.cache_resource
def get_engine():
    df = load_skills_data()
    vectorizer, tfidf_matrix = build_vectorizer(df)
    return df, vectorizer, tfidf_matrix


df, vectorizer, tfidf_matrix = get_engine()

st.markdown("#### Find your closest matching career path")
st.caption("Enter your skills, comma-separated.")

col_input, col_button = st.columns([4, 1])
with col_input:
    skills_input = st.text_input(
        "Skills",
        placeholder="Python, Docker, Kubernetes",
        label_visibility="collapsed",
    )
with col_button:
    submitted = st.button("Match", use_container_width=True)

# ---------------------------------------------------------------------------
# Empty state vs results
# ---------------------------------------------------------------------------
if not submitted or not skills_input.strip():
    st.markdown(
        """
        <div class="hero-empty">
            <div class="hero-empty-title">No skills entered yet</div>
            <div class="hero-empty-text">Your top 3 role matches will appear here</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    user_skills = [s.strip() for s in skills_input.split(",")]
    try:
        results, unknown = recommend(user_skills, df, vectorizer, tfidf_matrix, top_n=3)
        results_sorted = results.sort_values("similarity_score", ascending=True)

        col_chart, col_donut = st.columns([3, 1])

        with col_chart:
            colors = ["#A9D8D6", "#4FA6A3", "#0E7C7B"]
            fig = go.Figure(
                go.Bar(
                    x=results_sorted["similarity_score"] * 100,
                    y=results_sorted["job_role"],
                    orientation="h",
                    marker=dict(color=colors[-len(results_sorted):]),
                    text=[f"{v:.1f}%" for v in results_sorted["similarity_score"] * 100],
                    textposition="outside",
                    hovertemplate="%{y}<br>Match: %{x:.1f}%<extra></extra>",
                )
            )
            fig.update_layout(
                height=300,
                margin=dict(l=10, r=40, t=20, b=10),
                bargap=0.4,
                xaxis=dict(title="Match score (%)", range=[0, 100], gridcolor="#EFEFEC"),
                yaxis=dict(title=None, tickfont=dict(size=13)),
                plot_bgcolor="#FAFAF8",
                paper_bgcolor="#FAFAF8",
                font=dict(family="Inter, sans-serif", color="#14161A"),
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_donut:
            top_score = results.iloc[0]["similarity_score"] * 100
            donut = go.Figure(
                go.Pie(
                    values=[top_score, 100 - top_score],
                    hole=0.72,
                    marker=dict(colors=["#0E7C7B", "#EFEFEC"]),
                    textinfo="none",
                    hoverinfo="skip",
                )
            )
            donut.update_layout(
                height=180,
                margin=dict(l=0, r=0, t=10, b=0),
                showlegend=False,
                paper_bgcolor="#FAFAF8",
                annotations=[
                    dict(
                        text=f"{top_score:.0f}%",
                        font=dict(size=20, family="Space Grotesk", color="#14161A"),
                        showarrow=False,
                    )
                ],
            )
            st.plotly_chart(donut, use_container_width=True)
            st.caption(f"Top match: {results.iloc[0]['job_role']}")

        st.markdown("##### Skills used in matching")
        chips = "".join(
            f'<span class="skill-chip skill-chip-known">{s}</span>'
            for s in user_skills if s.strip() and s.lower() not in [u.lower() for u in unknown]
        )
        chips += "".join(
            f'<span class="skill-chip skill-chip-unknown">{s}</span>' for s in unknown
        )
        st.markdown(chips, unsafe_allow_html=True)

    except ValueError as e:
        st.error(str(e))