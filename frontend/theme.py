"""Visual theme for the OpsRAG Streamlit product demo."""

APP_CSS = """
<style>
:root {
    --ops-bg: #070b14;
    --ops-surface: rgba(15, 23, 42, 0.78);
    --ops-surface-strong: #111827;
    --ops-border: rgba(148, 163, 184, 0.18);
    --ops-text: #f8fafc;
    --ops-muted: #94a3b8;
    --ops-cyan: #22d3ee;
    --ops-blue: #3b82f6;
    --ops-green: #34d399;
}

.stApp {
    background:
        radial-gradient(circle at 8% 4%, rgba(34, 211, 238, 0.11), transparent 27rem),
        radial-gradient(circle at 94% 20%, rgba(59, 130, 246, 0.12), transparent 30rem),
        linear-gradient(180deg, #070b14 0%, #090f1d 52%, #070b14 100%);
    color: var(--ops-text);
}

[data-testid="stHeader"] {
    background: rgba(7, 11, 20, 0.72);
    border-bottom: 1px solid rgba(148, 163, 184, 0.08);
    backdrop-filter: blur(16px);
}

.block-container {
    max-width: 1180px !important;
    padding-top: 2.5rem !important;
    padding-bottom: 3rem !important;
}

.ops-nav {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 3.4rem;
}

.ops-brand {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    color: var(--ops-text);
    font-size: 1.08rem;
    font-weight: 750;
    letter-spacing: -0.02em;
}

.ops-brand-mark {
    display: grid;
    place-items: center;
    width: 2.15rem;
    height: 2.15rem;
    border: 1px solid rgba(34, 211, 238, 0.48);
    border-radius: 0.72rem;
    background: linear-gradient(145deg, rgba(34, 211, 238, 0.24), rgba(59, 130, 246, 0.13));
    box-shadow: 0 0 28px rgba(34, 211, 238, 0.14);
    color: var(--ops-cyan);
}

.ops-nav-meta {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    color: var(--ops-muted);
    font-size: 0.78rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.ops-live-dot {
    width: 0.48rem;
    height: 0.48rem;
    border-radius: 999px;
    background: var(--ops-green);
    box-shadow: 0 0 13px rgba(52, 211, 153, 0.8);
}

.ops-hero {
    position: relative;
    overflow: hidden;
    padding: 0.6rem 0 3.1rem;
}

.ops-eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 0.55rem;
    margin-bottom: 1.25rem;
    padding: 0.42rem 0.72rem;
    border: 1px solid rgba(34, 211, 238, 0.2);
    border-radius: 999px;
    background: rgba(8, 145, 178, 0.08);
    color: #a5f3fc;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}

.ops-hero h1 {
    max-width: 900px;
    margin: 0;
    color: var(--ops-text);
    font-size: clamp(3rem, 7vw, 5.7rem);
    font-weight: 760;
    line-height: 0.99;
    letter-spacing: -0.065em;
}

.ops-gradient-text {
    color: transparent;
    background: linear-gradient(92deg, #67e8f9 0%, #60a5fa 54%, #a78bfa 100%);
    -webkit-background-clip: text;
    background-clip: text;
}

.ops-hero-copy {
    max-width: 690px;
    margin: 1.45rem 0 1.65rem;
    color: #a8b4c7;
    font-size: 1.08rem;
    line-height: 1.75;
}

.ops-proof-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.7rem;
}

.ops-proof {
    display: inline-flex;
    align-items: center;
    gap: 0.42rem;
    padding: 0.48rem 0.68rem;
    border: 1px solid rgba(148, 163, 184, 0.14);
    border-radius: 0.62rem;
    background: rgba(15, 23, 42, 0.52);
    color: #cbd5e1;
    font-size: 0.77rem;
}

.ops-check {
    color: var(--ops-green);
    font-weight: 800;
}

.ops-section-kicker {
    margin-bottom: 0.25rem;
    color: var(--ops-cyan);
    font-size: 0.7rem;
    font-weight: 750;
    letter-spacing: 0.14em;
    text-transform: uppercase;
}

.ops-section-title {
    margin: 0 0 0.25rem;
    color: var(--ops-text);
    font-size: 1.65rem;
    font-weight: 700;
    letter-spacing: -0.035em;
}

.ops-section-copy {
    margin: 0 0 1.1rem;
    color: var(--ops-muted);
    font-size: 0.9rem;
}

div[data-testid="stForm"],
div[data-testid="stVerticalBlockBorderWrapper"] {
    border-color: var(--ops-border) !important;
    border-radius: 1.15rem !important;
    background: linear-gradient(150deg, rgba(17, 24, 39, 0.86), rgba(15, 23, 42, 0.66)) !important;
    box-shadow: 0 24px 75px rgba(0, 0, 0, 0.2);
}

div[data-testid="stForm"] {
    padding: 0.4rem !important;
}

[data-testid="stTextArea"] textarea {
    min-height: 142px;
    border: 1px solid rgba(148, 163, 184, 0.18) !important;
    border-radius: 0.85rem !important;
    background: rgba(2, 6, 23, 0.58) !important;
    color: var(--ops-text) !important;
    font-size: 0.98rem !important;
    line-height: 1.65 !important;
}

[data-testid="stTextArea"] textarea:focus {
    border-color: rgba(34, 211, 238, 0.62) !important;
    box-shadow: 0 0 0 3px rgba(34, 211, 238, 0.09) !important;
}

[data-testid="stFormSubmitButton"] button {
    min-height: 3.2rem;
    border: 0 !important;
    border-radius: 0.82rem !important;
    background: linear-gradient(100deg, #0891b2, #2563eb) !important;
    color: white !important;
    font-weight: 720 !important;
    box-shadow: 0 14px 34px rgba(37, 99, 235, 0.23);
    transition: transform 160ms ease, box-shadow 160ms ease;
}

[data-testid="stFormSubmitButton"] button:hover {
    transform: translateY(-1px);
    box-shadow: 0 17px 38px rgba(37, 99, 235, 0.32);
}

div[data-testid="stButton"] button {
    min-height: 3.2rem;
    border-color: rgba(148, 163, 184, 0.15) !important;
    border-radius: 0.78rem !important;
    background: rgba(15, 23, 42, 0.55) !important;
    color: #cbd5e1 !important;
    font-size: 0.8rem !important;
    text-align: left !important;
}

div[data-testid="stButton"] button:hover {
    border-color: rgba(34, 211, 238, 0.38) !important;
    color: #ecfeff !important;
    background: rgba(8, 145, 178, 0.1) !important;
}

.ops-flow-card {
    height: 100%;
    padding: 1.45rem;
    border: 1px solid var(--ops-border);
    border-radius: 1.15rem;
    background: linear-gradient(160deg, rgba(15, 23, 42, 0.76), rgba(10, 15, 28, 0.72));
    box-shadow: 0 24px 75px rgba(0, 0, 0, 0.18);
}

.ops-flow-title {
    margin-bottom: 1.2rem;
    color: #e2e8f0;
    font-size: 0.78rem;
    font-weight: 720;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

.ops-flow-step {
    display: grid;
    grid-template-columns: 2rem 1fr;
    gap: 0.65rem;
    padding: 0.6rem 0;
}

.ops-flow-number {
    display: grid;
    place-items: center;
    width: 1.65rem;
    height: 1.65rem;
    border: 1px solid rgba(34, 211, 238, 0.2);
    border-radius: 0.5rem;
    background: rgba(34, 211, 238, 0.07);
    color: var(--ops-cyan);
    font-size: 0.68rem;
    font-weight: 750;
}

.ops-flow-step strong {
    display: block;
    margin-bottom: 0.16rem;
    color: #e2e8f0;
    font-size: 0.86rem;
}

.ops-flow-step span {
    color: var(--ops-muted);
    font-size: 0.73rem;
    line-height: 1.5;
}

[data-testid="stMetric"] {
    padding: 1rem;
    border: 1px solid rgba(148, 163, 184, 0.13);
    border-radius: 0.85rem;
    background: rgba(2, 6, 23, 0.34);
}

[data-testid="stMetricLabel"] {
    color: var(--ops-muted);
}

[data-testid="stMetricValue"] {
    color: var(--ops-text);
    font-size: 1.35rem;
}

[data-testid="stExpander"] {
    margin-bottom: 0.55rem;
    border: 1px solid rgba(148, 163, 184, 0.14) !important;
    border-radius: 0.82rem !important;
    background: rgba(2, 6, 23, 0.3) !important;
}

[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, var(--ops-cyan), var(--ops-blue));
}

.ops-example-label {
    margin: 1.15rem 0 0.55rem;
    color: #64748b;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

.ops-footer {
    display: flex;
    justify-content: space-between;
    margin-top: 4rem;
    padding-top: 1.25rem;
    border-top: 1px solid rgba(148, 163, 184, 0.1);
    color: #64748b;
    font-size: 0.73rem;
}

@media (max-width: 720px) {
    .block-container {
        padding: 1.4rem 1rem 2rem !important;
    }

    .ops-nav {
        margin-bottom: 2.3rem;
    }

    .ops-nav-meta {
        display: none;
    }

    .ops-hero h1 {
        font-size: 3.1rem;
    }

    .ops-hero-copy {
        font-size: 0.96rem;
    }

    .ops-footer {
        flex-direction: column;
        gap: 0.45rem;
    }
}
</style>
"""
