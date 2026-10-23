"""Compact, accessible workspace theme for the Streamlit application."""

from __future__ import annotations

import streamlit as st


APP_CSS = r"""
<style>
:root {
  --hv-ink-950: #132a31;
  --hv-ink-900: #19343b;
  --hv-ink-800: #29464c;
  --hv-ink-700: #425a5f;
  --hv-ink-600: #5c6e71;
  --hv-sidebar: #18333b;
  --hv-sidebar-raised: #23434b;
  --hv-accent-700: #2f756b;
  --hv-accent-600: #3f887c;
  --hv-accent-500: #62a99c;
  --hv-accent-100: #e5f2ee;
  --hv-success: #2f7d4a;
  --hv-warning: #9a641f;
  --hv-danger: #b44242;
  --hv-canvas: #f3f0e9;
  --hv-surface: #fffdf9;
  --hv-surface-muted: #f8f5ef;
  --hv-border: #dedbd3;
  --hv-border-strong: #cbc8c0;
  --hv-focus: #78c8ba;
  --hv-radius-sm: 8px;
  --hv-radius-md: 12px;
  --hv-radius-lg: 18px;
  --hv-shadow-1: 0 1px 2px rgba(19, 42, 49, 0.04), 0 7px 22px rgba(19, 42, 49, 0.055);
  --hv-shadow-2: 0 16px 38px rgba(19, 42, 49, 0.09);
  color-scheme: light;
}

html {
  scroll-behavior: smooth;
  overflow-x: hidden;
}

body,
[data-testid="stAppViewContainer"],
[data-testid="stSidebar"] {
  font-family: Inter, "Noto Sans SC", "Microsoft YaHei", "PingFang SC", system-ui, sans-serif;
  color: var(--hv-ink-950);
  font-size: 16px;
}

[data-testid="stAppViewContainer"] {
  background: var(--hv-canvas);
}

[data-testid="stHeader"] {
  background: rgba(243, 240, 233, 0.94);
  border-bottom: 1px solid rgba(222, 219, 211, 0.7);
  backdrop-filter: blur(10px);
}

[data-testid="stAppDeployButton"] {
  display: none;
}

.block-container {
  max-width: 1260px;
  padding: 1.35rem 1.7rem 3.5rem;
}

/* Keep the product typography deliberately narrow and predictable. */
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li {
  color: var(--hv-ink-700);
  font-size: 0.94rem;
  line-height: 1.65;
}

[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3,
[data-testid="stMarkdownContainer"] h4 {
  color: var(--hv-ink-950);
  font-weight: 700;
  letter-spacing: -0.018em;
  text-wrap: balance;
  scroll-margin-top: 5rem;
}

[data-testid="stMarkdownContainer"] h3 {
  margin: 1.2rem 0 0.6rem;
  font-size: 1.06rem;
}

[data-testid="stMarkdownContainer"] h4 {
  margin: 1rem 0 0.45rem;
  font-size: 0.96rem;
}

[data-testid="stCaptionContainer"] {
  color: var(--hv-ink-600);
  font-size: 0.78rem;
  line-height: 1.55;
  overflow-wrap: anywhere;
}

.hv-skip-link {
  position: fixed;
  top: -100px;
  left: 1rem;
  z-index: 999999;
  padding: 0.7rem 1rem;
  border-radius: 0 0 var(--hv-radius-sm) var(--hv-radius-sm);
  background: var(--hv-ink-950);
  color: #fff !important;
  font-weight: 700;
}

.hv-skip-link:focus-visible {
  top: 0;
  outline: 3px solid var(--hv-focus);
  outline-offset: 2px;
}

/* Compact product header inspired by a desktop operations workspace. */
.hv-hero {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  align-items: center;
  gap: 1.25rem;
  padding: 1.2rem 1.35rem;
  margin-bottom: 0.75rem;
  border: 1px solid var(--hv-border);
  border-radius: var(--hv-radius-lg);
  background: var(--hv-surface);
  box-shadow: var(--hv-shadow-1);
}

.hv-hero-main {
  min-width: 0;
}

.hv-hero-topline,
.hv-sidebar-brand,
.hv-mode-row,
.hv-hero-status {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 0.55rem;
}

.hv-mark {
  display: inline-grid;
  place-items: center;
  width: 2rem;
  height: 2rem;
  flex: 0 0 2rem;
  border-radius: 9px;
  background: var(--hv-accent-600);
  color: #fff;
  font-size: 0.76rem;
  font-weight: 800;
  letter-spacing: -0.03em;
}

.hv-eyebrow {
  color: var(--hv-accent-700);
  font-size: 0.75rem;
  font-weight: 750;
  letter-spacing: 0.09em;
  text-transform: uppercase;
}

.hv-hero h1 {
  max-width: 780px;
  margin: 0.5rem 0 0.25rem;
  color: var(--hv-ink-950);
  font-size: clamp(1.5rem, 2.5vw, 1.8rem);
  line-height: 1.28;
  letter-spacing: -0.03em;
  text-wrap: balance;
}

.hv-hero-copy {
  max-width: 760px;
  margin: 0;
  color: var(--hv-ink-700);
  font-size: 0.9rem;
  line-height: 1.6;
  text-wrap: pretty;
}

.hv-hero-status {
  justify-content: flex-end;
}

.hv-status-pill,
.hv-citation-pill {
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  border: 1px solid var(--hv-border);
  border-radius: 999px;
  background: var(--hv-surface-muted);
  color: var(--hv-ink-800);
  font-size: 0.76rem;
  font-weight: 700;
  line-height: 1.2;
  white-space: nowrap;
}

.hv-status-pill {
  padding: 0.46rem 0.66rem;
}

.hv-status-pill.is-ready {
  border-color: #c6ddd4;
  background: #eaf4ef;
  color: #286542;
}

.hv-dot {
  width: 0.43rem;
  height: 0.43rem;
  border-radius: 50%;
  background: var(--hv-success);
  box-shadow: 0 0 0 3px rgba(47, 125, 74, 0.12);
}

.hv-overview-head {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 1rem;
  margin: 0.85rem 0 0.55rem;
}

.hv-overview-head strong {
  color: var(--hv-ink-900);
  font-size: 0.9rem;
  font-weight: 750;
}

.hv-overview-head span {
  color: var(--hv-ink-600);
  font-size: 0.76rem;
}

.hv-stat-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 0.7rem;
  margin: 0 0 0.85rem;
}

.hv-stat-card {
  min-width: 0;
  padding: 0.78rem 0.9rem;
  border: 1px solid var(--hv-border);
  border-radius: var(--hv-radius-md);
  background: var(--hv-surface);
  box-shadow: var(--hv-shadow-1);
}

.hv-stat-label {
  display: block;
  color: var(--hv-ink-600);
  font-size: 0.76rem;
  font-weight: 650;
}

.hv-stat-value {
  display: block;
  margin: 0.18rem 0 0.08rem;
  color: var(--hv-ink-950);
  font-size: 1.45rem;
  font-weight: 760;
  letter-spacing: -0.03em;
  font-variant-numeric: tabular-nums;
}

.hv-stat-note {
  display: block;
  color: var(--hv-ink-600);
  font-size: 0.76rem;
  line-height: 1.4;
  overflow-wrap: anywhere;
}

/* Every module has one clear heading and lives in its own surface. */
.hv-section-head {
  margin: 0 0 0.9rem;
  padding-bottom: 0.8rem;
  border-bottom: 1px solid var(--hv-border);
}

.hv-section-head .hv-kicker {
  margin-bottom: 0.22rem;
  color: var(--hv-accent-700);
  font-size: 0.75rem;
  font-weight: 750;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.hv-section-head h2 {
  margin: 0;
  color: var(--hv-ink-950);
  font-size: clamp(1.18rem, 2vw, 1.34rem);
  line-height: 1.35;
  letter-spacing: -0.02em;
}

.hv-section-head p {
  max-width: 760px;
  margin: 0.28rem 0 0;
  color: var(--hv-ink-700);
  font-size: 0.86rem;
  line-height: 1.55;
}

.hv-step-strip {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 0.6rem;
  margin: 0 0 0.9rem;
}

.hv-mini-step {
  padding: 0.68rem 0.78rem;
  border: 1px solid #d9e5e1;
  border-left: 3px solid var(--hv-accent-600);
  border-radius: 0 var(--hv-radius-sm) var(--hv-radius-sm) 0;
  background: #f2f8f5;
  color: var(--hv-ink-700);
  font-size: 0.76rem;
  line-height: 1.45;
}

.hv-mini-step strong {
  display: block;
  margin-bottom: 0.1rem;
  color: var(--hv-ink-900);
  font-size: 0.8rem;
}

.hv-citation-pill {
  max-width: 100%;
  padding: 0.36rem 0.58rem;
  border-color: #c6ded8;
  background: var(--hv-accent-100);
  color: #245f57;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 0.75rem;
  white-space: normal;
  overflow-wrap: anywhere;
}

/* Dark workspace rail: clear hierarchy without low-contrast gray-on-gray. */
[data-testid="stSidebar"] {
  background: var(--hv-sidebar);
  border-right: 1px solid #294951;
}

[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
  padding-top: 1rem;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] label p {
  color: #dce8e5;
}

.hv-sidebar-brand {
  margin-bottom: 0.3rem;
}

.hv-sidebar-title {
  color: #fff;
  font-size: 1.02rem;
  font-weight: 760;
  letter-spacing: -0.015em;
}

.hv-sidebar-copy {
  margin: 0.15rem 0 1rem;
  color: #bbcfca !important;
  font-size: 0.78rem !important;
  line-height: 1.55 !important;
}

.hv-sidebar-label {
  margin: 1rem 0 0.42rem !important;
  color: #8fb3ab !important;
  font-size: 0.75rem !important;
  font-weight: 750 !important;
  letter-spacing: 0.1em;
  text-transform: uppercase;
}

[data-testid="stSidebar"] [data-testid="stTextInput"] input {
  border: 1px solid #557078;
  background: #fffdf9;
  color: var(--hv-ink-950);
}

[data-testid="stSidebar"] [data-testid="stTextInput"] input:focus {
  border-color: var(--hv-focus);
  box-shadow: 0 0 0 2px rgba(120, 200, 186, 0.2);
}

[data-testid="stSidebar"] .stButton > button {
  border-color: #507079;
  background: var(--hv-sidebar-raised);
  color: #f3f8f6;
}

[data-testid="stSidebar"] [data-testid="stBaseButton-primary"] {
  border-color: #86c6ba;
  background: #83c5b8;
  color: #102f34;
  box-shadow: none;
}

[data-testid="stSidebar"] [data-testid="stBaseButton-primary"]:hover {
  border-color: #a1d8ce;
  background: #96d2c7;
  color: #102f34;
}

.hv-mode-row {
  justify-content: space-between;
  margin: 0;
  padding: 0.62rem 0;
  border-top: 1px solid #34515a;
  color: #b9cbc7;
  font-size: 0.78rem;
}

.hv-mode-row strong {
  color: #f4f8f7;
  font-size: 0.78rem;
}

.hv-demo-route {
  margin-top: 0.7rem;
  padding: 0.78rem;
  border: 1px solid #34535b;
  border-radius: var(--hv-radius-sm);
  background: rgba(255, 255, 255, 0.045);
  color: #c9d8d4;
  font-size: 0.78rem;
  line-height: 1.65;
}

.hv-demo-route strong {
  display: block;
  margin-bottom: 0.22rem;
  color: #fff;
}

[data-testid="stSidebar"] [data-testid="stAlert"] {
  border-color: #527269;
  background: rgba(241, 251, 247, 0.1);
}

/* Top-level module navigation. */
div[data-baseweb="tab-list"],
[role="tablist"] {
  gap: 0.25rem;
  padding: 0.3rem;
  border: 1px solid var(--hv-border);
  border-radius: var(--hv-radius-md);
  background: var(--hv-surface);
  box-shadow: var(--hv-shadow-1);
  overflow-x: auto;
}

button[data-baseweb="tab"],
[data-testid="stTab"] {
  min-height: 44px;
  padding-inline: 0.82rem;
  border-radius: var(--hv-radius-sm);
  color: var(--hv-ink-700);
  font-size: 0.84rem;
  font-weight: 650;
  touch-action: manipulation;
}

button[data-baseweb="tab"]:hover,
[data-testid="stTab"]:hover {
  background: var(--hv-surface-muted);
  color: var(--hv-ink-950);
}

button[data-baseweb="tab"][aria-selected="true"],
[data-testid="stTab"][aria-selected="true"] {
  background: var(--hv-accent-700) !important;
  color: #fff !important;
}

button[data-baseweb="tab"] > div[data-testid="stMarkdownContainer"] > p,
[data-testid="stTab"] > div[data-testid="stMarkdownContainer"] > p {
  color: inherit !important;
  font-size: inherit;
  white-space: nowrap;
}

.react-aria-SelectionIndicator {
  display: none;
}

div[role="tabpanel"] {
  margin-top: 0.72rem;
  padding: 1.15rem 1.25rem 1.45rem;
  border: 1px solid var(--hv-border);
  border-radius: var(--hv-radius-lg);
  background: var(--hv-surface);
  box-shadow: var(--hv-shadow-1);
}

/* Streamlit controls and feedback share one visual language. */
.stButton > button,
.stFormSubmitButton > button,
[data-testid="stBaseButton-primary"],
[data-testid="stBaseButton-secondary"] {
  min-height: 44px;
  border-radius: var(--hv-radius-sm);
  font-size: 0.86rem;
  font-weight: 680;
  cursor: pointer;
  touch-action: manipulation;
  transition: background-color 160ms ease, border-color 160ms ease, color 160ms ease, box-shadow 160ms ease;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
  border-color: var(--hv-accent-600);
  color: var(--hv-accent-700);
  box-shadow: 0 4px 12px rgba(47, 117, 107, 0.1);
}

.stButton > button:focus-visible,
.stFormSubmitButton > button:focus-visible,
button[data-baseweb="tab"]:focus-visible,
[data-testid="stTab"]:focus-visible,
input:focus-visible,
textarea:focus-visible,
[role="slider"]:focus-visible {
  outline: 3px solid rgba(120, 200, 186, 0.55) !important;
  outline-offset: 2px;
}

[data-testid="stBaseButton-primary"] {
  border: 1px solid var(--hv-accent-700);
  background: var(--hv-accent-700);
  color: #fff;
  box-shadow: 0 5px 14px rgba(47, 117, 107, 0.16);
}

[data-testid="stBaseButton-primary"]:hover {
  border-color: #285f58;
  background: #285f58;
  color: #fff;
}

[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea,
[data-testid="stNumberInput"] input,
[data-baseweb="select"] > div {
  border-color: var(--hv-border-strong);
  border-radius: var(--hv-radius-sm);
  background: #fff;
  color: var(--hv-ink-950);
}

[data-testid="stFileUploaderDropzone"] {
  min-height: 104px;
  border: 1.5px dashed #9bbcb4;
  border-radius: var(--hv-radius-md);
  background: #f3f8f5;
}

[data-testid="stAlert"] {
  border-radius: var(--hv-radius-sm);
  border-width: 1px;
  font-size: 0.86rem;
}

[data-testid="stMetric"] {
  padding: 0.72rem 0.82rem;
  border: 1px solid var(--hv-border);
  border-radius: var(--hv-radius-md);
  background: var(--hv-surface-muted);
}

[data-testid="stMetricLabel"] p {
  color: var(--hv-ink-600);
  font-size: 0.76rem;
  font-weight: 650;
}

[data-testid="stMetricValue"] {
  color: var(--hv-ink-950);
  font-size: 1.42rem;
  font-weight: 750;
  font-variant-numeric: tabular-nums;
}

[data-testid="stDataFrame"],
[data-testid="stVegaLiteChart"] {
  overflow: hidden;
  border: 1px solid var(--hv-border);
  border-radius: var(--hv-radius-md);
  background: #fff;
}

[data-testid="stExpander"] {
  overflow: hidden;
  border: 1px solid var(--hv-border);
  border-radius: var(--hv-radius-md);
  background: var(--hv-surface-muted);
}

[data-testid="stVerticalBlockBorderWrapper"] {
  border-color: var(--hv-border) !important;
  border-radius: var(--hv-radius-md) !important;
  background: var(--hv-surface-muted);
  box-shadow: none;
}

hr {
  border-color: var(--hv-border);
}

@media (max-width: 1024px) {
  .block-container {
    padding-inline: 1.2rem;
  }

  .hv-stat-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}

@media (max-width: 768px) {
  .block-container {
    padding: 1rem 0.9rem 3rem;
  }

  .hv-hero {
    grid-template-columns: 1fr;
    gap: 0.75rem;
    padding: 1rem;
  }

  .hv-hero-status {
    justify-content: flex-start;
  }

  .hv-step-strip {
    grid-template-columns: 1fr;
  }

  div[role="tabpanel"] {
    padding: 1rem 0.9rem 1.25rem;
  }
}

@media (max-width: 520px) {
  .hv-stat-grid {
    grid-template-columns: 1fr;
  }

  .hv-overview-head {
    align-items: flex-start;
    flex-direction: column;
    gap: 0.1rem;
  }

  .hv-hero h1 {
    font-size: 1.45rem;
  }
}

@media (prefers-reduced-motion: reduce) {
  *,
  *::before,
  *::after {
    scroll-behavior: auto !important;
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important;
  }
}
</style>
"""


def inject_theme() -> None:
    """Inject the local, dependency-free visual system."""

    st.markdown(APP_CSS, unsafe_allow_html=True)
