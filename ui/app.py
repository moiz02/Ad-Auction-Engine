"""Streamlit UI entrypoint for the ad auction demo."""

import streamlit as st

from ad_auction_engine.config import get_settings

settings = get_settings()

st.set_page_config(page_title=settings.app_name, layout="wide")

st.title(settings.app_name)
st.caption("Phase 1 scaffold: API and UI surfaces are live before engine logic is wired.")

query = st.text_input("Search query", placeholder="running shoes")
user_id = st.text_input("User ID", placeholder="123")

st.button("Run Search", disabled=True)

st.info(
    "The interactive search flow will be connected in a later phase. "
    "This UI exists now to prove the repository shape and app surfaces."
)

st.write(
    {
        "query": query,
        "user_id": user_id,
        "status": "pending_phase_2",
    }
)
