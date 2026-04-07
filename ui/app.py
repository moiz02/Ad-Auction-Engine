"""Streamlit UI entrypoint for the ad auction demo."""

import streamlit as st

from ad_auction_engine.api_app import run_search_flow
from ad_auction_engine.config import get_settings

settings = get_settings()

st.set_page_config(page_title=settings.app_name, layout="wide")

st.title(settings.app_name)
st.caption("Local demo for retrieval, CTR prediction, ranking, and GSP pricing.")

query = st.text_input("Search query", placeholder="running shoes")
user_id = st.text_input("User ID", placeholder="123")

run_search = st.button("Run Search")

if run_search:
    if not query.strip():
        st.warning("Enter a query before running search.")
    else:
        response = run_search_flow(query.strip())
        st.success(response.message)

        if response.results:
            rows = [
                {
                    "ad_id": row.ad_id,
                    "advertiser_name": row.advertiser_name,
                    "bid_price": row.bid_price,
                    "predicted_ctr": row.predicted_ctr,
                    "quality_score": row.quality_score,
                    "ad_rank": row.ad_rank,
                    "clearing_price": row.clearing_price,
                }
                for row in response.results
            ]
            st.dataframe(rows, use_container_width=True)
        else:
            st.info("No ads were returned for this query.")

st.write({"query": query, "user_id": user_id})
