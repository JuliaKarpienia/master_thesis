"""
Retail Recommendation System – Streamlit Demo App
Run with:  streamlit run app.py
"""

import os
import streamlit as st
import pandas as pd

# ── Page configuration ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Retail Recommender",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "preprocessed")


# Data loading 
@st.cache_data
def load_data():
    df = pd.read_csv(os.path.join(DATA_DIR, "cleaned_retail_data_2022_2025.csv"))
    rfm = pd.read_csv(os.path.join(DATA_DIR, "rfm_base.csv"), index_col="CustomerName")
    b2b_rules = pd.read_csv(os.path.join(DATA_DIR, "b2b_association_rules.csv"))
    b2c_rules = pd.read_csv(os.path.join(DATA_DIR, "b2c_association_rules.csv"))

    sim_path = os.path.join(DATA_DIR, "item_similarity_matrix.pkl")
    item_similarity_df = (
        pd.read_pickle(sim_path) if os.path.exists(sim_path) else pd.DataFrame()
    )

    popularity = (
        df.groupby("ProductName")["InvoiceID"]
        .nunique()
        .reset_index()
        .rename(columns={"InvoiceID": "PurchaseFrequency"})
        .sort_values("PurchaseFrequency", ascending=False)
    )

    return df, rfm, b2b_rules, b2c_rules, item_similarity_df, popularity


#  Recommendation engine 
def get_hybrid_recommendations(
    customer_name, current_basket, rfm, b2b_rules, b2c_rules, item_similarity_df, popularity
):
    """
    Hybrid engine: CF (B2B only) + Association Rules + Popularity fallback.
    Mirrors the logic from '3. Final Model.ipynb'.
    """
    is_b2c = customer_name == "SPRZEDAZ FISKALNA" or customer_name not in rfm.index

    rfm_profile = {}
    if not is_b2c and customer_name in rfm.index:
        r = rfm.loc[customer_name]
        rfm_profile = {
            "Recency": int(r["Recency"]),
            "Frequency": int(r["Frequency"]),
            "Monetary": round(float(r["Monetary"]), 2),
            "Segment": str(r.get("SegmentLabel", "Unknown")),
        }

    recommendations = {
        "Customer": customer_name,
        "Type": "B2C (Anonymous/New)" if is_b2c else "B2B (Known)",
        "RFM_Profile": rfm_profile,
        "Current_Basket": current_basket,
        "CF_Recs": [],
        "AR_Recs": [],
        "Popularity_Fallback": [],
    }

    # A. Collaborative Filtering – known B2B only
    if not is_b2c and not item_similarity_df.empty:
        cf_candidates = {}
        for item in current_basket:
            if item in item_similarity_df.columns:
                sims = item_similarity_df[item].sort_values(ascending=False).drop(item)
                for rec_item, score in sims.items():
                    if score > 0 and rec_item not in current_basket:
                        if rec_item not in cf_candidates or score > cf_candidates[rec_item]["Score"]:
                            cf_candidates[rec_item] = {
                                "Item": rec_item,
                                "Score": round(score, 2),
                                "Based_On": item,
                            }
        recommendations["CF_Recs"] = sorted(
            cf_candidates.values(), key=lambda x: x["Score"], reverse=True
        )[:3]

    # B. Association Rules
    rules_to_use = b2c_rules if is_b2c else b2b_rules
    if not rules_to_use.empty:
        ar_candidates = {}
        for item in current_basket:
            matches = rules_to_use[rules_to_use["antecedents"] == item]
            for _, row in matches.iterrows():
                rec_item = row["consequents"]
                if rec_item not in current_basket and rec_item not in ar_candidates:
                    ar_candidates[rec_item] = {
                        "Item": rec_item,
                        "Lift": round(row["lift"], 2),
                        "Confidence": round(row["confidence"], 2),
                        "Based_On": item,
                    }
        recommendations["AR_Recs"] = sorted(
            ar_candidates.values(), key=lambda x: x["Lift"], reverse=True
        )[:3]

    # C. Popularity fallback
    if not recommendations["CF_Recs"] and not recommendations["AR_Recs"]:
        top_items = [p for p in popularity["ProductName"].tolist() if p not in current_basket]
        recommendations["Popularity_Fallback"] = top_items[:3]

    return recommendations


# LLM helpers 
def generate_llm_prompt(hybrid_output):
    rfm_profile = hybrid_output.get("RFM_Profile", {})
    rfm_context = ""
    if rfm_profile:
        rfm_context = (
            f"- Customer History: Last purchase {rfm_profile['Recency']} days ago, "
            f"{rfm_profile['Frequency']} total invoices, "
            f"lifetime spend {rfm_profile['Monetary']:.0f} PLN. "
            f"Segment: {rfm_profile['Segment']}.\n"
        )

    prompt = (
        "You are an expert retail marketing assistant. Your job is to translate raw algorithmic "
        "recommendation data into a simple, actionable explanation for a store manager.\n\n"
        "Here is the data for the current customer interaction:\n"
        f"- Customer Profile: {hybrid_output['Type']}\n"
        f"- Customer Name/ID: {hybrid_output['Customer']}\n"
        f"{rfm_context}"
        f"- Items currently in their cart: {', '.join(hybrid_output['Current_Basket'])}\n\n"
        "Algorithmic Recommendations:"
    )

    if hybrid_output["AR_Recs"]:
        prompt += "\n1. Cross-Selling (Market Basket Analysis):\n"
        for rec in hybrid_output["AR_Recs"]:
            prompt += (
                f"   - Suggest '{rec['Item']}' because they are buying '{rec['Based_On']}'. "
                f"(Customers buying this combination are {rec['Lift']}x more likely than average).\n"
            )

    if hybrid_output["CF_Recs"]:
        prompt += "\n2. Contractor Purchase Patterns (Collaborative Filtering):\n"
        for rec in hybrid_output["CF_Recs"]:
            prompt += (
                f"   - Suggest '{rec['Item']}' because similar contractors who buy "
                f"'{rec['Based_On']}' also frequently buy this (similarity: {rec['Score']}).\n"
            )

    if hybrid_output["Popularity_Fallback"]:
        prompt += "\n3. Store Best-Sellers (Fallback):\n"
        prompt += (
            f"   - No specific signals found. Recommend best-sellers: "
            f"{', '.join(hybrid_output['Popularity_Fallback'])}.\n"
        )

    prompt += (
        "\nTask: Write a short, structured summary for the floor manager using this format:\n"
        "1. One sentence about who this customer is.\n"
        "2. A short bullet list of recommended products — each bullet has the "
        "product name in **bold** and a simple one-line reason (e.g. 'often bought together with X').\n"
        "3. One sentence telling the manager what to do next.\n\n"
        "RULES:\n"
        "- Keep the ENTIRE response under 150 words.\n"
        "- Use simple, everyday language a shop assistant would understand.\n"
        "- Do NOT mention numbers, scores, statistics, percentages, or technical terms.\n"
        "- Do NOT explain how the recommendations were generated.\n"
        "- Use markdown formatting."
        # "-Write ENTIRELY in Polish, as the store is located in Poland and the staff only speaks Polish."
    )
    return prompt


def call_llm(prompt):
    try:
        from openai import OpenAI  

        client = OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama",
        )
        response = client.chat.completions.create(
            model="mistral",
            # model='llama3.1:8b', 
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=400,
        )
        return response.choices[0].message.content
    except Exception as exc:  
        return f"LLM call failed: {exc}"


# ── Bootstrap ──────────────────────────────────────────────────────────────────
df, rfm, b2b_rules, b2c_rules, item_similarity_df, popularity = load_data()

all_products = sorted(df["ProductName"].dropna().unique().tolist())
known_customers = sorted(rfm.index.tolist())

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Retail Recommender")
    st.caption("Paint & hardware store · hybrid engine")
    st.divider()

    customer_mode = st.radio(
        "Customer type",
        ["B2B – Known Contractor", "B2C – Anonymous Walk-in", "New Customer"],
        help=(
            "B2B: existing contractor with purchase history.  \n"
            "B2C: anonymous POS walk-in.  \n"
            "New Customer: first-time visitor — gets best-seller / cross-sell recommendations."
        ),
    )

    if customer_mode == "B2B – Known Contractor":
        customer_name = st.selectbox(
            "Select customer",
            known_customers,
            help="Contractors recorded in the RFM database.",
        )
    elif customer_mode == "New Customer":
        typed_name = st.text_input(
            "Customer name (optional)",
            placeholder="e.g. Firma Budowlana XYZ",
            help="Enter any name or leave blank. No purchase history exists yet.",
        )
        customer_name = typed_name.strip() if typed_name.strip() else "NEW_CUSTOMER"
        st.info(
            "No purchase history available for this customer.  \n"
            "Recommendations are based on **cross-selling rules** and **store best-sellers**."
        )
    else:
        customer_name = "SPRZEDAZ FISKALNA"
        st.info("Anonymous POS transaction — using B2C rules only.")

    st.divider()
    st.subheader("Current basket")
    basket = st.multiselect(
        "Add products",
        options=all_products,
        placeholder="Search for a product…",
        help="Type to search. Add every item the customer is currently buying.",
    )

    if item_similarity_df.empty:
        st.warning(
            "Item similarity matrix not found.  \nRun **notebook 2** to generate it.  \n"
            "CF recommendations will be skipped.",
        )

# ── Main ───────────────────────────────────────────────────────────────────────
st.title("Retail Recommendation Engine")
st.caption(
    "Hybrid system: Collaborative Filtering · Association Rules · Popularity Fallback · Local LLM (Ollama)"
)

if not basket:
    st.info("Add products to the basket in the sidebar to see recommendations.")
    st.stop()

# Run engine
results = get_hybrid_recommendations(
    customer_name, basket, rfm, b2b_rules, b2c_rules, item_similarity_df, popularity
)

is_b2b = results["Type"].startswith("B2B")
is_new = customer_mode == "New Customer"

# ── Customer header ────────────────────────────────────────────────────────────
st.divider()
col_name, col_type, col_size = st.columns([4, 2, 1])
display_name = customer_name if customer_name != "NEW_CUSTOMER" else "New Customer"
col_name.subheader(display_name)
col_type.markdown(f"**Type:** {results['Type']}")
col_size.metric("Basket items", len(basket))

# RFM card (B2B only)
if is_b2b and results["RFM_Profile"]:
    rp = results["RFM_Profile"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Recency", f"{rp['Recency']} days", help="Days since last purchase")
    m2.metric("Frequency", f"{rp['Frequency']} invoices", help="Total invoices in history")
    m3.metric("Lifetime spend", f"{rp['Monetary']:,.0f} PLN")
    m4.metric("Segment", rp["Segment"])

# ── AI Manager Summary ─────────────────────────────────────────────────────────
st.divider()
st.subheader("AI Manager Summary")

prompt = generate_llm_prompt(results)

if st.button("Generate explanation", type="primary"):
    with st.spinner("Calling local LLM (Ollama)…"):
        explanation = call_llm(prompt)
    st.markdown(explanation)

with st.expander("View raw LLM prompt"):
    st.code(prompt, language="text")

# ── Recommendation tabs ────────────────────────────────────────────────────────
st.divider()
st.subheader("Recommendations")

has_ar = bool(results["AR_Recs"])
has_cf = bool(results["CF_Recs"])
has_pop = bool(results["Popularity_Fallback"])

tab_labels = []
if has_ar:
    tab_labels.append("Market Basket")
if has_cf:
    tab_labels.append("Similar Contractors")
if has_pop:
    tab_labels.append("Store Best-Sellers")

if not tab_labels:
    st.warning("No recommendations could be generated for this basket.")
else:
    tabs = st.tabs(tab_labels)
    tab_idx = 0

    if has_ar:
        with tabs[tab_idx]:
            st.markdown(
                "Items frequently bought *together* with the selected products "
                "(based on Market Basket / Apriori analysis)."
            )
            for rec in results["AR_Recs"]:
                with st.container(border=True):
                    left, right = st.columns([5, 2])
                    left.markdown(
                        f"**{rec['Item']}**  \n"
                        f"*Triggered by:* `{rec['Based_On']}`"
                    )
                    right.metric("Lift", rec["Lift"], help="How much more likely than random")
                    right.metric("Confidence", f"{rec['Confidence']:.0%}", help="Rule reliability")
        tab_idx += 1

    if has_cf:
        with tabs[tab_idx]:
            st.markdown(
                "Items typically purchased by **similar B2B contractors** "
                "(based on purchase-history cosine similarity)."
            )
            for rec in results["CF_Recs"]:
                with st.container(border=True):
                    left, right = st.columns([5, 2])
                    left.markdown(
                        f"**{rec['Item']}**  \n"
                        f"*Similar contractors who buy* `{rec['Based_On']}` *also buy this.*"
                    )
                    right.metric("Similarity", rec["Score"], help="Cosine similarity score (0–1)")
        tab_idx += 1

    if has_pop:
        with tabs[tab_idx]:
            st.markdown(
                "No basket-specific signals were found — showing the **store's top sellers** "
                "as a safe fallback."
            )
            for i, item in enumerate(results["Popularity_Fallback"], 1):
                st.markdown(f"{i}. **{item}**")


