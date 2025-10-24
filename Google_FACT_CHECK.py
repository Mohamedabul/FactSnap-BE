import streamlit as st
import requests

# Your API Key
API_KEY = "AIzaSyAoNjnqyIWAZS7oJ6aRwoDfLOak1fEA2qI"
BASE_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"

st.set_page_config(page_title="Fact Check App", layout="wide")

st.title("🔎 Google Fact Check Explorer")
st.write("Enter a statement or query to check if there are any fact-check articles published by trusted sources.")

# User input
query = st.text_input("Enter a statement to fact-check:")

if st.button("Check Fact"):
    if not query.strip():
        st.warning("Please enter a query first.")
    else:
        # Send request to Fact Check Tools API
        params = {
            "query": query,
            "languageCode": "en",
            "key": API_KEY
        }
        response = requests.get(BASE_URL, params=params)

        if response.status_code == 200:
            data = response.json()
            
            if "claims" in data and data["claims"]:
                st.subheader("✅ Fact-check articles found:")
                for claim in data["claims"]:
                    st.markdown(f"**Claim:** {claim.get('text', 'N/A')}")
                    if "claimReview" in claim:
                        for review in claim["claimReview"]:
                            publisher = review.get("publisher", {}).get("name", "Unknown")
                            rating = review.get("textualRating", "No rating")
                            url = review.get("url", "#")
                            st.write(f"- **Source:** {publisher}")
                            st.write(f"- **Rating:** {rating}")
                            st.write(f"- [Read more]({url})")
                            st.markdown("---")
            else:
                st.info("No fact-check articles found for this query.")
        else:
            st.error(f"API request failed: {response.status_code}")
