import streamlit as st
import requests
import json
import pandas as pd

st.set_page_config(
    page_title="SHL Assessment Recommender",
    page_icon="🧪",
    layout="wide"
)

st.title("SHL Assessment Recommendation System")
st.markdown("""
This system helps hiring managers find the right assessments for their job roles.
Enter a job description, natural language query, or job posting URL below.
""")

API_ENDPOINT = "https://shl-assessment-reco-21953.streamlit.app"

input_type = st.radio(
    "Select input type:",
    ["Natural Language Query", "Job Description Text", "Job Description URL"]
)

if input_type == "Natural Language Query":
    user_input = st.text_area("Enter your query:", 
                            height=150,
                            placeholder="Example: I am hiring for Java developers who can also collaborate effectively with my business teams. Looking for an assessment(s) that can be completed in 40 minutes.")
    
elif input_type == "Job Description Text":
    user_input = st.text_area("Paste job description:", 
                            height=250,
                            placeholder="Paste the full job description text here...")
    
else:  
    user_input = st.text_input("Enter job description URL:", 
                            placeholder="https://example.com/job-posting")

num_recommendations = st.slider(
    "Number of recommendations to display:", 
    min_value=1, 
    max_value=10, 
    value=5,
    help="Select how many assessment recommendations you want to see (maximum 10)"
)

col1, col2, col3 = st.columns([1, 1, 3])
with col1:
    submit_button = st.button("Get Recommendations", type="primary")
with col2:
    clear_button = st.button("Clear")

if clear_button:
    user_input = ""
    st.rerun()

def get_recommendations(query, top_k):
    try:
        payload = {
            "query": query,
            "top_k": top_k
        }
        
        response = requests.post(API_ENDPOINT, json=payload)
   
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error: API returned status code {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error connecting to the API: {str(e)}")
        return None

if submit_button and user_input:
    with st.spinner(f"Getting {num_recommendations} recommendations..."):
        result = get_recommendations(user_input, num_recommendations)
        
        if result and "recommended_assessments" in result:
            recommendations = result["recommended_assessments"]
            
            if not recommendations:
                st.warning("No matching assessments found. Try refining your query.")
            else:
                st.success(f"Found {len(recommendations)} matching assessments!")
              
                df = pd.DataFrame(recommendations)
               
                display_df = df.copy()
  
                display_df['Assessment'] = display_df.apply(
                    lambda x: f'<a href="{x["url"]}" target="_blank">{x["name"]}</a>', axis=1
                )
                
                display_columns = {
                    'adaptive_support': 'Adaptive/IRT Support',
                    'description': 'Description',
                    'duration': 'Duration (mins)',
                    'remote_support': 'Remote Testing',
                    'test_type': 'Test Type'
                }
             
                final_display = pd.DataFrame()
                final_display['Assessment'] = display_df['Assessment']
                for old_col, new_col in display_columns.items():
                    if old_col in df.columns:
                        if old_col == 'test_type':
                            # Convert list to string
                            final_display[new_col] = df[old_col].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)
                        else:
                            final_display[new_col] = df[old_col]
              
                st.markdown(final_display.to_html(escape=False, index=False), unsafe_allow_html=True)
      
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "Download Results as CSV",
                    csv,
                    "shl_recommendations.csv",
                    "text/csv",
                    key='download-csv'
                )
        else:
            st.error("Failed to get recommendations. Please try again.")
elif submit_button:
    st.warning("Please enter a query or job description first.")
    
# Footer
st.markdown("---")
st.markdown("© 2025 SHL Assessment Recommendation System")
