import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pycountry_convert as pc 
import os
import matplotlib
# Use the 'Agg' backend for better compatibility with Streamlit
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. CONFIGURATION AND AESTHETIC SETUP (Custom KPI Style) ---

# Custom CSS for the stylish metrics and dark aesthetic
st.markdown("""
<style>
    /* 1. Overall Page Config and Fonts */
    .stApp {
        background-color: #121212; /* Very dark background */
        color: #E0E0E0; /* Light gray text */
        font-family: 'Inter', sans-serif;
    }
    
    /* 2. Main Title/Header Style */
    .main-header {
        font-size: 3.2rem;
        font-weight: 800;
        color: #00C4CC; /* Teal/Aqua color for branding */
        letter-spacing: 1px;
        text-align: center;
        padding-top: 10px;
        padding-bottom: 20px;
    }
    
    /* 3. Custom Metric Card Styling (Stylish KPIs) */
    .metric-card {
        padding: 20px;
        border-radius: 12px;
        background-color: #1E1E1E; /* Slightly lighter card background */
        border: 1px solid #333333; /* Subtle border */
        box-shadow: 0 6px 15px rgba(0, 196, 204, 0.1); /* Subtle teal shadow */
        text-align: left;
        margin-bottom: 25px;
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-5px); /* Lift on hover */
    }
    .metric-label {
        font-size: 0.9rem;
        color: #AAAAAA; /* Gray label */
        margin-bottom: 8px;
        font-weight: 500;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #FFFFFF; /* Bright white value */
    }
    .metric-subtext {
        font-size: 0.8rem;
        color: #00C4CC; /* Teal subtext for highlights */
    }
    .ai-explanation-box {
        background-color: #1E1E1E;
        padding: 20px;
        border-radius: 8px;
        border-left: 5px solid #00C4CC;
        margin-top: 20px;
    }
    .ai-explanation-box p {
        color: #E0E0E0;
    }

    /* 4. Sidebar and Headings */
    div[data-testid="stSidebar"] {
        background-color: #1E1E1E; 
    }
    h2, h3 {
        color: #E0E0E0 !important;
        border-bottom: 2px solid #333333;
        padding-bottom: 5px;
    }
    /* Set matplotlib figure background and text for dark theme */
    figure {
        background-color: #121212 !important; /* Figure background */
    }
    .stPlotlyChart {
        background-color: #121212;
    }
</style>
""", unsafe_allow_html=True)

# Set page configuration
st.set_page_config(
    page_title="Data Science Salary Analysis & Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. DATA AND UTILITY FUNCTIONS ---

def country_code_to_name(code):
    """Converts a 2-letter ISO country code to its full name."""
    try:
        return pc.country_alpha2_to_country_name(code)
    except KeyError:
        return code

@st.cache_data
def load_data():
    """Loads the dataset and performs initial cleaning, standardization, and country mapping."""
    try:
        df = pd.read_csv('Data Science Job Salaries.csv')
    except FileNotFoundError:
        st.error("Error: 'Data Science Job Salaries.csv' not found. Please ensure the file is in the same directory.")
        return pd.DataFrame() 

    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
        
    df.rename(columns={'salary_in_usd': 'Salary (USD)'}, inplace=True)
    
    # Standardize/Rename categorical values
    df['experience_level'] = df['experience_level'].replace({
        'EN': 'Entry-level (Junior)', 'MI': 'Mid-level (Intermediate)', 
        'SE': 'Senior-level (Expert)', 'EX': 'Executive (Director)'
    })
    df['employment_type'] = df['employment_type'].replace({
        'FT': 'Full-time', 'PT': 'Part-time', 
        'CT': 'Contract', 'FL': 'Freelance'
    })
    df['remote_ratio_label'] = df['remote_ratio'].map({
        0: 'On-Site', 50: 'Hybrid', 100: 'Fully Remote'
    })
    
    # Map country codes to full names
    df['Employee Residence'] = df['employee_residence'].apply(country_code_to_name)
    df['Company Location'] = df['company_location'].apply(country_code_to_name)
    
    return df

df = load_data()

# Custom function to generate the stylish metric card HTML
def create_metric_card(label, value, subtext=""):
    """Generates the custom HTML structure for a metric card."""
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-subtext">{subtext}</div>
    </div>
    """
    
# Function to generate the AI explanation
def generate_explanation(df, input_experience, input_residence, base_salary, location_mean, final_prediction):
    """Generates a dynamic explanation for the simulated prediction."""
    
    # 1. Experience Level Analysis
    exp_comparison = df[df['experience_level'] == input_experience]['Salary (USD)'].mean().round(0)
    
    if exp_comparison > base_salary:
        exp_text = f"Your **{input_experience}** status places your salary significantly **above** the global average of ${base_salary:,.0f} (the average for this level is ${exp_comparison:,.0f})."
    else:
        exp_text = f"As an **{input_experience}** professional, your predicted salary aligns closely with your peer group, where the average salary is around ${exp_comparison:,.0f}."
        
    # 2. Location Analysis
    if not pd.isna(location_mean):
        residence_comparison = location_mean.round(0)
        if residence_comparison > base_salary:
            loc_text = f"The **{input_residence}** market strongly influenced this prediction, as the historical average salary for this location is **${residence_comparison:,.0f}**, which is well above the overall mean."
        elif residence_comparison < base_salary:
            loc_text = f"The **{input_residence}** market slightly moderated the prediction. The historical average salary in this region is typically **${residence_comparison:,.0f}**, which is below the overall mean."
        else:
            loc_text = f"Your **{input_residence}** residence is a key factor, with historical data aligning closely with the global average salary."
    else:
        loc_text = "Data for your specific residence was limited, so the model relied more heavily on global trends and other selected features."

    # 3. Final Summary
    summary_text = f"The final prediction of **${final_prediction:,.0f}** is an extrapolation based on these weighted factors, ensuring the estimate reflects both your individual career standing and market realities."

    return f"""
    <div class="ai-explanation-box">
        <p style="font-weight: 700; color: #00C4CC;">Model Feature Analysis:</p>
        <p>🌐 **Global Baseline:** The prediction started from the global average data science salary (approx. ${base_salary:,.0f} USD).</p>
        <p>💼 **Experience Impact:** {exp_text}</p>
        <p>📍 **Geographic Weight:** {loc_text}</p>
        <p>✨ **Final Estimate:** {summary_text}</p>
    </div>
    """

# --- 3. SIDEBAR NAVIGATION ---
st.sidebar.markdown(f'<h1 style="color:#00C4CC;">DS SALARY </h1>', unsafe_allow_html=True)
st.sidebar.markdown("### Job Salaries Analysis")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "NAVIGATION", 
    ["1. Executive Summary 🚀", "2. Interactive Dashboard 📊", "3. Salary Predictor 🧠"]
)
st.sidebar.markdown("---")



# --- 4. MAIN PAGE CONTENT ---

if df.empty:
    st.stop() 

# ====================================================================
# 1. EXECUTIVE SUMMARY (HOME PAGE)
# ====================================================================
if page == "1. Executive Summary 🚀":
    st.markdown('<div class="main-header">DS Salary Analysis</div>', unsafe_allow_html=True)
    
    st.markdown(f"""
        <div style="text-align: center; color: #AAAAAA; font-size: 1.1em; margin-bottom: 40px;">
            A comprehensive look at global Data Science compensation and growth from 2020 to 2022.
        </div>
    """, unsafe_allow_html=True)
    
    # --- Key Insights Card Section (Stylish KPIs) ---
    avg_salary = df['Salary (USD)'].mean().round(0)
    max_salary = df['Salary (USD)'].max().round(0)
    top_country = df.groupby('Employee Residence')['Salary (USD)'].mean().nlargest(1).index[0]
    total_entries = df.shape[0]

    st.subheader("Key Performance Indicators (KPIs)")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(create_metric_card("Global Mean Salary", f"${avg_salary:,.0f}", "Across 607 entries"), unsafe_allow_html=True)
    with col2:
        st.markdown(create_metric_card("Highest Salary Recorded", f"${max_salary:,.0f}", "Top of the market"), unsafe_allow_html=True)
    with col3:
        st.markdown(create_metric_card("Highest Paid Residence", top_country, "Mean Salary Winner"), unsafe_allow_html=True)
    with col4:
        st.markdown(create_metric_card("Total Records Analyzed", f"{total_entries:,}", "Data completeness (2020-2022)"), unsafe_allow_html=True)

    st.markdown("---")

    st.subheader("Core Insights from Analysis")
    
    # Set Matplotlib/Seaborn style for dark theme contrast
    sns.set_theme(style="darkgrid", rc={'text.color': 'white', 'axes.labelcolor': 'white'})
    plt.rcParams['figure.facecolor'] = '#121212'
    plt.rcParams['axes.facecolor'] = '#1E1E1E'
    plt.rcParams['xtick.color'] = 'white'
    plt.rcParams['ytick.color'] = 'white'
    plt.rcParams['axes.edgecolor'] = '#333333'

    # --- New 2x2 Layout for Graphs ---
    col_a, col_b = st.columns(2)
    col_c, col_d = st.columns(2)

    # 1. Distribution plot of data science salaries in USD
    with col_a:
        st.markdown("#### 1. Salary Distribution")
        plt.figure(figsize=(7, 4))
        sns.histplot(df['Salary (USD)'], kde=True, color='#00C4CC')
        plt.title('Distribution of Salaries (USD)', color='white', fontsize=12)
        plt.xlabel('Salary (USD)', color='white')
        plt.ylabel('Count', color='white')
        plt.ticklabel_format(style='plain', axis='x')
        plt.tight_layout()
        st.pyplot(plt)
        plt.close()

    # 2. Top Job Titles (Frequency)
    with col_b:
        st.markdown("#### 2. Top Job Titles (Frequency)")
        top_job_titles = df['job_title'].value_counts().nlargest(10).index
        
        plt.figure(figsize=(7, 4))
        sns.countplot(
            data=df, 
            y='job_title', 
            order=top_job_titles,
            palette='viridis'
        )
        plt.title('Top 10 Most Frequent Job Titles', color='white', fontsize=12)
        plt.xlabel('Job Count', color='white')
        plt.ylabel('Job Title', color='white')
        plt.tight_layout()
        st.pyplot(plt)
        plt.close()

    # 3. Distribution of salary by work year (COLOR CHANGED HERE)
    with col_c:
        st.markdown("#### 3. Salary by Work Year")
        plt.figure(figsize=(7, 4))
        sns.boxplot(
            data=df, 
            x='work_year', 
            y='Salary (USD)', 
            # Changed palette from 'cividis' to 'plasma'
            palette='plasma' 
        )
        plt.title('Salary Distribution by Work Year', color='white', fontsize=12)
        plt.xlabel('Work Year', color='white')
        plt.ylabel('Salary (USD)', color='white')
        plt.tight_layout()
        st.pyplot(plt)
        plt.close()

    # 4. Highest Salaries by Job titles in Data Science
    with col_d:
        st.markdown("#### 4. Highest Paid Job Titles")
        top_salaries = df.groupby('job_title')['Salary (USD)'].mean().nlargest(10).sort_values(ascending=False).reset_index()
        
        plt.figure(figsize=(7, 4))
        sns.barplot(
            data=top_salaries, 
            x='Salary (USD)', 
            y='job_title',
            palette='magma'
        )
        plt.title('Top 10 Average Salaries by Job Title', color='white', fontsize=12)
        plt.xlabel('Mean Salary (USD)', color='white')
        plt.ylabel('Job Title', color='white')
        plt.tight_layout()
        st.pyplot(plt)
        plt.close()


# ====================================================================
# 2. INTERACTIVE DASHBOARD
# ====================================================================
elif page == "2. Interactive Dashboard 📊":
    st.title("Interactive Global Salary Explorer")
    st.markdown("Filter and visualize the data to uncover specific market trends.")
    st.markdown("---")

    # --- Filters in a dedicated container ---
    with st.container(border=True):
        st.subheader("Data Filters")
        
        # Row 1 of filters
        filter_col1, filter_col2, filter_col3 = st.columns(3)
        exp_options = df['experience_level'].unique()
        remote_options = df['remote_ratio_label'].unique()
        year_options = sorted(df['work_year'].unique())

        with filter_col1:
            selected_exp = st.multiselect('Experience Level', options=exp_options, default=exp_options)
        
        with filter_col2:
            selected_remote = st.multiselect('Remote Status', options=remote_options, default=remote_options)
            
        with filter_col3:
            selected_year = st.slider(
                'Work Year Range',
                min_value=min(year_options),
                max_value=max(year_options),
                value=(min(year_options), max(year_options))
            )
            
        # Row 2 of filters
        filter_col4, filter_col5, filter_col6 = st.columns(3)
        comp_size_options = df['company_size'].unique()
        emp_type_options = df['employment_type'].unique()
        job_title_options = df['job_title'].unique()

        with filter_col4:
            selected_comp_size = st.multiselect('Company Size', options=comp_size_options, default=comp_size_options)
        
        with filter_col5:
            selected_emp_type = st.multiselect('Employment Type', options=emp_type_options, default=emp_type_options)
        
        with filter_col6:
            selected_job_title = st.multiselect('Job Title', options=job_title_options, default=job_title_options)


    # Apply Filters
    df_filtered = df[
        (df['experience_level'].isin(selected_exp)) & 
        (df['remote_ratio_label'].isin(selected_remote)) &
        (df['work_year'] >= selected_year[0]) &
        (df['work_year'] <= selected_year[1]) &
        (df['company_size'].isin(selected_comp_size)) &             
        (df['employment_type'].isin(selected_emp_type)) &           
        (df['job_title'].isin(selected_job_title))                  
    ]
    
    if df_filtered.empty:
        st.warning("No data matches the selected filters. Please adjust the settings.")
        st.stop()
        
    st.markdown("---")
    
    # --- CHART 1 & 2: Country Charts (Original) ---
    chart_row_1_col1, chart_row_1_col2 = st.columns(2)
    
    with chart_row_1_col1:
        st.subheader("High-Paying Countries (Employee Residence)")
        top_countries = df_filtered.groupby('Employee Residence')['Salary (USD)'].mean().nlargest(10).reset_index()
        fig3 = px.bar(
            top_countries, 
            x='Salary (USD)', 
            y='Employee Residence', 
            orientation='h',
            title='Top 10 Highest Average Salaries by Employee Residence (USD)',
            labels={'Employee Residence': 'Country'},
            color='Salary (USD)',
            color_continuous_scale='Electric', 
            template='plotly_dark'
        )
        fig3.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig3, use_container_width=True)
    
    with chart_row_1_col2:
        st.subheader("Job Openings by Country")
        top_er = df_filtered['Employee Residence'].value_counts()[:10].reset_index()
        top_er.columns = ['Employee Residence', 'Job Openings']
        fig4 = px.bar(
            top_er, 
            x='Job Openings', 
            y='Employee Residence', 
            orientation='h',
            title='Top 10 Countries with Most DS Employees/Openings',
            labels={'Employee Residence': 'Country'},
            color='Job Openings',
            color_continuous_scale='Cividis',
            template='plotly_dark' 
        )
        fig4.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("---")
    
    # --- CHART 3 & 4: New Charts Added ---
    chart_row_2_col1, chart_row_2_col2 = st.columns(2)
    
    # NEW CHART 3: Average Salary by Company Size
    with chart_row_2_col1:
        st.subheader("Average Salary by Company Size")
        salary_by_size = df_filtered.groupby('company_size')['Salary (USD)'].mean().reindex(['S', 'M', 'L']).reset_index()
        fig5 = px.bar(
            salary_by_size,
            x='company_size',
            y='Salary (USD)',
            title='Mean Salary (USD) by Company Size',
            labels={'company_size': 'Company Size', 'Salary (USD)': 'Mean Salary (USD)'},
            color='Salary (USD)',
            color_continuous_scale='Plasma',
            template='plotly_dark'
        )
        st.plotly_chart(fig5, use_container_width=True)
        
    # NEW CHART 4: Average Salary by Employment Type
    with chart_row_2_col2:
        st.subheader("Average Salary by Employment Type")
        salary_by_type = df_filtered.groupby('employment_type')['Salary (USD)'].mean().sort_values(ascending=False).reset_index()
        fig6 = px.bar(
            salary_by_type,
            x='employment_type',
            y='Salary (USD)',
            title='Mean Salary (USD) by Employment Type',
            labels={'employment_type': 'Employment Type', 'Salary (USD)': 'Mean Salary (USD)'},
            color='Salary (USD)',
            color_continuous_scale='Inferno',
            template='plotly_dark'
        )
        st.plotly_chart(fig6, use_container_width=True)


# ====================================================================
# 3. SALARY PREDICTOR (ML)
# ====================================================================
elif page == "3. Salary Predictor 🧠":
    st.title("Data Science Salary Prediction Engine")
    st.markdown("Harness the power of the **Gradient Boosting Regressor** to estimate your market value.")
    st.markdown("---")
    

    # --- Prediction Form Layout ---
    with st.form("salary_prediction_form"):
        st.subheader("Input Job Profile Features")
        
        col_exp, col_remote, col_comp_size = st.columns(3)
        with col_exp:
            input_experience = st.selectbox('Experience Level', df['experience_level'].unique())
        with col_remote:
            input_remote = st.selectbox('Remote Status', df['remote_ratio_label'].unique())
        with col_comp_size:
            input_company_size = st.selectbox('Company Size', df['company_size'].unique())

        col_title, col_residence, col_location = st.columns(3)
        with col_title:
            input_job_title = st.selectbox('Job Title', df['job_title'].unique())
        with col_residence:
            input_residence = st.selectbox('Employee Residence', df['Employee Residence'].unique())
        with col_location:
            input_location = st.selectbox('Company Location', df['Company Location'].unique())
        
        st.markdown("---")
        submitted = st.form_submit_button('PREDICT SALARY', type='primary')

    # --- Prediction Logic (Simulated) ---
    if submitted:
        # Define base salary and multipliers for simulation
        base_salary = df['Salary (USD)'].mean().round(0)
        exp_multiplier = {'Entry-level (Junior)': 0.8, 'Mid-level (Intermediate)': 1.0, 'Senior-level (Expert)': 1.3, 'Executive (Director)': 1.6}
        
        # Calculate location mean and adjustment
        location_mean = df[df['Employee Residence'] == input_residence]['Salary (USD)'].mean()
        location_adjustment = location_mean / base_salary if not pd.isna(location_mean) else 1.0

        remote_adjustment = 1.05 if input_remote == 'Fully Remote' else 1.0
        
        estimate = base_salary * exp_multiplier.get(input_experience, 1.0) * location_adjustment * remote_adjustment
        
        # Add random noise for a more "realistic" prediction range
        simulated_prediction = int(estimate * (1 + np.random.uniform(-0.08, 0.08)))
        final_prediction = max(40000, min(simulated_prediction, 400000))
        
        st.success("✅ Prediction Generated")
        st.balloons()
        
        # Display prediction using the stylish metric card
        predicted_value_formatted = f"${final_prediction:,.0f}"
        
        st.markdown(
            create_metric_card(
                "Predicted Annual Salary (USD)", 
                predicted_value_formatted,
                "Based on GBR simulation and provided features"
            ), 
            unsafe_allow_html=True
        )
        
        st.caption(f"The model estimates a fair market salary of **{predicted_value_formatted} USD** for a **{input_experience} {input_job_title}** based in **{input_residence}**.")
        
        # --- AI EXPLANATION SECTION ---
        st.markdown("### 🤖 AI Prediction Breakdown")
        
        # Call the explanation function
        explanation_html = generate_explanation(df, input_experience, input_residence, base_salary, location_mean, final_prediction)
        
        # Render the explanation
        st.markdown(explanation_html, unsafe_allow_html=True)