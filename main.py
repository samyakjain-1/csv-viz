import streamlit as st
import pandas as pd
import plotly.express as px
import google.generativeai as genai
import os
from dotenv import load_dotenv
import json
import io

# Load environment variables
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-2.0-flash-lite')

st.set_page_config(page_title="🤖 AI-Powered CSV Visualizer", layout="wide")
st.title("🤖 AI-Powered CSV Visualizer")
st.markdown("*Upload a CSV and let AI agents analyze, decide, and visualize your data!*")

def agent_1_analyze_data(df):
    """🧠 Agent 1: What is this data? What can we do?"""
    # Get basic info about the dataset
    info = {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.to_dict(),
        "sample_data": df.head(3).to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "numeric_columns": list(df.select_dtypes(include=['number']).columns),
        "categorical_columns": list(df.select_dtypes(include=['object']).columns)
    }
    
    prompt = f"""
    Analyze this dataset and provide insights:
    
    Dataset Info:
    - Shape: {info['shape']} (rows, columns)
    - Columns: {info['columns']}
    - Data Types: {info['dtypes']}
    - Sample Data: {info['sample_data']}
    - Missing Values: {info['missing_values']}
    - Numeric Columns: {info['numeric_columns']}
    - Categorical Columns: {info['categorical_columns']}
    
    Please provide:
    1. A brief description of what this data represents
    2. Key characteristics and patterns you notice
    3. Potential analysis opportunities
    4. Data quality observations
    
    Keep it concise and actionable.
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error in Agent 1: {str(e)}"

def agent_2_decide_action(df, agent1_analysis):
    """🎯 Agent 2: What should we actually do?"""
    
    prompt = f"""
    Based on this data analysis from Agent 1:
    {agent1_analysis}
    
    Dataset has {df.shape[0]} rows and {df.shape[1]} columns.
    Columns: {list(df.columns)}
    
    Decide on the BEST visualization and analysis approach. Choose ONE of these options:
    
    1. HISTOGRAM - for single numeric variable distribution
    2. SCATTER - for relationship between two numeric variables  
    3. BAR - for categorical data or comparisons
    4. LINE - for time series or trends
    5. BOX - for distribution comparison across categories
    6. CORRELATION - for correlation matrix of numeric variables
    
    Respond with EXACTLY this format:
    CHART_TYPE: [one of above]
    X_COLUMN: [column name]
    Y_COLUMN: [column name or None]
    COLOR_COLUMN: [column name or None]
    REASONING: [brief explanation]
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error in Agent 2: {str(e)}"

def agent_3_create_visualization(df, agent2_decision):
    """🖼 Agent 3: Create visual/code/output"""
    
    try:
        # Parse Agent 2's decision
        lines = agent2_decision.split('\n')
        decision = {}
        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                decision[key.strip()] = value.strip()
        
        chart_type = decision.get('CHART_TYPE', '').upper()
        x_col = decision.get('X_COLUMN', '')
        y_col = decision.get('Y_COLUMN', '')
        color_col = decision.get('COLOR_COLUMN', '')
        
        # Clean column names (remove quotes if present)
        x_col = x_col.strip('\'"')
        y_col = y_col.strip('\'"') if y_col and y_col.lower() != 'none' else None
        color_col = color_col.strip('\'"') if color_col and color_col.lower() != 'none' else None
        
        # Create visualization based on decision
        if chart_type == 'HISTOGRAM' and x_col in df.columns:
            fig = px.histogram(df, x=x_col, color=color_col, title=f"Distribution of {x_col}")
        elif chart_type == 'SCATTER' and x_col in df.columns and y_col in df.columns:
            fig = px.scatter(df, x=x_col, y=y_col, color=color_col, title=f"{y_col} vs {x_col}")
        elif chart_type == 'BAR' and x_col in df.columns:
            if y_col and y_col in df.columns:
                fig = px.bar(df, x=x_col, y=y_col, color=color_col, title=f"{y_col} by {x_col}")
            else:
                # Value counts for categorical
                counts = df[x_col].value_counts().reset_index()
                fig = px.bar(counts, x='index', y=x_col, title=f"Count of {x_col}")
        elif chart_type == 'LINE' and x_col in df.columns and y_col in df.columns:
            fig = px.line(df, x=x_col, y=y_col, color=color_col, title=f"{y_col} over {x_col}")
        elif chart_type == 'BOX' and x_col in df.columns:
            fig = px.box(df, x=x_col, y=y_col, color=color_col, title=f"Box plot: {y_col} by {x_col}")
        elif chart_type == 'CORRELATION':
            numeric_df = df.select_dtypes(include=['number'])
            if len(numeric_df.columns) > 1:
                corr_matrix = numeric_df.corr()
                fig = px.imshow(corr_matrix, text_auto=True, title="Correlation Matrix")
            else:
                fig = px.histogram(df, x=df.select_dtypes(include=['number']).columns[0])
        else:
            # Fallback to simple histogram of first numeric column
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                fig = px.histogram(df, x=numeric_cols[0], title=f"Distribution of {numeric_cols[0]}")
            else:
                fig = px.bar(df[df.columns[0]].value_counts().reset_index(), 
                           x='index', y=df.columns[0], title=f"Count of {df.columns[0]}")
        
        return fig, decision.get('REASONING', 'Visualization created successfully!')
        
    except Exception as e:
        # Fallback visualization
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            fig = px.histogram(df, x=numeric_cols[0], title=f"Distribution of {numeric_cols[0]}")
        else:
            fig = px.bar(df[df.columns[0]].value_counts().reset_index(), 
                       x='index', y=df.columns[0], title=f"Count of {df.columns[0]}")
        
        return fig, f"Agent 3 error: {str(e)}. Created fallback visualization."

# Main App
uploaded_file = st.file_uploader("📂 Upload a CSV file", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        
        # Show basic preview
        st.subheader("📊 Data Preview")
        st.dataframe(df.head(), use_container_width=True)
        
        # Add AI Analysis button
        if st.button("🚀 Start AI Analysis", type="primary"):
            
            # Create columns for the agent flow
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("🧠 Agent 1: Data Analysis")
                with st.spinner("Analyzing data..."):
                    agent1_result = agent_1_analyze_data(df)
                st.write(agent1_result)
            
            with col2:
                st.subheader("🎯 Agent 2: Decision Making")
                with st.spinner("Deciding visualization strategy..."):
                    agent2_result = agent_2_decide_action(df, agent1_result)
                st.write(agent2_result)
            
            with col3:
                st.subheader("🖼 Agent 3: Visualization")
                with st.spinner("Creating visualization..."):
                    fig, reasoning = agent_3_create_visualization(df, agent2_result)
                st.write(f"**Reasoning:** {reasoning}")
            
            # Show the final visualization
            st.subheader("✅ AI-Generated Visualization")
            st.plotly_chart(fig, use_container_width=True)
            
            # Export options
            st.subheader("📥 Export Options")
            col_export1, col_export2, col_export3 = st.columns(3)
            
            with col_export1:
                # Export cleaned data
                csv_buffer = io.StringIO()
                df.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="📄 Download CSV",
                    data=csv_buffer.getvalue(),
                    file_name=f"analyzed_{uploaded_file.name}",
                    mime="text/csv"
                )
            
            with col_export2:
                # Export analysis report
                report = f"""
# AI Analysis Report

## Agent 1 - Data Analysis
{agent1_result}

## Agent 2 - Visualization Decision
{agent2_result}

## Agent 3 - Visualization Output
{reasoning}

Generated by AI-Powered CSV Visualizer
                """
                st.download_button(
                    label="📋 Download Report",
                    data=report,
                    file_name=f"analysis_report_{uploaded_file.name.replace('.csv', '.md')}",
                    mime="text/markdown"
                )
            
            with col_export3:
                # Export chart as HTML
                html_buffer = io.StringIO()
                fig.write_html(html_buffer)
                st.download_button(
                    label="📊 Download Chart",
                    data=html_buffer.getvalue(),
                    file_name=f"chart_{uploaded_file.name.replace('.csv', '.html')}",
                    mime="text/html"
                )
        
        # Manual exploration option
        with st.expander("🔧 Manual Data Exploration"):
            st.subheader("📈 Column Statistics")
            st.write(df.describe(include='all'))
            
            st.subheader("📊 Quick Manual Visualization")
            col = st.selectbox("Select a column to visualize", df.columns)
            
            if pd.api.types.is_numeric_dtype(df[col]):
                st.plotly_chart(px.histogram(df, x=col), use_container_width=True)
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                if len(df.select_dtypes('number').columns) > 0:
                    st.plotly_chart(px.line(df, x=col, y=df.select_dtypes('number').columns[0]), use_container_width=True)
            else:
                value_counts = df[col].value_counts().reset_index()
                st.plotly_chart(px.bar(value_counts, x='index', y=col), use_container_width=True)
                
    except Exception as e:
        st.error(f"Error reading file: {e}")

# Sidebar with instructions
with st.sidebar:
    st.header("🤖 How it works")
    st.markdown("""
    **Multi-Agent AI Flow:**
    
    1. 🧠 **Agent 1** analyzes your data structure and content
    2. 🎯 **Agent 2** decides the best visualization approach  
    3. 🖼 **Agent 3** creates the perfect chart for your data
    4. ✅ **Export** your results in multiple formats
    
    **Setup:**
    - Create a `.env` file
    - Add your Gemini API key: `GEMINI_API_KEY=your_key_here`
    - Get your API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
    """)
    
    st.header("📁 Sample Files")
    st.write("Try uploading `sample_data.csv` or `sales_data.csv` from your project!")
