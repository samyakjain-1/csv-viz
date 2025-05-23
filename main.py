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
model = genai.GenerativeModel('gemini-2.5-flash-preview-05-20')

st.set_page_config(page_title="🤖 AI-Powered CSV Visualizer", layout="wide")
st.title("🤖 AI-Powered CSV Visualizer")
st.markdown("*Upload a CSV and let AI agents analyze, decide, and visualize your data!*")

def safe_display_dataframe(df, title="Data Preview"):
    """Safely display dataframe by handling Arrow serialization issues"""
    try:
        # Create a copy for display to avoid modifying original data
        display_df = df.copy()
        
        # Convert potentially problematic columns to string for display
        for col in display_df.columns:
            if display_df[col].dtype == 'object':
                # Check if column has mixed types or complex strings
                sample_vals = display_df[col].dropna().head(5).astype(str)
                if any(len(str(val)) > 50 or '_' in str(val) or '-' in str(val) for val in sample_vals):
                    display_df[col] = display_df[col].astype(str)
        
        st.dataframe(display_df.head(), use_container_width=True)
    except Exception as e:
        st.warning(f"Display issue resolved: {str(e)}")
        # Fallback: convert all object columns to string
        display_df = df.copy()
        for col in display_df.select_dtypes(include=['object']).columns:
            display_df[col] = display_df[col].astype(str)
        st.dataframe(display_df.head(), use_container_width=True)

def agent_1_analyze_data(df):
    """🧠 Agent 1: Smart Data Analysis + Generate Analysis Options for User"""
    
    original_shape = df.shape
    cleaning_actions = []
    
    # 1. SMART COLUMN ANALYSIS & CLEANING
    columns_to_drop = []
    
    for col in df.columns:
        col_info = {
            'name': col,
            'dtype': str(df[col].dtype),
            'unique_count': df[col].nunique(),
            'null_count': df[col].isnull().sum(),
            'null_percentage': (df[col].isnull().sum() / len(df)) * 100
        }
        
        # Check for columns to potentially remove
        if col_info['null_percentage'] > 90:
            columns_to_drop.append(col)
            cleaning_actions.append(f"❌ Removed '{col}': {col_info['null_percentage']:.1f}% missing values")
        elif col_info['unique_count'] <= 1:
            columns_to_drop.append(col)
            cleaning_actions.append(f"❌ Removed '{col}': Only {col_info['unique_count']} unique value(s)")
        elif (col_info['unique_count'] / len(df)) > 0.95 and any(keyword in col.lower() for keyword in ['id', 'index', 'key', 'uuid', 'guid', 'match']):
            columns_to_drop.append(col)
            cleaning_actions.append(f"❌ Removed '{col}': Appears to be an ID column ({col_info['unique_count']} unique values)")
        elif col_info['unique_count'] > len(df) * 0.8 and col_info['unique_count'] > 10:
            if not any(keyword in col.lower() for keyword in ['name', 'email', 'address', 'description']):
                columns_to_drop.append(col)
                cleaning_actions.append(f"❌ Removed '{col}': Too high cardinality ({col_info['unique_count']} unique values)")
        # Additional check for problematic ID-like patterns that cause Arrow serialization issues
        elif col_info['dtype'] == 'object' and col_info['unique_count'] == len(df):
            # Check if values look like IDs (contains underscores, dashes, or mixed patterns)
            sample_values = df[col].dropna().head(10).astype(str)
            if sample_values.str.contains(r'[_-]').sum() > len(sample_values) * 0.7:  # 70% contain underscores or dashes
                columns_to_drop.append(col)
                cleaning_actions.append(f"❌ Removed '{col}': Contains ID-like patterns that cause serialization issues")
    
    # Apply column drops
    df_cleaned = df.drop(columns=columns_to_drop)
    
    # 2. SMART DATA TYPE OPTIMIZATION
    for col in df_cleaned.columns:
        if df_cleaned[col].dtype == 'object':
            numeric_converted = pd.to_numeric(df_cleaned[col], errors='coerce')
            if not numeric_converted.isnull().all():
                non_null_percentage = (len(numeric_converted.dropna()) / len(df_cleaned)) * 100
                if non_null_percentage > 70:
                    df_cleaned[col] = numeric_converted
                    cleaning_actions.append(f"🔄 Converted '{col}' to numeric (improved {non_null_percentage:.1f}% of values)")
    
    # 3. HANDLE MISSING VALUES INTELLIGENTLY
    for col in df_cleaned.columns:
        missing_pct = (df_cleaned[col].isnull().sum() / len(df_cleaned)) * 100
        if 0 < missing_pct <= 30:
            if df_cleaned[col].dtype in ['int64', 'float64']:
                # Fix FutureWarning: Use proper pandas syntax instead of chained assignment
                df_cleaned.loc[:, col] = df_cleaned[col].fillna(df_cleaned[col].median())
                cleaning_actions.append(f"🔧 Filled {missing_pct:.1f}% missing values in '{col}' with median")
            elif df_cleaned[col].dtype == 'object':
                mode_val = df_cleaned[col].mode()
                if len(mode_val) > 0:
                    # Fix FutureWarning: Use proper pandas syntax instead of chained assignment
                    df_cleaned.loc[:, col] = df_cleaned[col].fillna(mode_val[0])
                    cleaning_actions.append(f"🔧 Filled {missing_pct:.1f}% missing values in '{col}' with mode")
    
    # 4. ANALYZE COLUMN RELATIONSHIPS & GENERATE ANALYSIS OPTIONS
    numeric_cols = list(df_cleaned.select_dtypes(include=['number']).columns)
    categorical_cols = list(df_cleaned.select_dtypes(include=['object']).columns)
    datetime_cols = list(df_cleaned.select_dtypes(include=['datetime']).columns)
    
    # Generate intelligent analysis options for user to choose from
    info = {
        "original_shape": original_shape,
        "cleaned_shape": df_cleaned.shape,
        "columns_removed": len(columns_to_drop),
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "datetime_columns": datetime_cols,
        "sample_data": df_cleaned.head(3).to_dict()
    }
    
    prompt = f"""
    As an expert data analyst, analyze this dataset and provide MULTIPLE ANALYSIS OPTIONS for the user to choose from.
    
    DATASET OVERVIEW:
    - Original: {info['original_shape']} → Cleaned: {info['cleaned_shape']}
    - Removed {info['columns_removed']} unnecessary columns
    
    AVAILABLE COLUMNS FOR ANALYSIS:
    - Numeric: {numeric_cols}
    - Categorical: {categorical_cols} 
    - DateTime: {datetime_cols}
    
    SAMPLE DATA:
    {info['sample_data']}
    
    Please provide your analysis in this EXACT format:
    
    **DATASET CONTEXT:**
    [What this data appears to represent and likely business domain]
    
    **ANALYSIS OPTIONS:**
    
    **Option 1: [Name of Analysis Type]**
    Focus: [What this analysis will explore]
    Best for: [When to use this analysis]
    Key Questions: [2-3 specific questions this will answer]
    
    **Option 2: [Name of Analysis Type]**
    Focus: [What this analysis will explore]
    Best for: [When to use this analysis]
    Key Questions: [2-3 specific questions this will answer]
    
    **Option 3: [Name of Analysis Type]**
    Focus: [What this analysis will explore]
    Best for: [When to use this analysis]
    Key Questions: [2-3 specific questions this will answer]
    
    **Option 4: [Name of Analysis Type]**
    Focus: [What this analysis will explore]
    Best for: [When to use this analysis]
    Key Questions: [2-3 specific questions this will answer]
    
    **COLUMN INSIGHTS:**
    [Brief insights about each important column and what they might represent]
    
    Make each option distinctly different and valuable for different business scenarios!
    """
    
    try:
        response = model.generate_content(prompt)
        analysis_text = response.text
        
        if cleaning_actions:
            analysis_text = f"""
**🧹 DATA CLEANING SUMMARY:**
{chr(10).join(cleaning_actions)}

{analysis_text}
            """
        
        return analysis_text, df_cleaned
    except Exception as e:
        return f"Error in Agent 1: {str(e)}", df_cleaned

def agent_2_decide_action(df, agent1_analysis, user_selection, custom_question=None):
    """🎯 Agent 2: Create Visualization Strategy Based on User's Choice"""
    
    numeric_cols = list(df.select_dtypes(include=['number']).columns)
    categorical_cols = list(df.select_dtypes(include=['object']).columns)
    datetime_cols = list(df.select_dtypes(include=['datetime']).columns)
    
    if custom_question:
        user_context = f"User wants to explore: '{custom_question}'"
    else:
        user_context = f"User selected: {user_selection}"
    
    prompt = f"""
    You are Agent 2, the Visualization Strategist. Based on Agent 1's analysis and the user's specific interests, create a targeted visualization strategy.
    
    AGENT 1'S ANALYSIS:
    {agent1_analysis}
    
    USER'S CHOICE:
    {user_context}
    
    AVAILABLE COLUMNS:
    - Numeric: {numeric_cols}
    - Categorical: {categorical_cols}
    - DateTime: {datetime_cols}
    
    Based on the user's specific interest, create a FOCUSED VISUALIZATION STRATEGY with 3 charts that directly address their needs.
    
    Respond in this EXACT format:
    
    **ADDRESSING USER'S INTEREST:**
    [How your strategy specifically addresses what the user wants to discover]
    
    **VISUALIZATION STRATEGY - TARGETED CHARTS:**
    
    CHART_1:
    TYPE: [HISTOGRAM/SCATTER/BAR/LINE/BOX/HEATMAP/CORRELATION]
    PRIMARY_COLUMN: [exact column name]
    SECONDARY_COLUMN: [exact column name or None]
    COLOR_GROUPING: [exact column name or None]
    PURPOSE: [How this chart answers the user's question]
    
    CHART_2:
    TYPE: [HISTOGRAM/SCATTER/BAR/LINE/BOX/HEATMAP/CORRELATION]
    PRIMARY_COLUMN: [exact column name]
    SECONDARY_COLUMN: [exact column name or None]
    COLOR_GROUPING: [exact column name or None]
    PURPOSE: [How this chart answers the user's question]
    
    CHART_3:
    TYPE: [HISTOGRAM/SCATTER/BAR/LINE/BOX/HEATMAP/CORRELATION]
    PRIMARY_COLUMN: [exact column name]
    SECONDARY_COLUMN: [exact column name or None]
    COLOR_GROUPING: [exact column name or None]
    PURPOSE: [How this chart answers the user's question]
    
    **STRATEGIC REASONING:**
    [Explain WHY these specific visualizations will provide the insights the user is looking for]
    
    **EXPECTED INSIGHTS:**
    [What specific findings the user can expect to discover]
    
    Focus on charts that directly answer the user's questions and interests!
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error in Agent 2: {str(e)}"

def agent_3_create_visualization(df, agent2_decision):
    """🖼 Agent 3: Create Multiple Visualizations with Descriptions"""
    
    def parse_chart_specs(decision_text):
        """Parse multiple chart specifications from Agent 2's response"""
        charts = []
        lines = decision_text.split('\n')
        current_chart = {}
        
        for line in lines:
            line = line.strip()
            if line.startswith('CHART_'):
                if current_chart:  # Save previous chart
                    charts.append(current_chart)
                current_chart = {'chart_num': line}
            elif ':' in line and current_chart:
                key, value = line.split(':', 1)
                current_chart[key.strip().upper()] = value.strip()
        
        if current_chart:  # Don't forget the last chart
            charts.append(current_chart)
        
        return charts
    
    def create_single_chart(df, chart_spec):
        """Create a single chart based on specifications"""
        try:
            chart_type = chart_spec.get('TYPE', '').upper().strip()
            primary_col = chart_spec.get('PRIMARY_COLUMN', '').strip('\'"')
            secondary_col = chart_spec.get('SECONDARY_COLUMN', '').strip('\'"')
            color_col = chart_spec.get('COLOR_GROUPING', '').strip('\'"')
            purpose = chart_spec.get('PURPOSE', 'Analysis chart')
            
            # Clean None values
            secondary_col = secondary_col if secondary_col and secondary_col.lower() != 'none' else None
            color_col = color_col if color_col and color_col.lower() != 'none' else None
            
            # Validate columns exist
            if primary_col not in df.columns:
                primary_col = df.columns[0]
            if secondary_col and secondary_col not in df.columns:
                secondary_col = None
            if color_col and color_col not in df.columns:
                color_col = None
            
            # Create visualization based on type
            if chart_type == 'HISTOGRAM':
                fig = px.histogram(df, x=primary_col, color=color_col, 
                                 title=f"Distribution Analysis: {primary_col}")
                description = f"📊 **Distribution of {primary_col}**: {purpose}. This histogram shows the frequency distribution and helps identify patterns, outliers, and data spread."
                
            elif chart_type == 'SCATTER':
                if secondary_col:
                    fig = px.scatter(df, x=primary_col, y=secondary_col, color=color_col,
                                   title=f"Relationship: {secondary_col} vs {primary_col}")
                    description = f"🔍 **{secondary_col} vs {primary_col} Relationship**: {purpose}. This scatter plot reveals correlations, trends, and outliers between these two variables."
                else:
                    fig = px.histogram(df, x=primary_col, color=color_col)
                    description = f"📊 **{primary_col} Distribution**: {purpose}. Showing the distribution pattern of this variable."
                    
            elif chart_type == 'BAR':
                if secondary_col:
                    fig = px.bar(df, x=primary_col, y=secondary_col, color=color_col,
                               title=f"Comparison: {secondary_col} by {primary_col}")
                    description = f"📈 **{secondary_col} by {primary_col}**: {purpose}. This bar chart compares values across different categories, highlighting differences and rankings."
                else:
                    counts = df[primary_col].value_counts().reset_index()
                    fig = px.bar(counts, x='index', y=primary_col, 
                               title=f"Count Analysis: {primary_col}")
                    description = f"📊 **{primary_col} Count Distribution**: {purpose}. Shows the frequency of each category, identifying the most and least common values."
                    
            elif chart_type == 'LINE':
                if secondary_col:
                    fig = px.line(df, x=primary_col, y=secondary_col, color=color_col,
                                title=f"Trend Analysis: {secondary_col} over {primary_col}")
                    description = f"📈 **{secondary_col} Trends over {primary_col}**: {purpose}. This line chart reveals trends, seasonality, and changes over time or sequence."
                else:
                    fig = px.histogram(df, x=primary_col)
                    description = f"📊 **{primary_col} Distribution**: {purpose}."
                    
            elif chart_type == 'BOX':
                if secondary_col:
                    fig = px.box(df, x=primary_col, y=secondary_col, color=color_col,
                               title=f"Distribution Comparison: {secondary_col} by {primary_col}")
                    description = f"📦 **{secondary_col} Distribution by {primary_col}**: {purpose}. Box plots show median, quartiles, and outliers for each group, perfect for comparing distributions."
                else:
                    fig = px.box(df, y=primary_col, title=f"Box Plot: {primary_col}")
                    description = f"📦 **{primary_col} Distribution Summary**: {purpose}. Shows the statistical distribution including median, quartiles, and potential outliers."
                    
            elif chart_type in ['HEATMAP', 'CORRELATION']:
                numeric_cols = df.select_dtypes(include=['number']).columns
                if len(numeric_cols) >= 2:
                    corr_matrix = df[numeric_cols].corr()
                    fig = px.imshow(corr_matrix, text_auto=True, title="Correlation Heatmap")
                    description = f"🔥 **Correlation Matrix**: {purpose}. This heatmap reveals relationships between all numeric variables, with colors indicating correlation strength."
                else:
                    fig = px.histogram(df, x=primary_col)
                    description = f"📊 **{primary_col} Distribution**: {purpose}."
                    
            else:
                # Intelligent fallback
                if pd.api.types.is_numeric_dtype(df[primary_col]):
                    fig = px.histogram(df, x=primary_col, color=color_col, 
                                     title=f"Distribution of {primary_col}")
                    description = f"📊 **{primary_col} Distribution**: {purpose}. Shows the frequency distribution of this numeric variable."
                else:
                    counts = df[primary_col].value_counts().reset_index()
                    fig = px.bar(counts, x='index', y=primary_col, 
                               title=f"Count of {primary_col}")
                    description = f"📊 **{primary_col} Frequency**: {purpose}. Shows how often each category appears in the dataset."
            
            return fig, description
            
        except Exception as e:
            # Create fallback chart
            if len(df.select_dtypes(include=['number']).columns) > 0:
                numeric_col = df.select_dtypes(include=['number']).columns[0]
                fig = px.histogram(df, x=numeric_col, title=f"Distribution of {numeric_col}")
                description = f"📊 **Fallback Visualization**: Created distribution chart for {numeric_col}. {purpose}"
            else:
                cat_col = df.columns[0]
                counts = df[cat_col].value_counts().reset_index()
                fig = px.bar(counts, x='index', y=cat_col, title=f"Count of {cat_col}")
                description = f"📊 **Fallback Visualization**: Created count chart for {cat_col}. {purpose}"
            
            return fig, description
    
    try:
        # Parse multiple chart specifications
        chart_specs = parse_chart_specs(agent2_decision)
        
        if not chart_specs:
            # Fallback: create default charts
            chart_specs = [
                {'TYPE': 'HISTOGRAM', 'PRIMARY_COLUMN': df.columns[0], 'PURPOSE': 'Basic distribution analysis'},
                {'TYPE': 'CORRELATION', 'PURPOSE': 'Variable relationships'},
                {'TYPE': 'BAR', 'PRIMARY_COLUMN': df.columns[0], 'PURPOSE': 'Category analysis'}
            ]
        
        # Create all charts
        charts_and_descriptions = []
        for i, spec in enumerate(chart_specs[:3]):  # Limit to 3 charts
            fig, description = create_single_chart(df, spec)
            charts_and_descriptions.append({
                'figure': fig,
                'description': description,
                'title': f"Chart {i+1}",
                'chart_type': spec.get('TYPE', 'Analysis')
            })
        
        # Extract overall reasoning
        reasoning = "Multiple visualizations created to provide comprehensive data insights."
        if '**STRATEGIC REASONING:**' in agent2_decision:
            reasoning = agent2_decision.split('**STRATEGIC REASONING:**')[1].split('**')[0].strip()
        elif '**EXPECTED INSIGHTS:**' in agent2_decision:
            reasoning = agent2_decision.split('**EXPECTED INSIGHTS:**')[1].strip()
        
        return charts_and_descriptions, reasoning
        
    except Exception as e:
        # Fallback: create default charts
        charts_and_descriptions = []
        
        # Chart 1: Distribution of first numeric column
        if len(df.select_dtypes(include=['number']).columns) > 0:
            numeric_col = df.select_dtypes(include=['number']).columns[0]
            fig1 = px.histogram(df, x=numeric_col, title=f"Distribution of {numeric_col}")
            charts_and_descriptions.append({
                'figure': fig1,
                'description': f"📊 **Distribution Analysis**: Shows the frequency distribution of {numeric_col}, helping identify patterns and outliers.",
                'title': "Chart 1: Distribution",
                'chart_type': 'HISTOGRAM'
            })
        
        # Chart 2: Correlation if multiple numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) >= 2:
            corr_matrix = df[numeric_cols].corr()
            fig2 = px.imshow(corr_matrix, text_auto=True, title="Correlation Matrix")
            charts_and_descriptions.append({
                'figure': fig2,
                'description': "🔥 **Correlation Analysis**: Reveals relationships between numeric variables. Strong correlations (closer to 1 or -1) indicate related variables.",
                'title': "Chart 2: Correlations",
                'chart_type': 'CORRELATION'
            })
        
        # Chart 3: Category analysis
        if len(df.select_dtypes(include=['object']).columns) > 0:
            cat_col = df.select_dtypes(include=['object']).columns[0]
            counts = df[cat_col].value_counts().reset_index()
            fig3 = px.bar(counts, x='index', y=cat_col, title=f"Distribution of {cat_col}")
            charts_and_descriptions.append({
                'figure': fig3,
                'description': f"📊 **Category Analysis**: Shows the frequency of each {cat_col} category, identifying the most and least common values.",
                'title': "Chart 3: Categories",
                'chart_type': 'BAR'
            })
        
        return charts_and_descriptions, f"Created fallback visualizations due to error: {str(e)}"

# Main App
uploaded_file = st.file_uploader("📂 Upload a CSV file", type=["csv"])

if uploaded_file:
    # Initialize session state
    if 'uploaded_file_name' not in st.session_state or st.session_state.uploaded_file_name != uploaded_file.name:
        st.session_state.uploaded_file_name = uploaded_file.name
        st.session_state.df_original = None
        st.session_state.df_cleaned = None
        st.session_state.agent1_result = None
        st.session_state.analysis_complete = False
        st.session_state.options_ready = False
        st.session_state.visualizations_ready = False
    
    try:
        # Load and process data only once
        if st.session_state.df_original is None:
            df = pd.read_csv(uploaded_file)
            st.session_state.df_original = df
            
            # Auto-detect and convert date columns
            for col in df.columns:
                if df[col].dtype == 'object':  # Only check string columns
                    try:
                        sample_values = df[col].dropna().head(5).astype(str)
                        if sample_values.str.match(r'\d{4}-\d{2}-\d{2}').any() or \
                           sample_values.str.match(r'\d{2}/\d{2}/\d{4}').any() or \
                           sample_values.str.match(r'\d{2}-\d{2}-\d{4}').any():
                            df[col] = pd.to_datetime(df[col], errors='coerce')
                            st.info(f"✅ Converted '{col}' to datetime format")
                    except:
                        pass
            
            st.session_state.df_original = df
        
        df = st.session_state.df_original
        
        # Show basic preview
        st.subheader("📊 Data Preview")
        safe_display_dataframe(df)
        
        # Manual exploration option (always available)
        with st.expander("🔧 Manual Data Exploration"):
            st.subheader("📈 Column Statistics")
            st.write(df.describe(include='all'))
            
            st.subheader("📊 Quick Manual Visualization")
            col = st.selectbox("Select a column to visualize", df.columns, key="manual_viz_column")
            
            if pd.api.types.is_numeric_dtype(df[col]):
                st.plotly_chart(px.histogram(df, x=col), use_container_width=True)
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                if len(df.select_dtypes('number').columns) > 0:
                    st.plotly_chart(px.line(df, x=col, y=df.select_dtypes('number').columns[0]), use_container_width=True)
            else:
                value_counts = df[col].value_counts().reset_index()
                st.plotly_chart(px.bar(value_counts, x='index', y=col), use_container_width=True)
        
        # Step 1: Initial Analysis
        if not st.session_state.analysis_complete:
            if st.button("🚀 Start AI Analysis", type="primary"):
                with st.spinner("🧠 Agent 1: Analyzing data and generating options..."):
                    agent1_result, df_cleaned = agent_1_analyze_data(df)
                    st.session_state.agent1_result = agent1_result
                    st.session_state.df_cleaned = df_cleaned
                    st.session_state.analysis_complete = True
                    st.session_state.options_ready = True
                st.rerun()
        
        # Step 2: Show Analysis Results and User Options
        if st.session_state.analysis_complete and st.session_state.options_ready:
            df_cleaned = st.session_state.df_cleaned
            agent1_result = st.session_state.agent1_result
            
            # Show cleaned data info
            if df.shape != df_cleaned.shape:
                st.info(f"📊 **Data cleaned**: {df.shape[0]} rows × {df.shape[1]} cols → {df_cleaned.shape[0]} rows × {df_cleaned.shape[1]} cols")
                st.subheader("🧹 Cleaned Data Preview")
                safe_display_dataframe(df_cleaned)
            
            # Display Agent 1 results
            st.subheader("🧠 Agent 1: Data Analysis & Options")
            st.write(agent1_result)
            
            # User Selection Interface
            st.subheader("🎯 What would you like to explore?")
            
            col_choice1, col_choice2 = st.columns([3, 1])
            
            with col_choice1:
                analysis_choice = st.radio(
                    "Choose your analysis focus:",
                    options=[
                        "Option 1: First analysis approach",
                        "Option 2: Second analysis approach", 
                        "Option 3: Third analysis approach",
                        "Option 4: Fourth analysis approach",
                        "Custom: I have a specific question"
                    ],
                    key="analysis_choice_radio",
                    help="Select what you're most interested in discovering from your data"
                )
            
            with col_choice2:
                if st.button("🔄 Regenerate Options", help="Get new analysis options from AI"):
                    # Reset analysis to get new options
                    st.session_state.analysis_complete = False
                    st.session_state.options_ready = False
                    st.session_state.agent1_result = None
                    st.rerun()
            
            # Custom question input
            custom_question = None
            if "Custom:" in analysis_choice:
                custom_question = st.text_area(
                    "What specific question do you want to answer with your data?",
                    placeholder="e.g., Which factors most influence sales performance? How do customer ratings vary by region?",
                    key="custom_question_input",
                    help="Be specific about what insights you're looking for"
                )
            
            # Step 3: Create Visualizations
            if st.button("📊 Create Visualizations", type="primary", key="create_viz_button"):
                if "Custom:" in analysis_choice and not custom_question:
                    st.warning("Please enter your custom question before proceeding.")
                else:
                    # Store user choices in session state
                    st.session_state.user_choice = analysis_choice
                    st.session_state.user_question = custom_question
                    st.session_state.options_ready = False
                    st.session_state.visualizations_ready = True
                    st.rerun()
        
        # Step 4: Show Visualizations
        if st.session_state.analysis_complete and st.session_state.visualizations_ready:
            df_cleaned = st.session_state.df_cleaned
            agent1_result = st.session_state.agent1_result
            analysis_choice = st.session_state.user_choice
            custom_question = st.session_state.user_question
            
            # Create visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🎯 Agent 2: Visualization Strategy")
                with st.spinner("Creating targeted visualization strategy..."):
                    agent2_result = agent_2_decide_action(df_cleaned, agent1_result, analysis_choice, custom_question)
                st.write(agent2_result)
            
            with col2:
                st.subheader("🖼 Agent 3: Creating Charts")
                with st.spinner("Generating visualizations..."):
                    charts_and_descriptions, reasoning = agent_3_create_visualization(df_cleaned, agent2_result)
                st.write(f"**Strategy Summary:** {reasoning}")
            
            # Show the final visualizations
            st.subheader("✅ Your Personalized Data Insights")
            if custom_question:
                st.info(f"**Your Question:** {custom_question}")
            else:
                st.info(f"**Analysis Focus:** {analysis_choice}")
            
            for i, chart_info in enumerate(charts_and_descriptions):
                st.subheader(f"{chart_info['title']}: {chart_info['chart_type']}")
                st.write(chart_info['description'])
                st.plotly_chart(chart_info['figure'], use_container_width=True)
                if i < len(charts_and_descriptions) - 1:
                    st.divider()
            
            # Back to options button
            if st.button("🔙 Try Different Analysis", key="back_to_options"):
                st.session_state.options_ready = True
                st.session_state.visualizations_ready = False
                st.rerun()
            
            # Export options
            st.subheader("📥 Export Options")
            col_export1, col_export2, col_export3 = st.columns(3)
            
            with col_export1:
                # Export cleaned data
                csv_buffer = io.StringIO()
                df_cleaned.to_csv(csv_buffer, index=False)
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

## User's Focus
{analysis_choice}
{f"Custom Question: {custom_question}" if custom_question else ""}

## Agent 1 - Data Analysis
{agent1_result}

## Agent 2 - Visualization Strategy
{agent2_result}

## Agent 3 - Visualization Results
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
                # Export charts as HTML
                combined_html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Data Visualization Dashboard</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .chart-container {{ margin-bottom: 40px; }}
        .chart-title {{ font-size: 24px; font-weight: bold; color: #333; }}
        .chart-description {{ font-size: 16px; margin: 10px 0; color: #666; }}
        .user-focus {{ background: #f0f8ff; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
    </style>
</head>
<body>
    <h1>📊 Data Visualization Dashboard</h1>
    <div class="user-focus">
        <strong>Analysis Focus:</strong> {analysis_choice}<br>
        {f"<strong>Custom Question:</strong> {custom_question}" if custom_question else ""}
    </div>
"""
                
                for i, chart_info in enumerate(charts_and_descriptions):
                    chart_html = chart_info['figure'].to_html(include_plotlyjs='cdn')
                    combined_html += f"""
    <div class="chart-container">
        <div class="chart-title">{chart_info['title']}: {chart_info['chart_type']}</div>
        <div class="chart-description">{chart_info['description']}</div>
        {chart_html}
    </div>
"""
                
                combined_html += """
</body>
</html>
"""
                
                st.download_button(
                    label="📊 Download Dashboard",
                    data=combined_html,
                    file_name=f"dashboard_{uploaded_file.name.replace('.csv', '.html')}",
                    mime="text/html"
                )
                
    except Exception as e:
        st.error(f"Error processing file: {e}")

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
