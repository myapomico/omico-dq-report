##############################
# Import necessary libraries
##############################

import pandas as pd
import altair as alt
import streamlit as st
import plotly.express as px
from datetime import datetime

##############################
# Page configuration
##############################

st.set_page_config(
    page_title="Data Quality Report",
    page_icon="☑️",
    layout="wide",
    initial_sidebar_state="expanded")

alt.themes.enable("dark")

##############################
# Specify input data
##############################

dict_filepath_dim = {
    'Uniqueness': 'data/20240812_u_bool.xlsx',
    'Completeness': 'data/20240812_c_bool.xlsx',
    'Validity': 'data/20240812_v_bool.xlsx',
    'Accuracy_OE': 'data/20240812_a_oe_bool.xlsx',
    'Accuracy_RO': ''
}

##############################
# Define functions
##############################

@st.cache_data
def load_data(dict_filepath, str_dim):
    filepath_data = dict_filepath[str_dim]
    excel_file = pd.ExcelFile(filepath_data)
    dict_out = {}
    for sheet_name in excel_file.sheet_names:
        dict_out[sheet_name] = pd.read_excel(filepath_data, sheet_name=sheet_name)
    return dict_out

def plot_donut(input_response, input_text):
    if input_text == 'Uniqueness':
        chart_color = ['#9E024d', '#5a012c']
    elif input_text == 'Completeness':
        chart_color = ['#03989f', '#005458']
    elif input_text == 'Validity':
        chart_color = ['#54019e', '#32005f']
    elif 'Accuracy' in input_text:
        chart_color = ['#9e7b01', '#584500']
    else:
        chart_color = ['#ffffff', '#999999']
    
    source = pd.DataFrame({
        "Dimension": ['', input_text],
        "Score (%)": [100-input_response, input_response]
    })
    source_bg = pd.DataFrame({
        "Dimension": ['', input_text],
        "Score (%)": [100, 0]
    })
    
    plot = alt.Chart(source).mark_arc(innerRadius=45, cornerRadius=25).encode(
        theta="Score (%)",
        color= alt.Color("Dimension:N",
                      scale=alt.Scale(
                          domain=[input_text, ''],
                          range=chart_color),
                      legend=None),
    ).properties(width=130, height=130)
    
    text = plot.mark_text(
        align='center', color="#29b5e8", 
        font="Lato", fontSize=32, fontWeight=700, fontStyle="italic"
        ).encode(text=alt.value(f'{round(input_response)} %'))
    plot_bg = alt.Chart(source_bg).mark_arc(innerRadius=45, cornerRadius=20).encode(
        theta="Score (%)",
        color= alt.Color("Dimension:N",
                      scale=alt.Scale(
                          domain=[input_text, ''],
                          range=chart_color),
                      legend=None),
    ).properties(width=130, height=130)
    return plot_bg + plot + text

def plot_barh(data, tablename='insert_tablename', metric=''):
    if metric == 'Completeness':
        metric_colour = '#02989E'
        status_positive = 'Complete'
        status_negative = 'Null'
    elif metric == 'Uniqueness':
        metric_colour = '#9E024d'
        status_positive = 'Unique'
        status_negative = 'Duplicate'
    elif metric == 'Validity':
        metric_colour = '#53029e'
        status_positive = 'Valid'
        status_negative = 'Invalid'

    # Summarise the data
    data_summary = data.copy()
    if metric == 'Uniqueness':
        data_summary.columns.values[1] = status_positive
    elif metric == 'Completeness' or metric == 'Validity':
        data_summary.rename(columns={metric: status_positive}, inplace=True)
    data_summary[status_negative] = 100 - data_summary[status_positive]
    data_summary = data_summary.melt(id_vars='Table', var_name='Status', value_name='Percentage')

    # Create a horizontal bar chart with a single color
    fig = px.bar(
        data_summary, y='Table', x='Percentage', 
        color='Status', orientation='h', 
        color_discrete_sequence=[metric_colour, '#999999']
        )
    
    # Customise appearance
    fig.update_layout(
        xaxis_title=f"{metric} percentage",
        yaxis_title="",
        plot_bgcolor='rgba(0, 0, 0, 0)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        font=dict(color='white'),
        height=250 + (data.shape[0] * 20),
        xaxis=dict(
            showgrid=True,
            gridcolor='#444444',
            color='white'
        ),
        yaxis=dict(
            color='white'
        )
    )
    # Invert y-axis
    fig.update_yaxes(
        autorange="reversed",
        ticksuffix="  "  # Add space between y-tick labels and y-axis
    )
    # Make the bars stacked
    fig.update_layout(barmode='stack')

    # Calculate the right end position of the first category for each column
    df_first_category = data_summary[data_summary['Status'] == status_positive]
    for _, row in df_first_category.iterrows():
        fig.add_annotation(
            x=row['Percentage'],  # Position at the end of the first stack
            y=row['Table'],
            text=f"{row['Percentage']:.1f}%",
            showarrow=False,
            font=dict(size=12, color="white"),
            xanchor='right',  # Anchor the text to the right
            yanchor='middle'
        )

    return fig

##############################
# Sidebar navigation
##############################

with st.sidebar:
    st.title('☑️ Data Quality Report')

    dataset_list = ['CaSP', 'MoST (WIP)']
    selected_dataset = st.selectbox('Select a dataset', dataset_list)

    if selected_dataset == 'MoST (WIP)':
        st.warning("We are working hard on preparing the report for MoST dataset, stay tuned!")

    dim_list = ['Uniqueness', 'Completeness', 'Validity', 'Accuracy (WIP)']
    selected_dim = st.selectbox('Select a dimension', dim_list)

    if selected_dim == 'Accuracy (WIP)':
        st.warning("We are working hard on preparing the Accuracy report, stay tuned!")
        acc_dim_list = ['Order of Events', 'Range and Outliers']
        selected_acc_dim = st.selectbox('Select a subdimension', acc_dim_list)

    st.markdown("<br><br>", unsafe_allow_html=True)

with st.sidebar.expander(f"💡 What is {selected_dim}?"):
    if selected_dim == "Uniqueness":
        st.write('''
            <span style="color: #db0069; font-weight: bold;">Uniqueness</span> ensures that each data entry is distinct and not duplicated within the dataset, maintaining the integrity of unique identifiers.
            <br><br>More details: [Confluence](https://omico.atlassian.net/wiki/spaces/RWD/pages/117866498/CaSP+Data+Quality+Architecture+DQv2#Uniqueness)
            ''', unsafe_allow_html=True)
    elif selected_dim == "Completeness":
        st.write('''
            <span style="color: #00c9d3; font-weight: bold;">Completeness</span> measures the extent to which all required data elements are present, ensuring that no necessary information is missing.
            <br><br>More details: [Confluence](https://omico.atlassian.net/wiki/spaces/RWD/pages/117866498/CaSP+Data+Quality+Architecture+DQv2#Completeness)
            ''', unsafe_allow_html=True)
    elif selected_dim == "Validity":
        st.write('''
            <span style="color: #923bdf; font-weight: bold;">Validity</span> assesses whether the data conforms to predefined formats, rules, or constraints, ensuring it is accurate and usable according to the defined standards.
            <br><br>More details: [Confluence](https://omico.atlassian.net/wiki/spaces/RWD/pages/117866498/CaSP+Data+Quality+Architecture+DQv2#Validity)
            ''', unsafe_allow_html=True)


##############################
# Backend calculation
##############################

dfs_boolean = load_data(dict_filepath_dim, selected_dim)

if selected_dim == "Uniqueness":

    # Calculate percentage of records unique
    table_record_uniqueness = {}
    for table, df in dfs_boolean.items():
        # Calculate the number of unique records (records without duplicates)
        unique_record_count = df.shape[0] - df.is_duplicate.sum()
        # Calculate the total number of records
        total_record_count = df.shape[0]
        
        # Initialise the dictionary for the current table if not already initialised
        if table not in table_record_uniqueness:
            table_record_uniqueness[table] = {}
        
        # Assign values to the dictionary
        table_record_uniqueness[table]['Unique records'] = unique_record_count
        table_record_uniqueness[table]['Total records'] = total_record_count
        table_record_uniqueness[table]['Record uniqueness'] = (unique_record_count / total_record_count) * 100

    # Convert dict to df
    df_record_uniqueness = pd.DataFrame(table_record_uniqueness).T
    # Ensure the first two columns are numeric and convert them to int
    for col in df_record_uniqueness.columns[:2]:
        df_record_uniqueness[col] = pd.to_numeric(df_record_uniqueness[col], errors='coerce').fillna(0).astype(int)

    # Prepare final table for plotting
    data_record_uniqueness = df_record_uniqueness.iloc[:,-1].reset_index().rename(columns={'index': 'Table'})

    # Get total number of records that are unique for the entire dataset
    num_records = df_record_uniqueness['Total records'].sum()
    overall_record_uniqueness = df_record_uniqueness['Unique records'].sum()/num_records*100

    # Define overall record score
    overall_record_score = overall_record_uniqueness

    # Calculate percentage of patients unique
    table_patient_duplicated_id = {}
    table_patient_uniqueness = {}
    for table, df in dfs_boolean.items():
        # Get list of duplicated patients
        grouped_df = df.groupby('PNum')['is_duplicate'].sum()
        patient_duplicated_ids = grouped_df[grouped_df>0].index.tolist()
        # Assign values to dictionary
        table_patient_duplicated_id[table] = patient_duplicated_ids
        # Calculate the number of unique patients (patients without duplicates)
        unique_patient_count = (df.groupby('PNum')['is_duplicate'].sum() == 0).sum()
        # Calculate the total number of patients
        total_patient_count = df['PNum'].nunique()
        
        # Initialise the dictionary for the current table if not already initialised
        if table not in table_patient_uniqueness:
            table_patient_uniqueness[table] = {}
        
        # Assign values to the dictionary
        table_patient_uniqueness[table]['Unique patients'] = unique_patient_count
        table_patient_uniqueness[table]['Total patients'] = total_patient_count
        table_patient_uniqueness[table]['Patient uniqueness'] = (unique_patient_count / total_patient_count) * 100

    # Convert dict to df
    df_patient_uniqueness = pd.DataFrame(table_patient_uniqueness).T
    # Ensure the first two columns are numeric and convert them to int
    for col in df_patient_uniqueness.columns[:2]:
        df_patient_uniqueness[col] = pd.to_numeric(df_patient_uniqueness[col], errors='coerce').fillna(0).astype(int)

    # Prepare final table for plotting
    data_patient_uniqueness = df_patient_uniqueness.iloc[:,-1].reset_index().rename(columns={'index': 'Table'})

    # Overall patient uniqueness
    # Combine all unique patient IDs into a single list
    dataset_patient_duplicated_ids = set()
    for patient_ids in table_patient_duplicated_id.values():
        dataset_patient_duplicated_ids.update(patient_ids)
    dataset_patient_duplicated_ids = list(dataset_patient_duplicated_ids)

    # Count total number of unique patients and overall patient uniqueness
    dataset_total_patients = df_patient_uniqueness['Total patients'].max()
    dataset_unique_patients = dataset_total_patients-len(dataset_patient_duplicated_ids)
    overall_patient_uniqueness = dataset_unique_patients/dataset_total_patients*100

    # Define overall record score
    overall_patient_score = overall_patient_uniqueness

    # Output relevant metadata
    num_tables = df_record_uniqueness.shape[0]
    num_patients = dataset_total_patients

elif selected_dim == "Completeness":

    # Calculate percentage of cells populated by 1's for each table
    table_completeness = {}
    for sheet_name, df in dfs_boolean.items():
        df = df.set_index('PNum')
        table_completeness[sheet_name] = (df.sum().sum()/df.size)*100

    # Convert dictionary to DataFrame
    data_completeness = pd.DataFrame(list(table_completeness.items()), columns=['Table', 'Completeness'])

    # Rename variables with datasheet as prefix
    dfs_boolean_renamed = []
    for datasheet, df in dfs_boolean.items():
        df = df.set_index('PNum')
        df = df.rename(columns=lambda var: f"{datasheet}.{var}")
        dfs_boolean_renamed.append(df)

    # Concatenate all dataframes
    df_boolean = pd.concat(dfs_boolean_renamed, axis=1).fillna(0)

    # Calculate percentage of cells populated by 1's
    overall_completeness = (df_boolean.sum().sum()/df_boolean.size)*100

    # Define overall patient score
    overall_patient_score = overall_completeness

    overall_record_score = None

    # Output relevant metadata
    num_tables = data_completeness.shape[0]
    num_patients = df_boolean.shape[0]
    num_records = sum(df.shape[0] for df in dfs_boolean.values())

elif selected_dim == "Validity":

    # Assign validity score (0 or 1) per patient
    def validity_per_pat(df, req):
        if req == 'lowerbound':
            # If all records are valid for each patient
            df_out = (df.set_index('PNum') != 0).groupby(level=0).all().astype(int)
        elif req == 'upperbound':
            # If at least one record is valid for each patient
            df_out = (df.set_index('PNum') != 0).groupby(level=0).any().astype(int)
        return df_out

    # Binary validity per variable per patient
    dfs_boolean_patients= {}
    for table, df in dfs_boolean.items():
        # df
        dfs_boolean_patients[table] = validity_per_pat(df, req='lowerbound')
    
    # Calculate percentage of cells populated by 1's for each table
    table_validity = {}
    for sheet_name, df in dfs_boolean_patients.items():
        table_validity[sheet_name] = (df.sum().sum()/df.size)*100

    # Convert dictionary to DataFrame
    data_validity = pd.DataFrame(list(table_validity.items()), columns=['Table', 'Validity'])

    # Rename variables with datasheet as prefix
    dfs_boolean_patients_renamed = []
    for datasheet, df in dfs_boolean_patients.items():
        df = df.rename(columns=lambda var: f"{datasheet}.{var}")
        dfs_boolean_patients_renamed.append(df)

    # Concatenate all dataframes
    df_boolean_patients = pd.concat(dfs_boolean_patients_renamed, axis=1).fillna(1)

    # Calculate percentage of cells populated by 1's
    overall_validity = (df_boolean_patients.sum().sum()/df_boolean_patients.size)*100

    # Define overall patient score
    overall_patient_score = overall_validity

    overall_record_score = None

    # Output relevant metadata
    num_tables = data_validity.shape[0]
    num_patients = df_boolean_patients.shape[0]
    num_records = sum(df.shape[0] for df in dfs_boolean_patients.values())


##############################
# Dashboard Main Panel
##############################

col1, col2, col3 = st.columns([1, 4, 1.5], gap='medium')

with col1:

    st.markdown('#### Overall Score')

    if overall_record_score:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f'**Record {selected_dim.lower()}**')
        donut_chart1 = plot_donut(overall_record_score, selected_dim)
        st.altair_chart(donut_chart1)

    if overall_patient_score:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f'**Patient {selected_dim.lower()}**')
        donut_chart2 = plot_donut(overall_patient_score, selected_dim)
        st.altair_chart(donut_chart2)

with col2:

    st.markdown('#### Table Scores')
    if selected_dim == "Uniqueness":

        bar_chart1 = plot_barh(data_record_uniqueness, tablename='table', metric=selected_dim)
        bar_chart2 = plot_barh(data_patient_uniqueness, tablename='table', metric=selected_dim)

        # Default to Chart 1
        if 'selected_chart' not in st.session_state:
            st.session_state.selected_chart = "chart1"

        # Define button classes based on selected chart
        chart1_class = "button selected" if st.session_state.selected_chart == "chart1" else "button"
        chart2_class = "button selected" if st.session_state.selected_chart == "chart2" else "button"

        st.markdown(
            """
            <style>
            .stButton > button {
                margin-right: 5px;
                font-size: 16px;
                padding: 10px 20px;
                border-radius: 15px;
                border: none;
                cursor: pointer;
            }
            .stButton > button:hover {
                background-color: #9e024d; /* Background color on hover */
                color: #ffffff;
            }
            .stButton > button:focus:not(:active) {
                background-color: #9e024d !important; /* Background color when selected */
                color: #ffffff;
            }
            .stButton > button:active {
                background-color: #9e024d !important; /* Background color when active */
                color: #ffffff;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        # Create button columns
        button_col1, button_col2 = st.columns([1, 3])

        with button_col1:
            if st.button(f"Record {selected_dim.lower()}"):
                st.session_state.selected_chart = "chart1"

        with button_col2:
            if st.button(f"Patient {selected_dim.lower()}"):
                st.session_state.selected_chart = "chart2"

        # Render the selected chart
        if st.session_state.selected_chart == "chart1":
            st.plotly_chart(bar_chart1)
        elif st.session_state.selected_chart == "chart2":
            st.plotly_chart(bar_chart2)

        # Apply the correct CSS class to buttons
        col1.markdown(f"<style>.stButton > button {{background-color: {'#5a012c' if st.session_state.selected_chart == 'chart1' else '#5a012c'};}}</style>", unsafe_allow_html=True)
        col2.markdown(f"<style>.stButton > button {{background-color: {'#5a012c' if st.session_state.selected_chart == 'chart2' else '#5a012c'};}}</style>", unsafe_allow_html=True)

    elif selected_dim == "Completeness":

        bar_chart1 = plot_barh(data_completeness, tablename='table', metric=selected_dim)

        # Default to Chart 1
        if 'selected_chart' not in st.session_state:
            st.session_state.selected_chart = "chart1"

        # Define button classes based on selected chart
        chart1_class = "button selected" if st.session_state.selected_chart == "chart1" else "button"

        st.markdown(
            """
            <style>
            .stButton > button {
                margin-right: 5px;
                font-size: 16px;
                padding: 10px 20px;
                border-radius: 15px;
                border: none;
                cursor: pointer;
            }
            .stButton > button:hover {
                background-color: #03989f; /* Background color on hover */
                color: #ffffff;
            }
            .stButton > button:focus:not(:active) {
                background-color: #03989f !important; /* Background color when selected */
                color: #ffffff;
            }
            .stButton > button:active {
                background-color: #03989f !important; /* Background color when active */
                color: #ffffff;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        # Create button columns
        button_col1, button_col2 = st.columns([1, 2])

        with button_col1:
            if st.button(f"Patient {selected_dim.lower()}"):
                st.session_state.selected_chart = "chart1"

        # Render the selected chart
        if st.session_state.selected_chart == "chart1":
            st.plotly_chart(bar_chart1)

        # Apply the correct CSS class to buttons
        col1.markdown(f"<style>.stButton > button {{background-color: {'#005458' if st.session_state.selected_chart == 'chart1' else '#005458'};}}</style>", unsafe_allow_html=True)
    
    elif selected_dim == "Validity":

        bar_chart1 = plot_barh(data_validity, tablename='table', metric=selected_dim)

        # Default to Chart 1
        if 'selected_chart' not in st.session_state:
            st.session_state.selected_chart = "chart1"

        # Define button classes based on selected chart
        chart1_class = "button selected" if st.session_state.selected_chart == "chart1" else "button"

        st.markdown(
            """
            <style>
            .stButton > button {
                margin-right: 5px;
                font-size: 16px;
                padding: 10px 20px;
                border-radius: 15px;
                border: none;
                cursor: pointer;
            }
            .stButton > button:hover {
                background-color: #54019e; /* Background color on hover */
                color: #ffffff;
            }
            .stButton > button:focus:not(:active) {
                background-color: #54019e !important; /* Background color when selected */
                color: #ffffff;
            }
            .stButton > button:active {
                background-color: #54019e !important; /* Background color when active */
                color: #ffffff;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        # Create button columns
        button_col1, button_col2 = st.columns([1, 2])

        with button_col1:
            if st.button(f"Patient {selected_dim.lower()}"):
                st.session_state.selected_chart = "chart1"

        # Render the selected chart
        if st.session_state.selected_chart == "chart1":
            st.plotly_chart(bar_chart1)

        # Apply the correct CSS class to buttons
        col1.markdown(f"<style>.stButton > button {{background-color: {'#32005f' if st.session_state.selected_chart == 'chart1' else '#32005f'};}}</style>", unsafe_allow_html=True)
    
with col3:

    if selected_dim == 'Uniqueness':
        num_variables = 'n/a'
    elif selected_dim == 'Completeness':
        num_variables = df_boolean.shape[1]
    elif selected_dim == 'Validity':
        num_variables = df_boolean_patients.shape[1]
    elif 'Accuracy' in selected_dim:
        num_variables = 'n/a'
    else:
        num_variables = 'n/a'

    filename_rawdata = '20240515_Quantium_CaSP.xlsx'

    st.markdown(f"""
        <div style="
            background-color: #262730;
            padding: 20px;
            border-radius: 10px;
            color: #ffffff;
            height: 100%;
            display: flex;
            flex-direction: column;
            justify-content: flex-start;
            ">
            <h4>Data Selection</h4
            >
            <ul>
                <li>Source file: <span style="color: orange; font-size:14px;">{filename_rawdata}</span></li>
                <li>Dataset: <span style="color: orange;">{selected_dataset}</span></li>
                <li>Patient count: <span style="color: orange;">{"{:,}".format(num_patients)}</span></li>
                <li>Table count: <span style="color: orange;">{num_tables}</span></li>
                <li>Sum of records: <span style="color: orange;">{"{:,}".format(num_records)}</span></li>
                <li>Variable count: <span style="color: orange;">{num_variables}</span></li>
            </ul>
            <hr>
            <p><strong>Disclaimer</strong></p>
            <p>This is a draft report and for INTERNAL USE ONLY. Numbers do not reflect the latest available dataset in Progeny.</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    with st.expander('About', expanded=False):
        st.write('''
            This report provides a high-level overview of the data quality dimensions across various data tables available in CaSP/MoST Progeny.
            - Version: 0.1.0
            - Updated: 13/08/2024
            - Author: Melvyn Yap
            - Email: m.yap@omico.org.au
            ''')