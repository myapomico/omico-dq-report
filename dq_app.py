##############################
# Import necessary libraries
##############################

import streamlit as st
import pandas as pd
import plotly.express as px
import pdfkit
import plotly.io as pio
import base64
import datetime

##############################
# Page configuration
##############################

st.set_page_config(
    page_title="Data Quality Report",
    page_icon="☑️",
    layout="wide",
    initial_sidebar_state="expanded"
)

##############################
# Specify input data
##############################

dict_filepath_dim = {
    'Uniqueness': 'data/20240812_u_bool.xlsx',
    'Completeness': 'data/20240814_c_bool.xlsx',
    'Validity': 'data/20240812_v_bool.xlsx',
    'Accuracy_OE': 'data/20240812_a_oe_bool.xlsx',
    'Accuracy_RO': ''
}

##############################
# Define functions
##############################

@st.cache_data
def load_data(filepath):
    """Load data from the specified Excel file."""
    excel_file = pd.ExcelFile(filepath)
    return {sheet_name: pd.read_excel(filepath, sheet_name=sheet_name) for sheet_name in excel_file.sheet_names}

def plot_donut_plotly(score, title, selected_dim):
    """Create a donut chart using Plotly Express, with color matching the selected_dim and rounding percentage to 0 decimal."""
    colors = {
        'Completeness': '#02989E',
        'Uniqueness': '#9E024d',
        'Validity': '#711abe'
    }
    bg_colors = {
        'Completeness': 'rgba(3, 152, 159, 0.3)',
        'Uniqueness': 'rgba(158, 2, 77, 0.3)',
        'Validity': 'rgba(113, 26, 190, 0.3)'
    }

    dim_color = colors.get(selected_dim, '#999999')

    # Create a DataFrame for the data
    df = pd.DataFrame({
        'Metric': ['Qualified', 'Non-qualified'],
        'Score': [score, 100 - score]
    })

    # Create the donut chart
    fig = px.pie(
        df,
        names='Metric',
        values='Score',
        hole=0.8,
        color_discrete_sequence=[dim_color, bg_colors.get(selected_dim, '#333333')]
    )

    fig.update_traces(
        textinfo='none',
        hovertemplate='<b>%{label}</b><br>Score (%) = %{value:.2f}',  # Custom hover template
    )

    fig.update_layout(
        showlegend=False,
        height=180,
        width=180,
        margin=dict(t=0, b=10, l=0, r=0),
        annotations=[dict(
            text=f"{score:.0f}%",
            x=0.5, y=0.5,
            font_size=40,
            showarrow=False,
            font=dict(color=dim_color, family='Arial Black')
        )],
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )

    return fig

def plot_barh(data, metric, chart_title=None):
    """Create a horizontal bar chart for the specified metric."""
    colors = {
        'Completeness': '#02989E',
        'Uniqueness': '#9E024d',
        'Validity': '#711abe'
    }
    bg_colors = {
        'Completeness': 'rgba(3, 152, 159, 0.3)',
        'Uniqueness': 'rgba(158, 2, 77, 0.3)',
        'Validity': 'rgba(113, 26, 190, 0.3)'
    }
    status_labels = {
        'Completeness': ('Complete', 'Null'),
        'Uniqueness': ('Unique', 'Duplicate'),
        'Validity': ('Valid', 'Invalid')
    }

    metric_color = colors.get(metric, '#999999')
    status_positive, status_negative = status_labels.get(metric, ('Positive', 'Negative'))

    # Summarize data
    data_summary = data.copy()
    data_summary.columns = [status_positive if col != 'Table' else col for col in data.columns]
    data_summary[status_negative] = 100 - data_summary[status_positive]
    data_summary = data_summary.melt(id_vars='Table', var_name='Status', value_name='Percentage')

    # Set the chart title
    if chart_title is None:
        chart_title = f"{metric} Bar Chart"

    # Create bar chart
    fig = px.bar(
        data_summary, y='Table', x='Percentage', 
        color='Status', orientation='h', 
        color_discrete_sequence=[metric_color, bg_colors.get(metric, '#999999')],
        title=chart_title
    )
    
    fig.update_layout(
        xaxis_title=f"{metric} percentage",
        yaxis_title="",
        plot_bgcolor='rgba(0, 0, 0, 0)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        font=dict(color='black'),
        height=200 + (data.shape[0] * 16),
        margin=dict(l=200, r=10, t=80, b=40), 
        xaxis=dict(
            showgrid=True,
            gridcolor='#444444',
            color='black',
            tickvals=[0, 20, 40, 60, 80],
            range=[0, 100]
        ),
        yaxis=dict(color='black'),
        barmode='stack'
    )
    fig.update_yaxes(autorange="reversed", ticksuffix="  ")

    # Add percentage annotations
    for _, row in data_summary[data_summary['Status'] == status_positive].iterrows():
        # Determine text color based on bar position (inside bar -> white, outside bar -> black)
        text_color = "white" if row['Percentage'] > 50 else "black"
        fig.add_annotation(
            x=row['Percentage'],
            y=row['Table'],
            text=f"{row['Percentage']:.1f}%",
            showarrow=False,
            font=dict(size=12, color=text_color), 
            xanchor='right',
            yanchor='middle'
        )

    return fig

def calculate_uniqueness(dfs):
    """Calculate uniqueness metrics for records and patients."""
    record_uniqueness, patient_uniqueness = {}, {}
    table_patient_duplicated_id = {}

    for table, df in dfs.items():
        # Record uniqueness
        unique_record_count = df.shape[0] - df['is_duplicate'].sum()
        total_record_count = df.shape[0]
        record_uniqueness[table] = {
            'Unique records': unique_record_count,
            'Total records': total_record_count,
            'Record uniqueness': (unique_record_count / total_record_count) * 100
        }

        # Patient uniqueness
        grouped_df = df.groupby('PNum')['is_duplicate'].sum()
        patient_duplicated_ids = grouped_df[grouped_df > 0].index.tolist()
        table_patient_duplicated_id[table] = patient_duplicated_ids
        unique_patient_count = (grouped_df == 0).sum()
        total_patient_count = df['PNum'].nunique()
        patient_uniqueness[table] = {
            'Unique patients': unique_patient_count,
            'Total patients': total_patient_count,
            'Patient uniqueness': (unique_patient_count / total_patient_count) * 100
        }

    # Convert dicts to dataframes
    df_record_uniqueness = pd.DataFrame(record_uniqueness).T
    df_patient_uniqueness = pd.DataFrame(patient_uniqueness).T

    # Ensure the first two columns are numeric and convert them to int
    for df in [df_record_uniqueness, df_patient_uniqueness]:
        df[df.columns[:2]] = df[df.columns[:2]].astype(int)

    # Calculate overall record uniqueness
    overall_record_uniqueness = df_record_uniqueness['Unique records'].sum() / df_record_uniqueness['Total records'].sum() * 100

    # Combine duplicated patient IDs from all tables
    dataset_patient_duplicated_ids = set()
    for patient_ids in table_patient_duplicated_id.values():
        dataset_patient_duplicated_ids.update(patient_ids)

    # Calculate overall patient uniqueness
    dataset_total_patients = df_patient_uniqueness['Total patients'].max()
    dataset_unique_patients = dataset_total_patients - len(dataset_patient_duplicated_ids)
    overall_patient_uniqueness = (dataset_unique_patients / dataset_total_patients) * 100

    # Calculate total number of patients
    num_patients_uniqueness = dataset_total_patients

    # Calculate total number of variables
    num_variables_uniqueness = 'n/a'

    return df_record_uniqueness, df_patient_uniqueness, overall_record_uniqueness, overall_patient_uniqueness, num_patients_uniqueness, num_variables_uniqueness

def calculate_completeness(dfs):
    """Calculate completeness metrics."""
    # Calculate percentage of cells populated by 1's for each table
    table_completeness = {}
    for sheet_name, df in dfs.items():
        df = df.set_index('PNum')
        completeness_percentage = (df.sum().sum() / df.size) * 100
        table_completeness[sheet_name] = completeness_percentage

    # Convert dictionary to DataFrame
    df_completeness = pd.DataFrame(list(table_completeness.items()), columns=['Table', 'Completeness'])

    # Rename variables with datasheet as prefix and concatenate all dataframes
    df_boolean = pd.concat(
        [df.set_index('PNum').rename(columns=lambda col: f"{sheet_name}.{col}") for sheet_name, df in dfs.items()],
        axis=1
    ).fillna(0)

    # Calculate overall completeness
    overall_completeness = (df_boolean.sum().sum() / df_boolean.size) * 100

    # Calculate total number of patients
    num_patients_completeness = df_boolean.shape[0]

    # Calculate total number of variables
    num_variables_completeness = df_boolean.shape[1]

    return df_completeness, overall_completeness, num_patients_completeness, num_variables_completeness

def calculate_validity(dfs):
    """Calculate validity metrics."""

    def validity_per_pat(df, req):
        df = df.set_index('PNum') != 0
        if req == 'lowerbound':
            df_out = df.groupby(level=0).all().astype(int)
        else:  # 'upperbound' case
            df_out = df.groupby(level=0).any().astype(int)
        return df_out
    
    # Calculate binary validity per variable per patient for each table
    dfs_patients = {table: validity_per_pat(df, 'lowerbound') for table, df in dfs.items()}
    
    # Calculate percentage of valid cells for each table
    table_validity = {
        sheet_name: (df.sum().sum() / df.size) * 100 for sheet_name, df in dfs_patients.items()
    }
    
    # Convert dictionary to DataFrame
    df_validity = pd.DataFrame(list(table_validity.items()), columns=['Table', 'Validity'])

    # Rename variables with datasheet as prefix and concatenate all dataframes
    df_patients_concat = pd.concat(
        [df.rename(columns=lambda col: f"{sheet_name}.{col}") for sheet_name, df in dfs_patients.items()],
        axis=1
    ).fillna(1)

    # Calculate overall validity percentage
    overall_validity = (df_patients_concat.sum().sum() / df_patients_concat.size) * 100

    # Calculate total number of patients
    num_patients_validity = df_patients_concat.shape[0]

    # Calculate total number of variables
    num_variables_validity = df_patients_concat.shape[1]

    return df_validity, overall_validity, num_patients_validity, num_variables_validity

def render_sidebar():
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
            st.selectbox('Select a subdimension', ['Order of Events', 'Range and Outliers'])

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<hr>", unsafe_allow_html=True)

    return selected_dim

def render_expander(selected_dim):
    """Render the sidebar expander with explanations."""
    with st.sidebar.expander(f"💡 What is {selected_dim}?"):
        explanations = {
            "Uniqueness": '''
                <span style="color: #db0069; font-weight: bold;">Uniqueness</span> ensures that each data entry is distinct and not duplicated within the dataset, maintaining the integrity of unique identifiers.
                <br><br>More details: [Confluence](https://omico.atlassian.net/wiki/spaces/RWD/pages/117866498/CaSP+Data+Quality+Architecture+DQv2#Uniqueness)
            ''',
            "Completeness": '''
                <span style="color: #00c9d3; font-weight: bold;">Completeness</span> measures the extent to which all required data elements are present, ensuring that no necessary information is missing.
                <br><br>More details: [Confluence](https://omico.atlassian.net/wiki/spaces/RWD/pages/117866498/CaSP+Data+Quality+Architecture+DQv2#Completeness)
            ''',
            "Validity": '''
                <span style="color: #923bdf; font-weight: bold;">Validity</span> assesses whether the data conforms to predefined formats, rules, or constraints, ensuring it is accurate and usable according to the defined standards.
                <br><br>More details: [Confluence](https://omico.atlassian.net/wiki/spaces/RWD/pages/117866498/CaSP+Data+Quality+Architecture+DQv2#Validity)
            '''
        }
        st.write(explanations.get(selected_dim, "Explanation not available."), unsafe_allow_html=True)
        
def render_main_panel(selected_dim, dfs, overall_scores, num_patients, num_variables):
    """Render the main panel with overall and table-specific scores."""
    col1, col2, col3 = st.columns([1, 4, 1.5], gap='medium')

    with col1:
        st.markdown("""
            <h3 style="margin-bottom: 5px;">Overall Scores</h3>
            <hr style="margin-top: 0; border: 1px solid #333;">
        """, unsafe_allow_html=True)

        # Retrieve precomputed scores based on selected dimension
        overall_record_score, overall_patient_score = overall_scores.get(selected_dim, (None, None))

        if overall_record_score:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f'**Record {selected_dim.lower()}**')
            st.plotly_chart(plot_donut_plotly(overall_record_score, f"Record {selected_dim}", selected_dim))

        if overall_patient_score:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f'**Patient {selected_dim.lower()}**')
            st.plotly_chart(plot_donut_plotly(overall_patient_score, f"Patient {selected_dim}", selected_dim))

    with col2:
        st.markdown("""
            <h3 style="margin-bottom: 5px;">Table Scores</h3>
            <hr style="margin-top: 0; border: 1px solid #333;">
        """, unsafe_allow_html=True)

        if selected_dim == "Uniqueness":
            df_record_uniqueness, df_patient_uniqueness, _, _, _, _ = dfs[selected_dim]
            
            # Radio button for selecting the type of uniqueness to display with custom styling
            st.markdown("""
                <style>
                .radio-container {
                    display: flex;
                    justify-content: center;
                    gap: 10px;
                }
                .radio-container label {
                    display: inline-block;
                    padding: 10px 20px;
                    border-radius: 20px;
                    border: 2px solid #9E024d;
                    cursor: pointer;
                    font-weight: bold;
                    color: white;
                }
                .radio-container input[type="radio"] {
                    display: none;
                }
                .radio-container input[type="radio"]:checked + label {
                    background-color: #9E024d;
                    color: white;
                }
                </style>
            """, unsafe_allow_html=True)

            # Adding a non-empty label and hiding it
            uniqueness_type = st.radio(
                "Uniqueness Type Selection",
                ("Record Uniqueness", "Patient Uniqueness"),
                index=0,
                key="uniqueness_type",
                label_visibility="collapsed"  # This hides the label but still satisfies the accessibility requirement
            )

            # Display one of the two bar charts based on the selected option
            if uniqueness_type == "Record Uniqueness":
                st.plotly_chart(plot_barh(df_record_uniqueness.iloc[:, -1].reset_index().rename(columns={'index': 'Table'}), selected_dim, uniqueness_type))
            else:
                st.plotly_chart(plot_barh(df_patient_uniqueness.iloc[:, -1].reset_index().rename(columns={'index': 'Table'}), selected_dim, uniqueness_type))

        elif selected_dim == "Completeness":
            df_completeness, _, _, _ = dfs[selected_dim]
            st.plotly_chart(plot_barh(df_completeness, selected_dim, f'Patient {selected_dim}'))
        elif selected_dim == "Validity":
            df_validity, _, _, _ = dfs[selected_dim]
            st.plotly_chart(plot_barh(df_validity, selected_dim, f'Patient {selected_dim}'))

    with col3:
        # num_variables = dfs[selected_dim][0].shape[1] if selected_dim in ["Completeness", "Validity"] else 'n/a'

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
                <h3 style="color: white;">Data Selection</h3>
                <ul>
                    <li>Source file: <span style="color: orange; word-wrap: break-word; word-break: break-all;">20240515_Quantium_CaSP.xlsx</span></li>
                    <li>Dataset: <span style="color: orange;">CaSP</span></li>
                    <li>Patient count: <span style="color: orange;">{"{:,}".format(num_patients[selected_dim])}</span></li>
                    <li>Table count: <span style="color: orange;">{len(dfs[selected_dim][0])}</span></li>
                    <li>Variable count: <span style="color: orange;">{num_variables[selected_dim]}</span></li>
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
                - Version: 0.1.1
                - Updated: 15/08/2024
                - Author: Melvyn Yap
                - Email: m.yap@omico.org.au
            ''')

def export_pdf(html_list):
    """Combine HTML content from all pages into a single PDF."""
    combined_html = "\n".join(html_list)
    
    # Wrap the HTML content with some padding
    combined_html = f"""
    <div style="padding-top: 0cm; padding-right: 2cm; padding-bottom: 1cm; padding-left: 2cm;">
        {combined_html}
    </div>
    """
    
    # Get the current date in YYYYMMDD format
    date_prefix = datetime.datetime.now().strftime("%Y%m%d")
    
    # Add the date prefix to the filename
    pdf_filename = f"{date_prefix}_data_quality_report.pdf"

    # Try using the wkhtmltopdf-binary included with the package
    try:
        config = pdfkit.configuration()  # No need to specify the path
    except Exception as e:
        raise OSError("wkhtmltopdf not found and could not be configured automatically.") from e

    # PDF options to set the layout to landscape and include page numbers in the footer
    options = {
        'page-size': 'A4',
        'orientation': 'Landscape',
        'margin-top': '1cm',
        'margin-right': '1cm',
        'margin-bottom': '1cm',
        'margin-left': '1cm',
        'footer-right': 'Page [page] of [topage]',
        'footer-font-size': '10',
        'footer-spacing': '5',
        'encoding': 'UTF-8',
    }

    # Generate the PDF with pdfkit
    pdfkit.from_string(combined_html, pdf_filename, configuration=config, options=options)

    return pdf_filename

def plot_to_base64(fig):
    """Convert a Plotly figure to a base64 encoded image."""
    img_bytes = pio.to_image(fig, format='png')
    img_base64 = base64.b64encode(img_bytes).decode('ascii')
    return f"data:image/png;base64,{img_base64}"

def render_main_panel_to_html(selected_dim, dfs, overall_scores, num_patients, num_variables):
    """Render the main panel to HTML instead of directly to Streamlit."""
    # Apply Arial font globally
    html_content = """
        <style>
            body, h4, h5, p, ul, li {
                font-family: Arial, sans-serif;
            }
            body {
                background-color: #eeeeee;  /* Light grey background */
            }
        </style>
    """

    # Add a page break before each section
    html_content += '<div style="page-break-after: always;"></div>'

    html_content += '<div style="padding-top: 1cm; padding-right: 0cm; padding-bottom: 1cm; padding-left: 0cm;">'

    # Column 1: Overall Score
    html_content += '<div style="width: 14%; float: left; padding-right: 2%;">'
    html_content += '<h3>Overall Score</h3>'
    html_content += '<hr style="border: 1px solid #262730;">'

    overall_record_score, overall_patient_score = overall_scores.get(selected_dim, (None, None))

    if overall_record_score:
        html_content += f'<p><strong>Record {selected_dim.lower()}</strong></p>'
        fig_record = plot_donut_plotly(overall_record_score, f"Record {selected_dim}", selected_dim)
        img_data = plot_to_base64(fig_record)
        html_content += f'<img src="{img_data}" alt="Record {selected_dim} Chart" style="width: 100%;">'

    if overall_patient_score:
        html_content += f'<p><strong>Patient {selected_dim.lower()}</strong></p>'
        fig_patient = plot_donut_plotly(overall_patient_score, f"Patient {selected_dim}", selected_dim)
        img_data = plot_to_base64(fig_patient)
        html_content += f'<img src="{img_data}" alt="Patient {selected_dim} Chart" style="width: 100%;">'
    
    # Add explanation text at the bottom of Column 1
    explanations = {
        "Uniqueness": '''
            <span style="color: #9E024d; font-weight: bold;">Uniqueness</span> ensures that each data entry is distinct and not duplicated within the dataset, maintaining the integrity of unique identifiers.
            <br><br>More details: <a href="https://omico.atlassian.net/wiki/spaces/RWD/pages/117866498/CaSP+Data+Quality+Architecture+DQv2#Uniqueness">Confluence</a>
        ''',
        "Completeness": '''
            <span style="color: #02989E; font-weight: bold;">Completeness</span> measures the extent to which all required data elements are present, ensuring that no necessary information is missing.
            <br><br>More details: <a href="https://omico.atlassian.net/wiki/spaces/RWD/pages/117866498/CaSP+Data+Quality+Architecture+DQv2#Completeness">Confluence</a>
        ''',
        "Validity": '''
            <span style="color: #711abe; font-weight: bold;">Validity</span> assesses whether the data conforms to predefined formats, rules, or constraints, ensuring it is accurate and usable according to the defined standards.
            <br><br>More details: <a href="https://omico.atlassian.net/wiki/spaces/RWD/pages/117866498/CaSP+Data+Quality+Architecture+DQv2#Validity">Confluence</a>
        '''
    }
    
    explanation_html = explanations.get(selected_dim, "Explanation not available.")
    html_content += f'''
        <div style="
            margin-top: 20px;
            background-color: #d9d9d9;
            padding: 20px;
            display: flex;
            flex-direction: column;
            justify-content: flex-start;
        ">
            {explanation_html}
        </div>
    '''

    html_content += '</div>'

    # Column 2: Table Scores
    html_content += '<div style="width: 64%; float: left; padding-right: 0cm;">'
    html_content += '<h3>Table Scores</h3>'
    html_content += '<hr style="border: 1px solid #262730;">'

    if selected_dim == "Uniqueness":
        df_record_uniqueness, df_patient_uniqueness, _, _, _, _ = dfs[selected_dim]

        # Display Record Uniqueness bar chart
        fig_barh_record = plot_barh(df_record_uniqueness.iloc[:, -1].reset_index().rename(columns={'index': 'Table'}), selected_dim, f'Record {selected_dim}')
        img_data_record = plot_to_base64(fig_barh_record)
        html_content += f'<img src="{img_data_record}" alt="Record Uniqueness Chart" style="width: 100%;">'

        # Display Patient Uniqueness bar chart
        fig_barh_patient = plot_barh(df_patient_uniqueness.iloc[:, -1].reset_index().rename(columns={'index': 'Table'}), selected_dim, f'Patient {selected_dim}')
        img_data_patient = plot_to_base64(fig_barh_patient)
        html_content += f'<img src="{img_data_patient}" alt="Patient Uniqueness Chart" style="width: 100%;">'

    elif selected_dim == "Completeness":
        df_completeness, _, _, _ = dfs[selected_dim]
        fig_barh = plot_barh(df_completeness, selected_dim, f'Patient {selected_dim}')
        img_data = plot_to_base64(fig_barh)
        html_content += f'<img src="{img_data}" alt="Completeness Chart" style="width: 100%;">'

    elif selected_dim == "Validity":
        df_validity, _, _, _ = dfs[selected_dim]
        fig_barh = plot_barh(df_validity, selected_dim, f'Patient {selected_dim}')
        img_data = plot_to_base64(fig_barh)
        html_content += f'<img src="{img_data}" alt="Validity Chart" style="width: 100%;">'

    html_content += '</div>'

    html_content += '</div>'

    # Column 3: Data Selection and Info
    html_content += '<div style="width: 20%; float: right; height: 500px;">'
    # num_variables = dfs[selected_dim][0].shape[1] if selected_dim in ["Completeness", "Validity"] else 'n/a'

    html_content += f"""
        <div style="
            background-color: #262730;
            padding: 20px;
            padding-bottom: 40px;
            border-radius: 10px;
            color: #ffffff;
            height: 100%;
            display: flex;
            flex-direction: column;
            justify-content: flex-start;
            ">
            <h3>Data Selection</h3>
            <p>Source file: <span style="color: orange; word-wrap: break-word; word-break: break-all;">20240515_Quantium_CaSP.xlsx</span></p>
            <p>Dataset: <span style="color: orange;">CaSP</span></p>
            <p>Patient count: <span style="color: orange;">{"{:,}".format(num_patients[selected_dim])}</span></p>
            <p>Table count: <span style="color: orange;">{len(dfs[selected_dim][0])}</span></p>
            <p>Variable count: <span style="color: orange;">{num_variables[selected_dim]}</span></p>
            <hr>
            <p><strong>Disclaimer</strong></p>
            <p>This is a draft report and for INTERNAL USE ONLY. Numbers do not reflect the latest available dataset in Progeny.</p>
        </div>
    """
    html_content += '</div>'

    # Clear float
    html_content += '<div style="clear: both;"></div>'

    return html_content

##############################
# Main execution
##############################

# Load data for all dimensions
dfs = {
    'Uniqueness': calculate_uniqueness(load_data(dict_filepath_dim['Uniqueness'])),
    'Completeness': calculate_completeness(load_data(dict_filepath_dim['Completeness'])),
    'Validity': calculate_validity(load_data(dict_filepath_dim['Validity']))
}

# Precompute overall scores
overall_scores = {
    'Uniqueness': (dfs['Uniqueness'][2], dfs['Uniqueness'][3]),
    'Completeness': (None, dfs['Completeness'][1]),
    'Validity': (None, dfs['Validity'][1])
}

# Precompute number of patients
num_patients = {
    'Uniqueness': dfs['Uniqueness'][4],
    'Completeness': dfs['Completeness'][2],
    'Validity': dfs['Validity'][2]
}

# Precompute number of variables
num_variables = {
    'Uniqueness': dfs['Uniqueness'][5],
    'Completeness': dfs['Completeness'][3],
    'Validity': dfs['Validity'][3]
}

# Render the sidebar and expander
selected_dim = render_sidebar()
render_expander(selected_dim)

# Capture HTML from each page
html_list = [
    """
    <div style="width: 100%; padding: 20px; font-family: Arial, sans-serif;">
        <h1 style="text-align: center; color: #434343; font-size: 80px; margin-bottom: 40px; padding-top: 160px;">
            Data Quality Report
        </h1>
        <p style="text-align: center; font-size: 18px; margin-bottom: 20px;">
            This report provides a high-level overview of the data quality dimensions across various data tables available in CaSP/MoST Progeny.
        </p>
        <div style="text-align: center; font-size: 16px; margin-top: 50px;">
            <p><strong>Version:</strong> 0.1.1</p>
            <p><strong>Updated:</strong> 15/08/2024</p>
            <p><strong>Author:</strong> Melvyn Yap</p>
            <p><strong>Email:</strong> <a href="mailto:m.yap@omico.org.au">m.yap@omico.org.au</a></p>
        </div>
    </div>
    """
]
for dim in ['Uniqueness', 'Completeness', 'Validity']:
    html_list.append(render_main_panel_to_html(dim, dfs, overall_scores, num_patients, num_variables))

# Display the selected dimension
render_main_panel(selected_dim, dfs, overall_scores, num_patients, num_variables)

# Button to export all pages to a PDF
st.sidebar.markdown("<hr>", unsafe_allow_html=True)
if st.sidebar.button("Export to PDF"):
    pdf_file = export_pdf(html_list)
    st.sidebar.success(f"PDF ready for download: {pdf_file}")
    st.sidebar.download_button(label="Download PDF", data=open(pdf_file, 'rb'), file_name=pdf_file, mime="application/pdf")
