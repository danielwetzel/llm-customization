import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import os
import sys

def init_session_states():
    if 'metric' not in st.session_state:
        st.session_state.metric = 'actual_emissions_per_10k_prompts'

    if 'degree_list' not in st.session_state:
        st.session_state.degree_list = [1]

    if 'curr_page' not in st.session_state:
        st.session_state.curr_page = 'vllm_tests'


@st.cache_data
def load_csv_data(type='emission_regression'):
    """
    Load data from a csv file
    """

    df = pd.read_csv(f'notebooks/results/data/{type}.csv')

    return df


def label_func(input): 

    label_dict = {
        'actual_emissions_per_10k_prompts': 'Emissions per 10,000 Prompts',
        'actual_cpu_energy_per_10k_prompts': 'CPU Energy per 10,000 Prompts',
        'actual_gpu_energy_per_10k_prompts': 'GPU Energy per 10,000 Prompts',
        'actual_ram_energy_per_10k_prompts': 'RAM Energy per 10,000 Prompts',
        'actual_non_idle_gpu_energy_per_10k_prompts' : 'Idle GPU Energy per 10,000 Prompts', 
        'actual_idle_gpu_energy_per_10k_prompts' : 'Non-Idle GPU Energy per 10,000 Prompts', 
        'actual_emissions_per_1M_out_tok': 'Emissions per 1M Output Tokens',
    }


    return label_dict[input]


# Function to create charts for each test type
def create_chart(df, test_type, metric = 'actual_emissions_per_10k_prompts', remove_x_title=False):
    llama2_note = 'Note: Emissions normalized to number of output tokens for Llama2 because the Llama2 and Llama3 output differed drastically'
    chart_data = df[df['test_type'] == test_type]

    if test_type == 'Output-tok': 
        x_title = 'Average Output Tokens per Prompt'
        x_data = 'avg_out_tok'
    elif test_type == 'Input-tok':
        x_title = 'Average Input Tokens per Prompt'
        x_data = 'avg_in_tok'
    elif test_type == 'Llama2 Params':
        x_title = 'Parameters (billions)'
        x_data = 'parameters'
    elif test_type == 'Llama3 Params':
        x_title = 'Parameters (billions)'
        x_data = 'parameters'
    
    scatter = alt.Chart(chart_data).mark_circle(size=100).encode(
        x=alt.X(x_data, title=x_title),
        y=alt.Y(metric, title=label_func(metric)),
        color = alt.Color('test_type:N', title='Test Type').sort(df['test_type'].unique()),
        tooltip=[
            alt.Tooltip('parameters', title='Parameters (billions)'),
            alt.Tooltip(metric, title=label_func(metric)),
            alt.Tooltip('pred_emissions_per_10k_prompts', title='Predicted Emissions per 10,000 Prompts'),
            alt.Tooltip('avg_out_tok', title='Average Output Tokens per Prompt'),
            alt.Tooltip('avg_in_tok', title='Average Input Tokens per Prompt'),
            alt.Tooltip('num_examples', title='Number of Examples'),
            alt.Tooltip('num_prompts', title='Number of Prompts'),
            alt.Tooltip('model_type', title='Model Type'),
            alt.Tooltip('test_type', title='Test Type'),
            alt.Tooltip('test_type', title='Test Type'),
        ]
    ).properties(
        title=f'{label_func(metric)} for {test_type}',
    )

    # Create line plots for predicted emissions
    line = alt.Chart(chart_data).mark_line().encode(
        x=alt.X(x_data, title=x_title),
        y=alt.Y(metric, title=label_func(metric)),
        color = alt.Color('test_type:N', title='Test Type').sort(df['test_type'].unique()),
        tooltip=[
            alt.Tooltip('parameters', title='Parameters (billions)'),
            alt.Tooltip(metric, title=label_func(metric)),
            alt.Tooltip('pred_emissions_per_10k_prompts', title='Predicted Emissions per 10,000 Prompts'),
            alt.Tooltip('avg_out_tok', title='Average Output Tokens per Prompt'),
            alt.Tooltip('avg_in_tok', title='Average Input Tokens per Prompt'),
            alt.Tooltip('num_examples', title='Number of Examples'),
            alt.Tooltip('num_prompts', title='Number of Prompts'),
            alt.Tooltip('model_type', title='Model Type'),
            alt.Tooltip('test_type', title='Test Type'),
        ]
    )
    
    # Add note for Llama2
    if test_type == 'Llama2 Params':
        chart_data['Note'] = llama2_note
        #print(chart_data)
        scatter = scatter.encode(
            tooltip=[
                alt.Tooltip('parameters', title='Parameters (billions)'),
                alt.Tooltip(metric, title=label_func(metric)),
                alt.Tooltip('pred_emissions_per_10k_prompts', title='Predicted Emissions per 10,000 Prompts'),
                alt.Tooltip('avg_out_tok', title='Average Output Tokens per Prompt'),
                alt.Tooltip('avg_in_tok', title='Average Input Tokens per Prompt'),
                alt.Tooltip('num_examples', title='Number of Examples'),
                alt.Tooltip('num_prompts', title='Number of Prompts'),
                alt.Tooltip('model_type', title='Model Type'),
                alt.Tooltip('test_type', title='Test Type'),
                alt.Tooltip('Note', title='Normalization Note')
            ]
        )
    
    if remove_x_title:
        scatter = scatter.encode(
            x=alt.X(x_data, title=None, axis=None),
        )
        line = line.encode(
            x=alt.X(x_data, title=None, axis=None),
        )


    return scatter + line

def create_reg_chart(df, test_type, metric='actual_emissions_per_10k_prompts',  degree_list = [1, 2, 5], remove_x_title=False):

    llama2_note = 'Note: Emissions normalized to number of output tokens for Llama2 because the Llama2 and Llama3 output differed drastically'
    chart_data = df[df['test_type'] == test_type]

    if test_type == 'Output-tok': 
        x_title = 'Average Output Tokens per Prompt'
        x_data = 'avg_out_tok'
    elif test_type == 'Input-tok':
        x_title = 'Average Input Tokens per Prompt'
        x_data = 'avg_in_tok'
    elif test_type == 'Llama2 Params':
        x_title = 'Parameters (billions)'
        x_data = 'parameters'
    elif test_type == 'Llama3 Params':
        x_title = 'Parameters (billions)'
        x_data = 'parameters'
    elif test_type == 'framework_comp_vllm':
        x_title = 'Parameters (billions)'
        x_data = 'parameters'
    elif test_type == 'Output-tok-vllm':
        x_title = 'Average Output Tokens per Prompt'
        x_data = 'avg_out_tok'


    scatter_base = alt.Chart(chart_data).mark_circle(size=100).encode(
        x=alt.X(x_data, title=x_title),
        y=alt.Y(metric, title=label_func(metric)),
        #color = alt.Color('test_type:N', title='Test Type').sort(df['test_type'].unique()),
        tooltip=[
            alt.Tooltip('parameters', title='Parameters (billions)'),
            alt.Tooltip(metric, title=label_func(metric)),
            alt.Tooltip('pred_emissions_per_10k_prompts', title='Predicted Emissions per 10,000 Prompts'),
            alt.Tooltip('avg_out_tok', title='Average Output Tokens per Prompt'),
            alt.Tooltip('avg_in_tok', title='Average Input Tokens per Prompt'),
            alt.Tooltip('num_examples', title='Number of Examples'),
            alt.Tooltip('num_prompts', title='Number of Prompts'),
            alt.Tooltip('model_type', title='Model Type'),
            alt.Tooltip('test_type', title='Test Type'),
            alt.Tooltip('test_type', title='Test Type'),
        ]
    ).properties(
        title=f'{label_func(metric)} for {test_type}',
    )

    polynomial_fit = [
        scatter_base.transform_regression(
            x_data, metric, method="poly", order=order, as_=[x_data, str(order)]
        )
        .mark_line()
        .transform_fold([str(order)], as_=["degree", metric])
        .encode(alt.Color("degree:N", title="Regression Degree", legend=None))
        for order in degree_list
    ]

    return alt.layer(scatter_base, *polynomial_fit).interactive()


def sidebar():
    
    with st.sidebar:

        st.image("pages/img/logo.png", use_column_width=True)

        st.divider()

        st.title("Navigation")
        st.page_link(page="pages/vllm_tests.py", label="vLLM Tests", icon="⭐")
        st.page_link(page="pages/initial_tests.py", label="Early Tests", icon="⏳")

        st.write("")
        st.divider()
        st.write("")

        if st.session_state.curr_page == 'initial_tests':
            st.info("""
                    The tests on the selected page were performed during the initial 
                    investigative phase of the research using a suboptimal setup.
                    \n 
                    If you want to review the latest tests, check out the vLLM Tests page.
                    """)


if __name__ == "__main__":
    init_session_states()
    sidebar()