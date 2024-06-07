import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import os

st.set_page_config(page_title="LLM Emission Tests", page_icon=':seedling:', layout="wide")


if 'metric' not in st.session_state:
    st.session_state.metric = 'actual_emissions_per_10k_prompts'

if 'degree_list' not in st.session_state:
    st.session_state.degree_list = [1]

if 'curr_page' not in st.session_state:
    st.session_state.curr_page = 'overview_page'

@st.cache_data
def load_csv_data(type='emission_regression'):
    """
    Load data from a csv file
    """

    df = pd.read_csv(f'results/{type}.csv')

    return df


def label_func(input): 

    label_dict = {
        'actual_emissions_per_10k_prompts': 'Emissions per 10,000 Prompts',
        'actual_cpu_energy_per_10k_prompts': 'CPU Energy per 10,000 Prompts',
        'actual_gpu_energy_per_10k_prompts': 'GPU Energy per 10,000 Prompts',
        'actual_ram_energy_per_10k_prompts': 'RAM Energy per 10,000 Prompts',
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



def overview_page(df):

    st.subheader("Overview")
    st.write("This section provides an overview of all the impacts on the interference emissions.")


    # Create charts for each test type
    stacked_charts = []
    stacked_df = df[df.num_examples != 90]
    for test_type in stacked_df['test_type'].unique():
        stacked_charts.append(create_chart(stacked_df, test_type, metric=st.session_state.metric, remove_x_title=True))


    # Create a combined chart with overlays for all emission types
    combined_chart = alt.layer(*stacked_charts).resolve_scale(
        x='independent'
    ).properties(
        title='Emissions per Ten-Thousand Prompts by Test Type',
        height=700
    )

    st.altair_chart(combined_chart, use_container_width=True, theme="streamlit")

    st.write(df)


def output_tok_page(df):

    st.subheader("Output Token Impact")
    st.write("This section visualizes the impact of the output token on the interference emissions.")

    output_chart = create_chart(df, test_type='Output-tok').properties(height=700)

    st.altair_chart(output_chart, use_container_width=True, theme="streamlit")

    st.write(df[df.test_type=='Output-tok'])


def input_tok_page(df):




    st.subheader("Input Token Impact")
    st.write("This section visualizes the impact of the input token on the interference emissions.")

    #input_chart = create_chart(df, test_type='Input-tok').properties(height=700)

    vis, select = st.columns([9, 1])

    with select:

        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.session_state.degree_list[0] = st.selectbox("Polynomial Degree", [1, 2, 3, 4, 5], index=0)

    with vis:
    
        input_chart = create_reg_chart(df, test_type='Input-tok', metric=st.session_state.metric, degree_list=st.session_state.degree_list).properties(height=700)
        st.altair_chart(input_chart, use_container_width=True, theme="streamlit")


    st.write(df[df.test_type=='Input-tok'])






def params_page(df):

    st.subheader("Model Parameter Impact")
    st.write("This section visualizes the impact of the model parameters on the interference emissions.")

    tests = ['Llama2 Params', 'Llama3 Params']

    params_df = df[df['test_type'].isin(tests)]
    params_charts = []

    for test_type in params_df['test_type'].unique():
        params_charts.append(create_chart(params_df, test_type))


    # Create a combined chart with overlays for all emission types
    params_chart = alt.layer(*params_charts).resolve_scale(
        #x='independent'
    ).properties(
        title='Emissions for different Parameter Sizes',
        height=700
    )

    st.altair_chart(params_chart, use_container_width=True, theme="streamlit")

    st.write(params_df)


def sidebar():

    metrics = ['actual_emissions_per_10k_prompts', 'actual_cpu_energy_per_10k_prompts', 'actual_gpu_energy_per_10k_prompts', 'actual_ram_energy_per_10k_prompts']

    with st.sidebar:

        st.header("Filter Selection")

        st.session_state.metric = st.selectbox("Review Metric", metrics, format_func=label_func, index=0)


def main():

    st.title("LLM Emission Tests 🌍🌱")

    st.caption("This is a dashboard to visualize the results of the LLM emission tests.")

    st.subheader("", divider='grey')

    df = load_csv_data('emission_regression')

    sidebar()

    overview, output_tok, input_tok, params = st.tabs(["Overview", "Output Token Impact", "Input Token Impact", "Model Parameter Impact"])


    with overview:

        
        overview_page(df)

    

    with output_tok:

            
        output_tok_page(df)



    with input_tok:


        input_tok_page(df)



    with params:


        params_page(df)

    
    



if __name__ == "__main__":
    main()