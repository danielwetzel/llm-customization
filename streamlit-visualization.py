import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import os

st.set_page_config(page_title="LLM Emission Tests", page_icon="🌍", layout="wide")

@st.cache_data
def load_csv_data(type='emission_regression'):
    """
    Load data from a csv file
    """

    df = pd.read_csv(f'results/{type}.csv')

    return df




# Function to create charts for each test type
def create_chart(df, test_type, remove_x_title=False):
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
        y=alt.Y('actual_emissions_per_10k_prompts', title='Actual Emissions per 10,000 Prompts'),
        color = alt.Color('test_type:N', title='Test Type').sort(df['test_type'].unique()),
        tooltip=[
            alt.Tooltip('parameters', title='Parameters (billions)'),
            alt.Tooltip('actual_emissions_per_10k_prompts', title='Actual Emissions per 10,000 Prompts'),
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
        title=f'Actual Emissions for {test_type} per 10,000 Prompts',
    )

    # Create line plots for predicted emissions
    line = alt.Chart(chart_data).mark_line().encode(
        x=alt.X(x_data, title=x_title),
        y=alt.Y('pred_emissions_per_10k_prompts', title='Predicted Emissions per 10,000 Prompts'),
        color = alt.Color('test_type:N', title='Test Type').sort(df['test_type'].unique()),
        tooltip=[
            alt.Tooltip('parameters', title='Parameters (billions)'),
            alt.Tooltip('actual_emissions_per_10k_prompts', title='Actual Emissions per 10,000 Prompts'),
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
                alt.Tooltip('actual_emissions_per_10k_prompts', title='Actual Emissions per 10,000 Prompts'),
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


def main():

    st.title("LLM Emission Tests 🌍🌱")

    st.caption("This is a dashboard to visualize the results of the LLM emission tests.")

    st.subheader("", divider='grey')

    df = load_csv_data('emission_regression')

    overview, output_tok, input_tok, params = st.tabs(["Overview", "Output Token Impact", "Input Token Impact", "Model Parameter Impact"])

    with overview:

        st.subheader("Overview")
        st.write("This section provides an overview of all the impacts on the interference emissions.")


        # Create charts for each test type
        stacked_charts = []
        stacked_df = df[df.num_examples != 90]
        for test_type in stacked_df['test_type'].unique():
            stacked_charts.append(create_chart(stacked_df, test_type, remove_x_title=True))


        # Create a combined chart with overlays for all emission types
        combined_chart = alt.layer(*stacked_charts).resolve_scale(
            x='independent'
        ).properties(
            title='Emissions per Ten-Thousand Prompts by Test Type',
            height=700
        )

        st.altair_chart(combined_chart, use_container_width=True, theme="streamlit")

        st.write(df)
    

    with output_tok:

        st.subheader("Output Token Impact")
        st.write("This section visualizes the impact of the output token on the interference emissions.")

        output_chart = create_chart(df, test_type='Output-tok').properties(height=700)

        st.altair_chart(output_chart, use_container_width=True, theme="streamlit")

        st.write(df[df.test_type=='Output-tok'])


    with input_tok:

        st.subheader("Input Token Impact")
        st.write("This section visualizes the impact of the input token on the interference emissions.")

        input_chart = create_chart(df, test_type='Input-tok').properties(height=700)

        st.altair_chart(input_chart, use_container_width=True, theme="streamlit")

        st.write(df[df.test_type=='Input-tok'])

    with params:

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



if __name__ == "__main__":
    main()