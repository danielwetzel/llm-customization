import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import os
import sys

from pages.utils.streamlit_utils import *

def overview_page(df):

    st.subheader("Overview")
    st.write("This section provides an overview of all the impacts on the interference emissions.")

    metrics = ['actual_emissions_per_10k_prompts', 'actual_cpu_energy_per_10k_prompts', 'actual_gpu_energy_per_10k_prompts', 'actual_ram_energy_per_10k_prompts']

    with st.container(border=True):

        st.session_state.metric = st.selectbox("Review Metric", metrics, format_func=label_func, index=0, key='overview_metric')

        st.divider()

        # Create charts for each test type
        stacked_charts = []
        stacked_df = df[df.num_examples != 90]
        stacked_df = stacked_df[df.test_type != 'framework_comp_vllm']
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

    st.dataframe(df)


def output_tok_page(df):



    st.subheader("Output Token Impact")
    st.write("This section visualizes the impact of the output token on the interference emissions.")

    #st.divider()

    with st.container(border=True):
        st.write("")
        st.write("")


        output_chart = create_chart(df, test_type='Output-tok').properties(height=700)

        st.altair_chart(output_chart, use_container_width=True, theme="streamlit")

    st.dataframe(df[df.test_type=='Output-tok'])


def input_tok_page(df):




    st.subheader("Input Token Impact")
    st.write("This section visualizes the impact of the input token on the interference emissions.")

    metrics = ['actual_emissions_per_10k_prompts', 'actual_cpu_energy_per_10k_prompts', 'actual_gpu_energy_per_10k_prompts', 'actual_ram_energy_per_10k_prompts']

    with st.container(border=True):

        st.session_state.metric = st.selectbox("Review Metric", metrics, format_func=label_func, index=0, key='input_tok_metric')

        st.divider()

        #input_chart = create_chart(df, test_type='Input-tok').properties(height=700)

        vis, select = st.columns([9, 1])

        with select:

            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.session_state.degree_list[0] = st.selectbox("Polynomial Degree", [1, 2], index=0)

        with vis:
        
            input_chart = create_reg_chart(df, test_type='Input-tok', metric=st.session_state.metric, degree_list=st.session_state.degree_list).properties(height=700)
            st.altair_chart(input_chart, use_container_width=True, theme="streamlit")


    st.dataframe(df[df.test_type=='Input-tok'])






def params_page(df):

    st.subheader("Model Parameter Impact")
    st.write("This section visualizes the impact of the model parameters on the interference emissions.")

    #st.divider()

    with st.container(border=True):

        st.write("")
        st.write("")

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

    st.dataframe(params_df)


def framework_page(df):

    st.subheader("Inference Framework Impact")
    st.write("This section visualizes the impact of the inference framework on a models emissions.")

    #st.divider()

    with st.container(border=True):

        st.write("")
        st.write("")

        df_framework_comp = df[df['model_type'].str.contains('bloomz')]
        #df_framework_comp = df_framework_comp[df_framework_comp['model_type'] != 'bloomz-3b']

        vis, select = st.columns([9, 1])

        with select:

            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.session_state.degree_list[0] = st.selectbox("Polynomial Degree", [1, 2, 3], index=0)

        with vis:
        
            input_chart = create_reg_chart(df_framework_comp, test_type='framework_comp_vllm', metric='actual_emissions_per_1M_out_tok', degree_list=st.session_state.degree_list).properties(height=700)
            st.altair_chart(input_chart, use_container_width=True, theme="streamlit")


    st.dataframe(df[df.test_type=='framework_comp_vllm'])


def initial_tests():

    

    st.title("LLM Emission Tests 🌍🌱")

    st.caption("This is a dashboard to visualize the results of the LLM emission tests.")

    st.subheader("", divider='grey')

    df = load_csv_data('emission_regression')

    overview, output_tok, input_tok, params, frameworks = st.tabs(["Overview", "Output Token Impact", "Input Token Impact", "Model Parameter Impact", "Inference Framework Impact"])


    with overview:
        st.write("")
        st.write("")
        
        overview_page(df)

    

    with output_tok:
        st.write("")
        st.write("")
            
        output_tok_page(df)



    with input_tok:
        st.write("")
        st.write("")

        input_tok_page(df)



    with params:
        st.write("")
        st.write("")

        params_page(df)

    
    with frameworks: 
        st.write("")
        st.write("")

        framework_page(df)

    
    



if __name__ == "__main__":
    init_session_states()

    st.session_state.curr_page = 'initial_tests'
    sidebar()
    initial_tests()