import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import os
import sys
import plotly.express as px
import plotly.graph_objects as go
import re

from pages.utils.streamlit_utils import *





def arena_results_page(df):

    st.subheader("Arena Hard v0.1 Results")
    st.write("This section provides an overview of the results of the Arena Hard v0.1 Automated Benchmark.")
    st.write("The Judge Model was GPT-4o and the Baseline Model was GPT-4-0314")

    st.write("")

    with st.container(border=True):

        seperate = st.selectbox("Seperate:", options=['All', 'Guided Only', 'Quantization Only', 'Model Only'], index=0, key='seperate')


        if seperate == 'Guided Only':
            df_transformed = df[df.quantization == 'bf16']
            color = ['guided:N', 'Guided']

        elif seperate == 'Quantization Only':
            select_guidance = st.selectbox("Guidance:", options=['guided', 'non-guided'], index=1, key='guidance')
            color = ['quantization:N', 'Quantization']

            if select_guidance == 'guided':
                df_transformed = df[df.guided == 1]
            else:
                df_transformed = df[df.guided == 0]

        elif seperate == 'Model Only':
            select_guidance = st.selectbox("Guidance:", options=['guided', 'non-guided'], index=1, key='guidance2')
            df_transformed = df[df.quantization == 'bf16']
            color = ['guided:N', 'Guided']

            if select_guidance == 'guided':
                df_transformed = df_transformed[df_transformed.guided == 1]
            else:
                df_transformed = df_transformed[df_transformed.guided == 0]
        
        else:
            df_transformed = df
            color = ['guided:N', 'Guided']

        st.write("")
        st.write("")


        df_transformed['guided'] = df_transformed['guided'].map({1: 'True', 0: 'False'})

        df_transformed['model_name'] = df_transformed['model_name'].apply(lambda x: re.sub(r'-2024-08-06', '', x))

        df_table = df_transformed.groupby(['model_name', 'quantization', 'guided']).agg({
            'arena_score': 'mean', 
            'CI': 'first',
            'output_tok': 'mean'
        }).reset_index()

        # Sort the DataFrame by 'arena_score' in descending order
        df_table = df_table.sort_values(by='arena_score', ascending=False)

        # Format the 'arena_score' to show only two decimal places
        df_table['arena_score'] = df_table['arena_score'].map('{:.2f}'.format)

        # Format the 'output_tok' to show no decimal places and rename the column
        df_table['output_tok'] = df_table['output_tok'].map('{:.0f}'.format)

        df_table = df_table[['model_name', 'arena_score', 'CI', 'output_tok', 'quantization', 'guided']]

        df_table = df_table.rename(columns={'output_tok': 'avg #output_tok'})

        df_vis = df_transformed.groupby(['model_name', 'quantization', 'guided']).agg({
            'arena_score': 'mean', 
            'CI': 'first',
            'output_tok': 'mean', 
            'model_class': 'first', 
            '95_conf_plus' : 'mean',
            '95_conf_minus' : 'mean'
            }).reset_index()


        df_vis = df_vis.sort_values(by='arena_score', ascending=False)

        # Create the Altair chart
        chart = alt.Chart(df_vis).mark_point(size=100).encode(  # Increased size of circles
            x=alt.X('model_name:N', sort=None, title='Model Name'),
            y=alt.Y('arena_score:Q', title='Arena Score', scale=alt.Scale(domain=[0, 100])),
            color=alt.Color(color[0], title=color[1], scale=alt.Scale(scheme='category10')),  # Different colors for guided/non-guided
            tooltip=[
                alt.Tooltip('model_name:N', title='Model Name'),
                alt.Tooltip('quantization:N', title='Quantization'),
                alt.Tooltip('model_class:N', title='Model Class'),
                alt.Tooltip('output_tok:Q', title='Output Tokens'),
                alt.Tooltip('CI:N', title='95% Conf Interval'),
            ]
        ).properties(
            height=600
        )

        # Add the confidence intervals as error bars
        error_bars = chart.mark_errorbar(extent='ci', size=8, thickness=2, ticks=True).encode(
            y=alt.Y('95_conf_minus:Q',title='Arena Score', scale=alt.Scale(domain=[0, 100])),
            y2=alt.Y2('95_conf_plus:Q')
        )

        # Add a baseline rule at arena_score = 50
        baseline = alt.Chart(pd.DataFrame({'y': [50]})).mark_rule(color='red', strokeDash=[5,5]).encode(
            y='y:Q'
        )

        # Combine the score points, confidence intervals, and baseline
        final_chart = chart + error_bars + baseline

        # Display the chart in Streamlit
        st.altair_chart(final_chart, use_container_width=True)

        st.write("")
        st.table(df_table)


def benchmarks(): 
    st.title("LLM Emission Tests 🌍🌱")

    st.caption("This is a dashboard to visualize the results of the LLM emission tests.")

    st.divider()

    df = load_parquet_data('results')

    arena_results, param_sizes, quant, guided = st.tabs([
        "Arena Results",
        "Varying Parameter Sizes", 
        "Varying Quantization Levels", 
        "Knowledge Embedding"
        ])
    

    with arena_results:
        st.write("")
        st.write("")

        arena_results_page(df)

if __name__ == "__main__":
    init_session_states()
    
    st.session_state.curr_page = 'benchmarks'
    sidebar()
    benchmarks()