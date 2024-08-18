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
                alt.Tooltip('arena_score:Q', title='Arena Score'),
                alt.Tooltip('CI:N', title='95% Conf Interval'),
                alt.Tooltip('output_tok:Q', title='AVG # Output Tokens'),
                alt.Tooltip('quantization:N', title='Quantization'),
                alt.Tooltip('guided:N', title='Guided - Knowledge Embedding'),
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
        st.altair_chart(final_chart, use_container_width=True, theme="streamlit")

        st.divider()
        st.write("")

        with st.expander("View Table", expanded=False):
            st.table(df_table)

def quant_page(df):

    st.subheader("Arena Results vs. Energy Consumption")
    st.write("This section provides an overview of how the Quantization of Models affects their Arena Scores and Energy Consumption.")
    st.write("The Judge Model was GPT-4o and the Baseline Model was GPT-4-0314")

    st.write("")

    with st.container():
        
        mod, inst, guid = st.columns([2, 1, 1])

        with mod:
            model = st.selectbox("Model:", options=['LLaMA-3.1-70B-Instruct', 'LLaMA-3.1-8B-Instruct', 'Mistral-NeMo'], index=0, key='model')
        
        with inst:
            # Add a selector in Streamlit to choose between largest and smallest instance types
            instance_type = st.selectbox('Select Instance Type', ['Largest', 'Smallest'])
        
        with guid:
            # Add a selector in Streamlit to choose between guided and non-guided models
            guided = st.selectbox('Select Guidance', ['Non-guided', 'Guided'], index=0)

        # Filter the data based on the selected model
        if model == 'Mistral-NeMo':
            df_transformed = df[df.model_name == 'mistral_nemo']
            scale = [0, 0.3]
        elif model == 'LLaMA-3-8B-Instruct':
            df_transformed = df[df.model_name == 'llama3_8b']
            scale = [0, 0.2]
        elif model == 'LLaMA-3-70B-Instruct':
            df_transformed = df[df.model_name == 'llama3_70b']
            scale = [0, 2]
        elif model == 'LLaMA-3.1-8B-Instruct':
            df_transformed = df[df.model_name == 'llama3_1_8b']
            scale = [0, 0.2]
        else:
            df_transformed = df[df.model_name == 'llama3_1_70b']
            scale = [0, 2]
        
        # Apply the filter for largest or smallest instance type
        if instance_type == 'Largest':
            df_transformed = df_transformed[df_transformed['gpu_count'] == df_transformed.groupby(['quantization', 'guided'])['gpu_count'].transform('max')]
        else:
            df_transformed = df_transformed[df_transformed['gpu_count'] == df_transformed.groupby(['quantization', 'guided'])['gpu_count'].transform('min')]

        if guided == 'Guided':
            df_transformed = df_transformed[df_transformed.guided == 'True']
        else:
            df_transformed = df_transformed[df_transformed.guided == 'False']

        # Ensure the quantization order
        quant_order = ['bf16', 'fp8', 'int4']
        df_transformed['quantization'] = pd.Categorical(df_transformed['quantization'], categories=quant_order, ordered=True)

        # Display the filtered DataFrame
        st.write("")
        st.write("")

        # Define colors for each line
        performance_color = '#4c72b0'
        energy_color = '#55a868'

        # Create the Altair chart for Arena Score with Confidence Intervals
        base_scores = alt.Chart(df_transformed).mark_point(size=100, filled=True, color=performance_color).encode(
            x=alt.X('quantization:N', title='Quantization', sort=quant_order),
            y=alt.Y('arena_score:Q', title='Arena Score', scale=alt.Scale(domain=[0, 100]), axis=alt.Axis(titleColor=performance_color, orient='right')),
            tooltip=[
                alt.Tooltip('model_name:N', title='Model Name'),
                alt.Tooltip('arena_score:Q', title='Arena Score'),
                alt.Tooltip('CI:N', title='95% Conf Interval'),
                alt.Tooltip('output_tok:Q', title='AVG # Output Tokens'),
                alt.Tooltip('quantization:N', title='Quantization'),
                alt.Tooltip('guided:N', title='Guided - Knowledge Embedding'),
                alt.Tooltip('gpu_count:Q', title='GPU Count'),
                alt.Tooltip('energy_consumed:Q', title='Energy Consumed (kWh)', format='.2f'),
                alt.Tooltip('duration_minutes:Q', title='Time to complete 500 Questions (minutes)', format='.2f')
            ]
        ).properties(
            height=600
        )

        # Add the confidence intervals as error bars
        error_bars = base_scores.mark_errorbar(size=8, thickness=2, ticks=True).encode(
            y=alt.Y('95_conf_minus:Q', title='Arena Score', scale=alt.Scale(domain=[0, 100]), axis=alt.Axis(titleColor=performance_color, orient='right')),
            y2=alt.Y2('95_conf_plus:Q')
        )

        scores = base_scores + error_bars

        # Performance line on the right y-axis
        performance = alt.Chart(df_transformed).mark_line(color=performance_color, thickness=1).encode(
            x=alt.X('quantization:N', title='Quantization', sort=quant_order),
            y=alt.Y('arena_score:Q', title='Arena Score', scale=alt.Scale(domain=[0, 100]), axis=alt.Axis(titleColor=performance_color, orient='right')),
            tooltip=[
                alt.Tooltip('model_name:N', title='Model Name'),
                alt.Tooltip('arena_score:Q', title='Arena Score'),
                alt.Tooltip('CI:N', title='95% Conf Interval'),
                alt.Tooltip('output_tok:Q', title='AVG # Output Tokens'),
                alt.Tooltip('quantization:N', title='Quantization'),
                alt.Tooltip('guided:N', title='Guided - Knowledge Embedding'),
                alt.Tooltip('gpu_count:Q', title='GPU Count'),
                alt.Tooltip('energy_consumed:Q', title='Energy Consumed (kWh)', format='.2f'),
                alt.Tooltip('duration_minutes:Q', title='Time to complete 500 Questions (minutes)', format='.2f')
            ]
        ).properties(
            height=600
        )

        # Energy consumption as bar charts on the left y-axis with adjusted width
        energy_bars = alt.Chart(df_transformed).mark_bar(color=energy_color, opacity=0.6, size=20).encode(
            x=alt.X('quantization:N', title='Quantization', sort=quant_order),
            y=alt.Y('energy_consumed:Q', title='Energy Consumed (kWh)', axis=alt.Axis(titleColor=energy_color)),
            tooltip=[
                alt.Tooltip('quantization:N', title='Quantization'),
                alt.Tooltip('gpu_count:Q', title='GPU Count'),
                alt.Tooltip('energy_consumed:Q', title='Energy Consumed (kWh)', format='.2f'), 
                alt.Tooltip('duration_minutes:Q', title='Time to complete 500 Questions (minutes)', format='.2f')
            ]
        )

        # Combine the performance, energy consumption charts, and confidence intervals
        final_chart = alt.layer(energy_bars, performance, scores).resolve_scale(
            y='independent'
        ).configure_legend(
            orient='right'
        ).properties(
            height=600
        )

        # Display the chart in Streamlit
        st.altair_chart(final_chart, use_container_width=True, theme="streamlit")

        
def guided_page(df):

    st.subheader("Arena Results vs. Knowledge Embedding")
    st.write("This section provides an overview of how a Knowledge Embedding of Models affects their Arena Scores and Energy Consumption.")
    st.write("The Knowledge Embedding was created automatically using GPT-4o. It features Guidance in form of important context, useful information and a step-by-step plan to solve the task.")
    st.write("The Judge Model was GPT-4o and the Baseline Model was GPT-4-0314")

    st.write("")

    with st.container():
        
        mod, inst, quant = st.columns([2, 1, 1])

        with mod:
            model = st.selectbox("Model:", options=['LLaMA-3.1-70B-Instruct', 'LLaMA-3.1-8B-Instruct', 'LLaMA-3-70B-Instruct', 'Mistral-NeMo'], index=0, key='model2')
        
        with inst:
            # Add a selector in Streamlit to choose between largest and smallest instance types
            instance_type = st.selectbox('Select Instance Type', ['Largest', 'Smallest'], index=0 , key='instance_type2')
        
        with quant:
            # Add a selector in Streamlit to choose between guided and non-guided models
            quant = st.selectbox('Select Quantization', ['bf16', 'fp8', 'int4'], index=0, key='quant2')

        # Filter the data based on the selected model
        if model == 'Mistral-NeMo':
            df_transformed = df[df.model_name == 'mistral_nemo']
            scale = [0, 0.2]
        elif model == 'LLaMA-3-8B-Instruct':
            df_transformed = df[df.model_name == 'llama3_8b']
            scale = [0, 0.2]
        elif model == 'LLaMA-3-70B-Instruct':
            df_transformed = df[df.model_name == 'llama3_70b']
            scale = [0, 2]
        elif model == 'LLaMA-3.1-8B-Instruct':
            df_transformed = df[df.model_name == 'llama3_1_8b']
            scale = [0, 0.2]
        else:
            df_transformed = df[df.model_name == 'llama3_1_70b']
            scale = [0, 2]
        

        # Apply the filter for largest or smallest instance type
        if instance_type == 'Largest':
            df_transformed = df_transformed[df_transformed['gpu_count'] == df_transformed.groupby(['quantization', 'guided'])['gpu_count'].transform('max')]
        else:
            df_transformed = df_transformed[df_transformed['gpu_count'] == df_transformed.groupby(['quantization', 'guided'])['gpu_count'].transform('min')]

        
        df_transformed = df_transformed[df_transformed.quantization == quant]
        

        # Display the filtered DataFrame
        st.write("")
        st.write("")

        # Define colors for each line
        performance_color = '#4c72b0'
        energy_color = '#55a868'

        # Create the Altair chart for Arena Score with Confidence Intervals
        base_scores = alt.Chart(df_transformed).mark_point(size=100, filled=True, color=performance_color).encode(
            x=alt.X('guided:N', title='Guided - Knowledge Embedding'),
            y=alt.Y('arena_score:Q', title='Arena Score', scale=alt.Scale(domain=[0, 100]), axis=alt.Axis(titleColor=performance_color, orient='right')),
            tooltip=[
                alt.Tooltip('model_name:N', title='Model Name'),
                alt.Tooltip('arena_score:Q', title='Arena Score'),
                alt.Tooltip('CI:N', title='95% Conf Interval'),
                alt.Tooltip('output_tok:Q', title='AVG # Output Tokens'),
                alt.Tooltip('quantization:N', title='Quantization'),
                alt.Tooltip('guided:N', title='Guided - Knowledge Embedding'),
                alt.Tooltip('gpu_count:Q', title='GPU Count'),
                alt.Tooltip('energy_consumed:Q', title='Energy Consumed (kWh)', format='.2f'),
                alt.Tooltip('duration_minutes:Q', title='Time to complete 500 Questions (minutes)', format='.2f')
            ]
        ).properties(
            height=600
        )

        # Add the confidence intervals as error bars
        error_bars = base_scores.mark_errorbar(size=8, thickness=2, ticks=True).encode(
            y=alt.Y('95_conf_minus:Q', title='Arena Score', scale=alt.Scale(domain=[0, 100]), axis=alt.Axis(titleColor=performance_color, orient='right')),
            y2=alt.Y2('95_conf_plus:Q')
        )

        scores = base_scores + error_bars

        # Performance line on the right y-axis
        performance = alt.Chart(df_transformed).mark_line(color=performance_color, thickness=1).encode(
            x=alt.X('guided:N', title='Guided - Knowledge Embedding'),
            y=alt.Y('arena_score:Q', title='Arena Score', scale=alt.Scale(domain=[0, 100]), axis=alt.Axis(titleColor=performance_color, orient='right')),
            tooltip=[
                alt.Tooltip('model_name:N', title='Model Name'),
                alt.Tooltip('arena_score:Q', title='Arena Score'),
                alt.Tooltip('CI:N', title='95% Conf Interval'),
                alt.Tooltip('output_tok:Q', title='AVG # Output Tokens'),
                alt.Tooltip('quantization:N', title='Quantization'),
                alt.Tooltip('guided:N', title='Guided - Knowledge Embedding'),
                alt.Tooltip('gpu_count:Q', title='GPU Count'),
                alt.Tooltip('energy_consumed:Q', title='Energy Consumed (kWh)', format='.2f'),
                alt.Tooltip('duration_minutes:Q', title='Time to complete 500 Questions (minutes)', format='.2f')
            ]
        ).properties(
            height=600
        )

        # Energy consumption as bar charts on the left y-axis with adjusted width
        energy_bars = alt.Chart(df_transformed).mark_bar(color=energy_color, opacity=0.6, size=20).encode(
            x=alt.X('guided:N', title='Guided - Knowledge Embedding'),
            y=alt.Y('energy_consumed:Q', title='Energy Consumed (kWh)', scale=alt.Scale(domain=scale), axis=alt.Axis(titleColor=energy_color, orient='left')),
            tooltip=[
                alt.Tooltip('quantization:N', title='Quantization'),
                alt.Tooltip('gpu_count:Q', title='GPU Count'),
                alt.Tooltip('energy_consumed:Q', title='Energy Consumed (kWh)', format='.2f'), 
                alt.Tooltip('duration_minutes:Q', title='Time to complete 500 Questions (minutes)', format='.2f')
            ]
        )

        # Energy consumption as bar charts on the left y-axis with adjusted width
        energy_lines = alt.Chart(df_transformed).mark_line(color=energy_color, opacity=0.7, strokeDash=[4, 4], thickness=1).encode(
            x=alt.X('guided:N', title='Guided - Knowledge Embedding'),
            y=alt.Y('energy_consumed:Q', title='Energy Consumed (kWh)', scale=alt.Scale(domain=scale), axis=alt.Axis(titleColor=energy_color, orient='left')),
            tooltip=[
                alt.Tooltip('quantization:N', title='Quantization'),
                alt.Tooltip('gpu_count:Q', title='GPU Count'),
                alt.Tooltip('energy_consumed:Q', title='Energy Consumed (kWh)', format='.2f'), 
                alt.Tooltip('duration_minutes:Q', title='Time to complete 500 Questions (minutes)', format='.2f')
            ]
        )
        

        # Combine the performance, energy consumption charts, and confidence intervals
        final_chart = alt.layer(energy_bars, energy_lines, performance, scores).resolve_scale(
            y='independent'
        ).configure_legend(
            orient='right'
        ).properties(
            height=600
        )

        # Display the chart in Streamlit
        st.altair_chart(final_chart, use_container_width=True, theme="streamlit")
    



def benchmarks(): 
    st.title("LLM Emission Tests 🌍🌱")

    st.caption("This is a dashboard to visualize the results of the LLM emission tests.")

    st.divider()

    df = load_parquet_data('results')

    arena_results, quant, guided = st.tabs([
        "Arena Results",
        "Varying Quantization Levels", 
        "Knowledge Embedding"
        ])
    

    with arena_results:
        st.write("")
        st.write("")

        arena_results_page(df)
    

    with quant: 
        st.write("")
        st.write("")

        quant_page(df)

    
    with guided:
        st.write("")
        st.write("")

        guided_page(df)


if __name__ == "__main__":
    init_session_states()
    
    st.session_state.curr_page = 'benchmarks'
    sidebar()
    benchmarks()