import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import os
import sys

from pages.utils.streamlit_utils import *


def comp_page(df):
    labels = [
        '1: GPU Non-Idle',
        '2: GPU Idle',
        '0: GPU',
        '3: CPU',
        '4: RAM'
    ]

    default_labels = [
        '1: GPU Non-Idle',
        '2: GPU Idle',
        '3: CPU',
        '4: RAM'
    ]

    st.subheader("Inference Serving Engine / Framework Comparison")
    st.write("This section visualizes the energy consumption of vLLM compared to HuggingFace Transformers.")
    st.write("")

    with st.container():
        label, type = st.columns([8, 2], gap='large')
        with label:
            label_selections = st.multiselect(label='Choose Energy Types', options=labels, default=default_labels, key='comp')
        with type:
            display_type = st.radio("Display Type", ('Stacked', 'Grouped', 'Line'))

        st.divider()

        # Prepare the dataframe for the stacked bar chart
        df_melted = df.melt(id_vars=[
                                    'engine', 
                                    'num_prompts', 
                                    'model_type', 
                                    'parameters', 
                                    'total_time',
                                    'time_per_prompt', 
                                    'tok_per_sec',
                                    'avg_out_tok', 
                                    'total_out_tok'], 
                            value_vars=[
                                    'actual_non_idle_gpu_energy_per_10k_prompts',
                                    'actual_idle_gpu_energy_per_10k_prompts', 
                                    'actual_gpu_energy_per_10k_prompts',
                                    'actual_cpu_energy_per_10k_prompts', 
                                    'actual_ram_energy_per_10k_prompts'],
                            var_name='Energy_Type', 
                            value_name='Energy_Consumption')

        # Add kWh unit to the energy consumption values
        df_melted['Energy_Consumption_kWh'] = df_melted['Energy_Consumption'].apply(lambda x: f"{x:.3f} kWh")

        # Define a sorting index for Energy_Type
        df_melted['Energy_Type_Order'] = df_melted['Energy_Type'].map({
            'actual_non_idle_gpu_energy_per_10k_prompts': '1: GPU Non-Idle',
            'actual_idle_gpu_energy_per_10k_prompts': '2: GPU Idle',
            'actual_gpu_energy_per_10k_prompts': '0: GPU',
            'actual_cpu_energy_per_10k_prompts': '3: CPU',
            'actual_ram_energy_per_10k_prompts': '4: RAM'
        })

        df_melted['display_colors'] = df_melted['Energy_Type_Order'].map({
            '1: GPU Non-Idle': '#16c473',
            '2: GPU Idle': '#5bf0d2',
            '0: GPU': '#005239',
            '3: CPU': '#9a86b8',
            '4: RAM': '#86b7b8'
        })

        filtered_df = df_melted[df_melted['Energy_Type_Order'].isin(label_selections)]

        if display_type == 'Stacked':
            # Stacked Bar Chart: Breakdown of Actual Energy Consumption
            chart = alt.Chart(filtered_df).mark_bar(size=50).encode(
                x=alt.X('engine', title='Inference Engine / Framework', axis=alt.Axis(labelAngle=0)),
                y=alt.Y('sum(Energy_Consumption)', title='Energy Consumed per 10k Prompts (in kWh)'),
                color=alt.Color('Energy_Type_Order', title='Energy Type', 
                                scale=alt.Scale(
                                    domain=label_selections,
                                    range=filtered_df['display_colors'].unique())
                                ),
                order=alt.Order('Energy_Type_Order', sort='descending'),
                tooltip=['num_prompts', 'Energy_Type_Order', 'Energy_Consumption_kWh']
            ).properties(
                width=600,
                height=600,
                title='Breakdown of Actual Energy Consumption per 10k Prompts'
            )
        
        elif display_type == 'Line':
            # Line Chart: Showing decrease from transformers to vllm
            line_chart = alt.Chart(filtered_df).mark_line(point=True).encode(
                x=alt.X('engine:N', title='Inference Engine / Framework', axis=alt.Axis(labelAngle=0)),
                y=alt.Y('Energy_Consumption', title='Energy Consumed per 10k Prompts (in kWh)'),
                color=alt.Color('Energy_Type_Order', title='Energy Type', 
                                scale=alt.Scale(
                                    domain=label_selections,
                                    range=filtered_df['display_colors'].unique())
                                ),
                detail='Energy_Type_Order',
                tooltip=['num_prompts', 'Energy_Type_Order', 'Energy_Consumption_kWh']
            ).properties(
                width=600,
                height=600,
                title='Breakdown of Actual Energy Consumption per 10k Prompts'
            )

            # Add text annotations for the points
            text_chart = alt.Chart(filtered_df).mark_text(align='left',fontSize=12, dx=5, dy=-5).encode(
                x=alt.X('engine:N'),
                y=alt.Y('Energy_Consumption'),
                text=alt.Text('Energy_Consumption:Q', format='.3f'),
                color=alt.Color('Energy_Type_Order', scale=alt.Scale(
                    domain=label_selections,
                    range=filtered_df['display_colors'].unique())),
                tooltip=['num_prompts', 'Energy_Type_Order', 'Energy_Consumption_kWh']
            )

            # Calculate decrease factors and create decrease annotations
            decreases = []
            for energy_type in filtered_df['Energy_Type_Order'].unique():
                temp_df = filtered_df[filtered_df['Energy_Type_Order'] == energy_type]
                transformers_value = temp_df[temp_df['engine'] == 'transformers']['Energy_Consumption'].values[0]
                vllm_value = temp_df[temp_df['engine'] == 'vllm']['Energy_Consumption'].values[0]
                decrease_factor = transformers_value / vllm_value if vllm_value != 0 else 0
                decreases.append({
                    'Energy_Type_Order': energy_type,
                    'x': 'u',
                    'y': (transformers_value + vllm_value) / 2,
                    'decrease_factor': f"{decrease_factor:.2f}x"
                })

            decreases_df = pd.DataFrame(decreases)

            decrease_chart = alt.Chart(decreases_df).mark_text(align='center', fontSize=14, fontWeight='bold').encode(
                x=alt.X('x'),
                y=alt.Y('y:Q'),
                text=alt.Text('decrease_factor:N'),
                color=alt.Color('Energy_Type_Order', scale=alt.Scale(
                    domain=label_selections,
                    range=filtered_df['display_colors'].unique())),
                tooltip=['Energy_Type_Order', 'decrease_factor']
            )

            # Add white background for text
            rect_chart = alt.Chart(decreases_df).mark_rect(
                width=50, height=20, color='white', opacity=0.7
            ).encode(
                x=alt.X('x'),
                y=alt.Y('y:Q')
            )

            chart = line_chart + text_chart + rect_chart + decrease_chart
        
        else: 
            # Grouped Bar Chart: Breakdown of Actual Energy Consumption
            chart = alt.Chart(filtered_df).mark_bar(size=50).encode(
                x=alt.X('engine:N', title='Inference Engine / Framework', axis=alt.Axis(labelAngle=0)),
                y=alt.Y('Energy_Consumption', title='Energy Consumed per 10k Prompts (in kWh)'),
                color=alt.Color('Energy_Type_Order', title='Energy Type', 
                                scale=alt.Scale(
                                    domain=label_selections,
                                    range=filtered_df['display_colors'].unique())
                                ),
                xOffset='Energy_Type_Order',
                tooltip=['num_prompts', 'Energy_Type_Order', 'Energy_Consumption_kWh']
            ).properties(
                width=600,
                height=600,
                title='Breakdown of Actual Energy Consumption per 10k Prompts'
            )     

        st.altair_chart(chart, use_container_width=True)


    

    
    #st.write(df)

    #st.write(filtered_df)



def breackdown_page(df): 

    lables = [
        '1: GPU Non-Idle',
        '2: GPU Idle',
        '0: GPU',
        '3: CPU',
        '4: RAM'
    ]

    default_labels = [
        '1: GPU Non-Idle',
        '2: GPU Idle',
        '3: CPU',
        '4: RAM'
    ]

    st.subheader("Energy Type Breakdown")
    st.write("This section visualizes the energy consumption of different energy types for increasing Output Tokens per Prompt.")

    st.write("")

    with st.container(border=True):

        label_selections = st.multiselect(label='Choose Energy Types', options=lables, default=default_labels, key='breakdown')

        st.divider()
        

        # Prepare the dataframe for the stacked bar chart
        df_melted = df.melt(id_vars=[
                                    'test_type', 
                                    'num_prompts', 
                                    'model_type', 
                                    'parameters', 
                                    'num_examples', 
                                    'total_time',
                                    'time_per_prompt', 
                                    'tok_per_sec',
                                    'avg_out_tok', 
                                    'total_out_tok'], 
                            value_vars=[
                                    'actual_non_idle_gpu_energy_per_10k_prompts',
                                    'actual_idle_gpu_energy_per_10k_prompts', 
                                    'actual_gpu_energy_per_10k_prompts',
                                    'actual_cpu_energy_per_10k_prompts', 
                                    'actual_ram_energy_per_10k_prompts'],
                            var_name='Energy_Type', 
                            value_name='Energy_Consumption')


        # Define a sorting index for Energy_Type
        df_melted['Energy_Type_Order'] = df_melted['Energy_Type'].map({
            'actual_non_idle_gpu_energy_per_10k_prompts': '1: GPU Non-Idle',
            'actual_idle_gpu_energy_per_10k_prompts': '2: GPU Idle',
            'actual_gpu_energy_per_10k_prompts': '0: GPU',
            'actual_cpu_energy_per_10k_prompts': '3: CPU',
            'actual_ram_energy_per_10k_prompts': '4: RAM'
        })

        df_melted['display_colors'] = df_melted['Energy_Type_Order'].map({
            '1: GPU Non-Idle': '#16c473',
            '2: GPU Idle': '#5bf0d2',
            '0: GPU': '#005239',
            '3: CPU': '#9a86b8',
            '4: RAM': '#86b7b8'
        })

        filtered_df = df_melted[df_melted['Energy_Type_Order'].isin(label_selections)]

        # Stacked Bar Chart: Breakdown of Actual Energy Consumption
        stacked_bar_chart = alt.Chart(filtered_df).mark_bar(size=10).encode(
            x=alt.X('avg_out_tok', title='Average Output Tokens per Prompt'),
            y=alt.Y('sum(Energy_Consumption)', title='Energy Consumed per 10k Prompts'),
            color=alt.Color('Energy_Type_Order', title='Energy Type', 
                            scale=alt.Scale(
                                domain=label_selections,
                                range=filtered_df['display_colors'].unique())
                            ),
            order=alt.Order('Energy_Type_Order', sort='descending'),
            tooltip=['num_prompts', 'test_type', 'Energy_Type', 'Energy_Consumption']
        ).properties(
            width=600,
            height=600,
            title='Breakdown of Actual Energy Consumption per 10k Prompts'
        )

        # Stacked Area Chart: Breakdown of Actual Energy Consumption
        stacked_area_chart = alt.Chart(filtered_df).mark_area(opacity=0.2).encode(
            x=alt.X('avg_out_tok', title='Average Output Tokens per Prompt'),
            y=alt.Y('sum(Energy_Consumption)', title='Energy Consumed per 10k Prompts'),
            color=alt.Color('Energy_Type_Order', title='Energy Type', 
                            scale=alt.Scale(
                                domain=label_selections,
                                range=filtered_df['display_colors'].unique())
                            ),
            order=alt.Order('Energy_Type_Order', sort='descending'),
            tooltip=['num_prompts', 'test_type', 'Energy_Type', 'Energy_Consumption']
        ).properties(
            width=600,
            height=600,
            title='Breakdown of Actual Energy Consumption per 10k Prompts'
        )

        st.altair_chart(stacked_bar_chart+stacked_area_chart, use_container_width=True)

    st.divider()

    st.subheader("Underlying Data")

    st.dataframe(df, hide_index=True)


def regression_page(df): 

    st.subheader("Energy Type Regressions")
    st.write("This section visualizes regressions for the different Energy Types.")

    st.write("")

    with st.container(border=True):

        options = ['actual_non_idle_gpu_energy_per_10k_prompts',
                    'actual_idle_gpu_energy_per_10k_prompts', 
                    'actual_gpu_energy_per_10k_prompts',
                    'actual_cpu_energy_per_10k_prompts', 
                    'actual_ram_energy_per_10k_prompts']

        label_selections = st.selectbox(label='Choose Energy Type', options=options, index=0, format_func=label_func, key='regression')

        st.divider()

        #input_chart = create_chart(df, test_type='Input-tok').properties(height=700)

        vis, select = st.columns([9, 1])

        with select:

            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.session_state.degree_list[0] = st.selectbox("Polynomial Degree", [1, 2], index=1)

        with vis:
        
            input_chart = create_reg_chart(df, test_type='Output-tok-vllm', metric=label_selections, degree_list=st.session_state.degree_list).properties(height=700)
            st.altair_chart(input_chart, use_container_width=True, theme="streamlit")
    

    st.divider()

    st.subheader("Underlying Data")

    st.dataframe(df, hide_index=True)


def model_param_comp_page(df):
    labels = [
        '1: GPU Non-Idle',
        '2: GPU Idle',
        '0: GPU',
        '3: CPU',
        '4: RAM', 
        '5: Total'
    ]

    default_labels = [
        '1: GPU Non-Idle',
        '2: GPU Idle',
        '3: CPU',
        '4: RAM'
    ]

    st.subheader("Model Parameter Comparison")
    st.write("This section visualizes the model parameters and their energy consumption.")
    st.write("")

    with st.container(border=True):
        
        sort, display = st.columns([5, 5])
        with sort:
            sort_option = st.selectbox("Sort by", options=['Number of GPUs', 'Model Parameters'], key='sort_option_param_comp')

            energy_type_selections = st.multiselect("Select Energy Types", options=labels, default=default_labels, key='energy_type_param_comp')

        with display: 
            display = st.selectbox("Display Settings", options=["Show all", "Largest Instance Only", "Smallest Instance Possible", "Lowest Energy Consumption"], key='display_param_comp')

            param_size_selection = st.multiselect("Select Model Sizes", options=['7B', '13B', '34B', '70B'], default=['7B', '13B', '34B', '70B'], key='param_size_sel')

            
        st.divider()

        # Map labels back to actual column names
        energy_type_mapping = {
            '3: CPU': 'cpu_energy_per_1M_prompts',
            '4: RAM': 'ram_energy_per_1M_prompts',
            '2: GPU Idle': 'idle_gpu_energy_per_1M_prompts',
            '1: GPU Non-Idle': 'non_idle_gpu_energy_per_1M_prompts',
            '0: GPU': 'gpu_energy_per_1M_prompts',
            '5: Total': 'total_energy_per_1M_prompts',
        }

        # Map selections to the corresponding column names
        selected_columns = [energy_type_mapping[label] for label in energy_type_selections]


        # Filter the dataframe based on selected model sizes
        df = df[df['model_setup'].str.split('_').str[0].isin(param_size_selection)]

        # Prepare the dataframe for the stacked bar chart
        df_melted = df.melt(id_vars=[
                                    'model_setup', 
                                    'parameters', 
                                    'num_gpus', 
                                    'num_prompts', 
                                    'total_time',
                                    'time_per_prompt', 
                                    'tok_per_sec',
                                    'total_out_tok', 
                                    'total_in_tok', 
                                    'avg_out_tok', 
                                    'avg_in_tok'], 
                            value_vars=selected_columns,
                            var_name='Energy_Type', 
                            value_name='Energy_Consumption')

        # Add kWh unit to the energy consumption values
        df_melted['Energy_Consumption_kWh'] = df_melted['Energy_Consumption'].apply(lambda x: f"{x:.3f} kWh")

        # Reverse map the energy type columns back to labels for display
        reverse_energy_type_mapping = {v: k for k, v in energy_type_mapping.items()}
        df_melted['Energy_Type_Order'] = df_melted['Energy_Type'].map(reverse_energy_type_mapping)

        # Color mapping
        color_mapping = {
            'total_energy_per_1M_prompts': '#ff6600',
            'cpu_energy_per_1M_prompts': '#9a86b8',
            'gpu_energy_per_1M_prompts': '#005239',
            'ram_energy_per_1M_prompts': '#86b7b8',
            'idle_gpu_energy_per_1M_prompts': '#5bf0d2',
            'non_idle_gpu_energy_per_1M_prompts': '#16c473'
        }

        df_melted['display_colors'] = df_melted['Energy_Type'].map(color_mapping)

        # Define custom sort orders
        sort_order_gpus = [
            '7B_1GPUs', 
            '7B_4GPUs', '13B_4GPUs', '34B_4GPUs', 
            '7B_8GPUs', '13B_8GPUs', '34B_8GPUs', '70B_8GPUs'
        ]

        sort_order_params = [
            '7B_1GPUs', '7B_4GPUs', '7B_8GPUs',
            '13B_4GPUs', '13B_8GPUs',
            '34B_4GPUs', '34B_8GPUs',
            '70B_8GPUs'
        ]

        smallest_display_options = [
            '7B_1GPUs', '13B_4GPUs', '34B_4GPUs', '70B_8GPUs'
        ]

        largest_display_options = [
            '70B_8GPUs', '34B_8GPUs', '13B_8GPUs', '7B_8GPUs'
        ]

        lowest_energy_display_options = [
            '7B_4GPUs', '13B_4GPUs', '34B_4GPUs', '70B_8GPUs'
        ]

        if sort_option == 'Number of GPUs':
            sort_order = sort_order_gpus
        else:
            sort_order = sort_order_params

        df_melted['model_setup'] = pd.Categorical(df_melted['model_setup'], categories=sort_order, ordered=True)

        # Filter by selected energy types
        filtered_df = df_melted[df_melted['Energy_Type'].isin(selected_columns)]

        if display == 'Largest Instance Only':
            filtered_df = filtered_df[filtered_df['model_setup'].isin(largest_display_options)]
        elif display == 'Smallest Instance Possible':
            filtered_df = filtered_df[filtered_df['model_setup'].isin(smallest_display_options)]
        elif display == 'Lowest Energy Consumption':
            filtered_df = filtered_df[filtered_df['model_setup'].isin(lowest_energy_display_options)]

        chart = alt.Chart(filtered_df).mark_bar(size=40).encode(
            x=alt.X('model_setup:O', title='Model Setup', sort=sort_order),
            y=alt.Y('Energy_Consumption:Q', title='Energy Consumed per 1M Prompts (kWh)'),
            color=alt.Color('Energy_Type_Order', title='Energy Type', 
                             scale=alt.Scale(
                                domain=energy_type_selections,
                                range=[color_mapping[energy_type_mapping[label]] for label in energy_type_selections])
                             ),
            order=alt.Order('Energy_Type_Order', sort='descending'),
            tooltip=['num_prompts', 'Energy_Type_Order', 'Energy_Consumption_kWh']
        ).properties(
            width=600,
            height=600,
            title=f'Energy Consumption by Model Setup'
        )

        st.altair_chart(chart, use_container_width=True)

    st.divider()

    st.subheader("Underlying Data")

    st.dataframe(df, hide_index=True)


def vllm_tests(): 
    st.title("LLM Emission Tests 🌍🌱")

    st.caption("This is a dashboard to visualize the results of the LLM emission tests.")

    st.divider()

    #st.subheader("", divider='grey')

    df = load_csv_data('emission_regression_vllm')
    comp_df = load_csv_data('transformers_vs_vllm')
    param_comp_df = load_csv_data('params_test')

    comp, breackdown, regression, model_param_comp = st.tabs([
        "Transformers vs. vLLM",
        "Energy Type Breakdown", 
        "Energy Regression", 
        "Model Parameter Comparison"
        ])
    
    with comp:
        st.write("")
        st.write("")

        comp_page(comp_df)

    with breackdown:

        st.write("")
        st.write("")

        breackdown_page(df)

    with regression:

        st.write("")
        st.write("")

        regression_page(df)
    
    with model_param_comp:
        st.write("")
        st.write("")
        model_param_comp_page(param_comp_df)

if __name__ == "__main__":
    init_session_states()
    
    st.session_state.curr_page = 'vllm_tests'
    sidebar()
    vllm_tests()