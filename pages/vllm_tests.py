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
        '4: RAM', 
        '5: Total'
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

        model_selection = st.selectbox("Select Model", options=['LLaMA-3', 'LLaMA-3.1'], index=0, key='model_selection')

        label, type = st.columns([8, 2], gap='large')
        with label:
            label_selections = st.multiselect(label='Choose Energy Types', options=labels, default=default_labels, key='comp')
        with type:
            display_type = st.radio("Display Type", ('Line', 'Stacked', 'Grouped'))

        st.divider()

        if model_selection == 'LLaMA-3.1':
            used_df = load_csv_data('transformers_vs_vllm_llama3_1')
        
        else: 
            used_df = df
        

        # Transform back to 7500 prompts and Wh instead of kWh
        used_df['actual_non_idle_gpu_energy'] = used_df['actual_non_idle_gpu_energy_per_10k_prompts'] / 10000 * 7500 * 1000 
        used_df['actual_idle_gpu_energy'] = used_df['actual_idle_gpu_energy_per_10k_prompts'] / 10000 * 7500 * 1000
        used_df['actual_gpu_energy'] = used_df['actual_gpu_energy_per_10k_prompts'] / 10000 * 7500 * 1000
        used_df['actual_cpu_energy'] = used_df['actual_cpu_energy_per_10k_prompts'] / 10000 * 7500 * 1000
        used_df['actual_ram_energy'] = used_df['actual_ram_energy_per_10k_prompts'] / 10000 * 7500 * 1000
        used_df['actual_total_energy'] = used_df['actual_gpu_energy'] + used_df['actual_cpu_energy'] + used_df['actual_ram_energy']
        

        # Prepare the dataframe for the stacked bar chart
        df_melted = used_df.melt(id_vars=[
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
                                    'actual_non_idle_gpu_energy',
                                    'actual_idle_gpu_energy', 
                                    'actual_gpu_energy',
                                    'actual_cpu_energy', 
                                    'actual_ram_energy',
                                    'actual_total_energy'],
                            var_name='Energy_Type', 
                            value_name='Energy_Consumption')

        # Add kWh unit to the energy consumption values
        df_melted['Energy_Consumption_Wh'] = df_melted['Energy_Consumption'].apply(lambda x: f"{x:.3f} Wh")

        # Define a sorting index for Energy_Type
        df_melted['Energy_Type_Order'] = df_melted['Energy_Type'].map({
            'actual_non_idle_gpu_energy': '1: GPU Non-Idle',
            'actual_idle_gpu_energy': '2: GPU Idle',
            'actual_gpu_energy': '0: GPU',
            'actual_cpu_energy': '3: CPU',
            'actual_ram_energy': '4: RAM', 
            'actual_total_energy': '5: Total'
        })

        df_melted['display_colors'] = df_melted['Energy_Type_Order'].map({
            '1: GPU Non-Idle': '#16c473',
            '2: GPU Idle': '#5bf0d2',
            '0: GPU': '#005239',
            '3: CPU': '#9a86b8',
            '4: RAM': '#86b7b8', 
            '5: Total': '#ff6600'
        })


        filtered_df = df_melted[df_melted['Energy_Type_Order'].isin(label_selections)]

        if display_type == 'Stacked':
            # Stacked Bar Chart: Breakdown of Actual Energy Consumption
            chart = alt.Chart(filtered_df).mark_bar(size=50).encode(
                x=alt.X('engine', title='Inference Engine / Framework', axis=alt.Axis(labelAngle=0)),
                y=alt.Y('sum(Energy_Consumption)', title='Energy Consumed (in Wh)'),
                color=alt.Color('Energy_Type_Order', title='Energy Type', 
                                scale=alt.Scale(
                                    domain=label_selections,
                                    range=filtered_df['display_colors'].unique())
                                ),
                order=alt.Order('Energy_Type_Order', sort='descending'),
                tooltip=['num_prompts', 'Energy_Type_Order', 'Energy_Consumption_Wh']
            ).properties(
                width=600,
                height=600,
                title='Breakdown of Actual Energy Consumption for 7.500 Prompts'
            )
        
        elif display_type == 'Line':
            # Line Chart: Showing decrease from transformers to vllm
            line_chart = alt.Chart(filtered_df).mark_line(point=True).encode(
                x=alt.X('engine:N', title='Inference Engine / Framework', axis=alt.Axis(labelAngle=0)),
                y=alt.Y('Energy_Consumption', title='Energy Consumed (in Wh)'),
                color=alt.Color('Energy_Type_Order', title='Energy Type', 
                                scale=alt.Scale(
                                    domain=label_selections,
                                    range=filtered_df['display_colors'].unique())
                                ),
                detail='Energy_Type_Order',
                tooltip=['num_prompts', 'Energy_Type_Order', 'Energy_Consumption_Wh']
            ).properties(
                width=600,
                height=600,
                title='Breakdown of Actual Energy Consumption for 7.500 Prompts'
            )

            # Add text annotations for the points
            text_chart = alt.Chart(filtered_df).mark_text(align='left',fontSize=12, dx=5, dy=-5).encode(
                x=alt.X('engine:N'),
                y=alt.Y('Energy_Consumption'),
                text=alt.Text('Energy_Consumption:Q', format='.3f'),
                color=alt.Color('Energy_Type_Order', scale=alt.Scale(
                    domain=label_selections,
                    range=filtered_df['display_colors'].unique())),
                tooltip=['num_prompts', 'Energy_Type_Order', 'Energy_Consumption_Wh']
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
                y=alt.Y('Energy_Consumption', title='Energy Consumed (in Wh)'),
                color=alt.Color('Energy_Type_Order', title='Energy Type', 
                                scale=alt.Scale(
                                    domain=label_selections,
                                    range=filtered_df['display_colors'].unique())
                                ),
                xOffset='Energy_Type_Order',
                tooltip=['num_prompts', 'Energy_Type_Order', 'Energy_Consumption_Wh']
            ).properties(
                width=600,
                height=600,
                title='Breakdown of Actual Energy Consumption for 7.500 Prompts'
            )     

        st.altair_chart(chart, use_container_width=True, theme="streamlit")


    

    
    #st.write(df)

    #st.write(filtered_df)



def breakdown_page(df): 

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

    st.subheader("Energy Type Breakdown")
    st.write("This section visualizes the energy consumption of different energy types for increasing Output Tokens per Prompt.")

    st.write("")

    with st.container(border=True):

        label_selections = st.multiselect(label='Choose Energy Types', options=labels, default=default_labels, key='breakdown')

        st.divider()
        
        # Transform back to 7500 prompts and Wh instead of kWh
        df['actual_non_idle_gpu_energy'] = df['actual_non_idle_gpu_energy_per_10k_prompts'] / 10000 * 7500 * 1000 
        df['actual_idle_gpu_energy'] = df['actual_idle_gpu_energy_per_10k_prompts'] / 10000 * 7500 * 1000
        df['actual_gpu_energy'] = df['actual_gpu_energy_per_10k_prompts'] / 10000 * 7500 * 1000
        df['actual_cpu_energy'] = df['actual_cpu_energy_per_10k_prompts'] / 10000 * 7500 * 1000
        df['actual_ram_energy'] = df['actual_ram_energy_per_10k_prompts'] / 10000 * 7500 * 1000
        df['actual_total_energy'] = df['actual_gpu_energy'] + df['actual_cpu_energy'] + df['actual_ram_energy']

        df_cleaned = df[['test_type', 
                        'num_prompts', 
                        'model_type', 
                        'parameters', 
                        'num_examples', 
                        'total_time',
                        'time_per_prompt', 
                        'tok_per_sec',
                        'avg_out_tok', 
                        'total_out_tok', 
                        'actual_non_idle_gpu_energy',
                        'actual_idle_gpu_energy', 
                        'actual_gpu_energy',
                        'actual_cpu_energy', 
                        'actual_ram_energy',
                        'actual_total_energy']]



        # Prepare the dataframe for the stacked bar chart
        df_melted = df_cleaned.melt(id_vars=[
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
                                    'actual_non_idle_gpu_energy',
                                    'actual_idle_gpu_energy', 
                                    'actual_gpu_energy',
                                    'actual_cpu_energy', 
                                    'actual_ram_energy',
                                    'actual_total_energy'],
                            var_name='Energy_Type', 
                            value_name='Energy_Consumption')


        # Define a sorting index for Energy_Type
        df_melted['Energy_Type_Order'] = df_melted['Energy_Type'].map({
            'actual_non_idle_gpu_energy': '1: GPU Non-Idle',
            'actual_idle_gpu_energy': '2: GPU Idle',
            'actual_gpu_energy': '0: GPU',
            'actual_cpu_energy': '3: CPU',
            'actual_ram_energy': '4: RAM', 
            'actual_total_energy': '5: Total'
        })

        df_melted['display_colors'] = df_melted['Energy_Type_Order'].map({
            '1: GPU Non-Idle': '#16c473',
            '2: GPU Idle': '#5bf0d2',
            '0: GPU': '#005239',
            '3: CPU': '#9a86b8',
            '4: RAM': '#86b7b8', 
            '5: Total': '#ff6600'
        })

        filtered_df = df_melted[df_melted['Energy_Type_Order'].isin(label_selections)]

        # Stacked Bar Chart: Breakdown of Actual Energy Consumption
        stacked_bar_chart = alt.Chart(filtered_df).mark_bar(size=10).encode(
            x=alt.X('avg_out_tok', title='Average Output Tokens per Prompt'),
            y=alt.Y('sum(Energy_Consumption)', title='Energy Consumed (Wh)'),
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
            title='Breakdown of Actual Energy Consumption for 7.500 Prompts'
        )

        # Stacked Area Chart: Breakdown of Actual Energy Consumption
        stacked_area_chart = alt.Chart(filtered_df).mark_area(opacity=0.2).encode(
            x=alt.X('avg_out_tok', title='Average Output Tokens per Prompt'),
            y=alt.Y('sum(Energy_Consumption)', title='Energy Consumed (Wh)'),
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
            title='Breakdown of Actual Energy Consumption for 7.500 Prompts'
        )

        st.altair_chart(stacked_bar_chart+stacked_area_chart, use_container_width=True, theme="streamlit")

    st.divider()

    with st.expander("Underlying Data", expanded=False):
        st.dataframe(df_cleaned, hide_index=True)


def regression_page(output_comp_df, input_comp_df, param_comp_df): 

    st.subheader("Energy Regressions")
    st.write("This section visualizes regressions for the different Energy Types.")

    st.write("")

    with st.container(border=True):

        options = ['Output Tokens', 
                    'Input Tokens',
                    'Model Size',]

        label_selections = st.selectbox(label='Choose Test for Regression', options=options, index=0, key='regression')

        st.divider()

        st.write("")
        st.write("")


        with st.container(border=True):

            # Define the X values for the regressions
            input_feature = input_comp_df[['avg_in_tok']]
            output_feature = output_comp_df[['avg_out_tok']]
            param_feature = param_comp_df[['parameters']]

            # Define the Y values for the regressions
            input_target = input_comp_df[['total_energy_100k_output_tokens_Wh']]
            output_target = output_comp_df[['total_energy_10k_prompts_Wh']]
            param_target = param_comp_df[['total_energy_7500_prompts_Wh']]

            # Define the regression models
            input_model = fit_regression(X=input_feature, y=input_target, regression_type='linear')
            output_model = fit_regression(X=output_feature, y=output_target, regression_type='polynomial', polynomial_degree=2)
            param_model = fit_regression(X=param_feature, y=param_target, regression_type='exponential')

            input_predict_values = {'avg_in_tok': [10, 50, 250, 500, 1000, 1500, 2000, 3000, 4000, 5000, 10000, 30000]}
            output_predict_values = {'avg_out_tok': [10, 50, 250, 500, 1000, 1500, 2000]}
            param_predict_values = {'parameters': [7, 10, 25, 45, 50, 75]}

            # Predict values for altair graphical regression
            input_pred_df = predict_values(model=input_model, 
                                            pred_value_dict=input_predict_values, 
                                            type='linear', 
                                            target='total_energy_100k_output_tokens_Wh')

            output_pred_df = predict_values(model=output_model, 
                                            pred_value_dict=output_predict_values, 
                                            type='polynomial', 
                                            target='total_energy_10k_prompts_Wh')
            
            param_pred_df = predict_values(model=param_model,
                                            pred_value_dict=param_predict_values,
                                            type='exponential',
                                            target='total_energy_7500_prompts_Wh')


            # Combine the actual and predicted values
            input_df = combine_actual_predict_df(actual_df=input_comp_df, 
                                                    predict_df=input_pred_df,
                                                        feature='avg_in_tok', 
                                                        target='total_energy_100k_output_tokens_Wh')

            output_df = combine_actual_predict_df(actual_df=output_comp_df, 
                                                    predict_df=output_pred_df,
                                                        feature='avg_out_tok', 
                                                        target='total_energy_10k_prompts_Wh')
            
            param_df = combine_actual_predict_df(actual_df=param_comp_df,
                                                        predict_df=param_pred_df,
                                                        feature='parameters',
                                                        target='total_energy_7500_prompts_Wh')
            

            if label_selections == 'Input Tokens':

                st.subheader("Regression for the Input Token Count")


                input_tok_end = st.slider("Input Token Count to Predict",  min_value=1, max_value=128000, value=1024, step=64)
                
        
                
                end_pred_df = predict_values(model=input_model, 
                                        pred_value_dict={'avg_in_tok': [input_tok_end]}, 
                                        type='linear', 
                                        target='total_energy_100k_output_tokens_Wh')
                
                end_pred_df = pd.concat([end_pred_df, pd.DataFrame({'Type': ['Prediction']})], axis=1, copy=False)

                input_df = pd.concat([input_df, end_pred_df], axis=0, ignore_index=True, copy=False)


                # Create Altair chart
                input_base = alt.Chart(input_df[input_df['Type'] == 'Actual']).mark_point(size=20, filled=True).encode(
                    x=alt.X('avg_in_tok', title='Average Input Tokens per Prompt'),
                    y=alt.Y('total_energy_100k_output_tokens_Wh', title='Energy Consumed (in Wh)'),
                    tooltip=['avg_in_tok', 'Type']
                ).properties(
                    width=1200,
                    height=600
                )


                # Highlight predicted values
                input_predicted = alt.Chart(input_df[input_df['Type'].isin(['Predicted', 'Prediction'])]).mark_point(size=1, filled=False).encode(
                    x=alt.X('avg_in_tok', title='Average Input Tokens per Prompt'), 
                    y=alt.Y('total_energy_100k_output_tokens_Wh', title='Energy Consumed (in Wh)'),
                    tooltip=['avg_in_tok', 'Type']
                )

                # Highlight predicted values
                input_highlight = alt.Chart(input_df[input_df['Type'].isin(['Prediction'])]).mark_point(size=300, filled=False, strokeWidth=4).encode(
                    x=alt.X('avg_in_tok', title='Average Input Tokens per Prompt'), 
                    y=alt.Y('total_energy_100k_output_tokens_Wh', title='Energy Consumed (in Wh)'),
                    color=alt.Color('Type', title='Input Token Points', 
                                    scale=alt.Scale(
                                    domain=["Prediction"],
                                    range=["#DD7F70", "#A43489"])
                                ),
                    tooltip=['avg_in_tok', 'Type']
                )

                input_regression = input_predicted.transform_regression('avg_in_tok', 'total_energy_100k_output_tokens_Wh', method="linear").mark_line()

                # Combine charts
                input_chart = input_highlight + input_base + input_predicted + input_regression

                st.altair_chart(input_chart, use_container_width=True, theme="streamlit")


                
            elif label_selections == 'Output Tokens':

            

                st.subheader("Regression for the Output Token Count")

                output_tok_end = st.slider("Output Token Count to Predict",  min_value=1, max_value=32000, value=256, step=64)
                
        
                
                end_pred_df = predict_values(model=output_model, 
                                        pred_value_dict={'avg_out_tok': [output_tok_end]}, 
                                        type='polynomial', 
                                        target='total_energy_10k_prompts_Wh')
                
                end_pred_df = pd.concat([end_pred_df, pd.DataFrame({'Type': ['Prediction']})], axis=1, copy=False)

                output_df = pd.concat([output_df, end_pred_df], axis=0, ignore_index=True, copy=False)

                # Create Altair chart
                output_base = alt.Chart(output_df[output_df['Type'] == 'Actual']).mark_point(size=20, filled=True).encode(
                    x=alt.X('avg_out_tok', title='Average Output Tokens per Prompt'),
                    y=alt.Y('total_energy_10k_prompts_Wh', title='Energy Consumed (in Wh)'),
                    tooltip=['avg_out_tok', 'Type']
                ).properties(
                    width=1200,
                    height=600
                )


                # Highlight predicted values
                output_predicted = alt.Chart(output_df[output_df['Type'].isin(['Predicted', 'Prediction'])]).mark_point(size=1, filled=False).encode(
                    x=alt.X('avg_out_tok', title='Average Output Tokens per Prompt'), 
                    y=alt.Y('total_energy_10k_prompts_Wh', title='Energy Consumed (in Wh)'),
                    tooltip=['avg_out_tok', 'Type']
                )

                # Highlight predicted values
                output_highlight = alt.Chart(output_df[output_df['Type'].isin(['Prediction'])]).mark_point(size=300, filled=False, strokeWidth=4).encode(
                    x=alt.X('avg_out_tok', title='Average Output Tokens per Prompt'), 
                    y=alt.Y('total_energy_10k_prompts_Wh', title='Energy Consumed (in Wh)'),
                    color=alt.Color('Type', title='Input Token Points', 
                                    scale=alt.Scale(
                                    domain=['Prediction'],
                                    range=["#DD7F70", "#A43489"])
                                ),
                    tooltip=['avg_out_tok', 'Type']
                )

                output_regression = output_predicted.transform_regression('avg_out_tok', 'total_energy_10k_prompts_Wh', method="quad").mark_line()

                # Combine charts
                output_chart = output_highlight + output_base + output_predicted + output_regression

                st.altair_chart(output_chart, use_container_width=True, theme="streamlit")

        

            else:

                st.subheader("Regression for Changes in the Model Size")

                st.info("This Regression is calculated based on changing hardware requirements to run the different model sizes.\n The larger the model, the more GPUs are needed to run the model (which contributes drastically to the energy consumption).")


                param_end = st.slider("Model Size after Change (in Billion)",  min_value=0.1, max_value=150.0, value=70.0, step=2.0)

                
                end_pred_df = predict_values(model=param_model, 
                                        pred_value_dict={'parameters': [param_end]}, 
                                        type='exponential', 
                                        target='total_energy_7500_prompts_Wh')
                
                end_pred_df = pd.concat([end_pred_df, pd.DataFrame({'Type': ['Prediction']})], axis=1, copy=False)

                param_df = pd.concat([param_df, end_pred_df], axis=0, ignore_index=True, copy=False)

                

                # Create Altair chart
                param_base = alt.Chart(param_df[param_df['Type'] == 'Actual']).mark_point(size=20, filled=True).encode(
                    x=alt.X('parameters', title='Model Size (in Billion)'),
                    y=alt.Y('total_energy_7500_prompts_Wh', title='Energy Consumed (in Wh) for 7.500 Prompts'),
                    tooltip=['parameters', 'Type']
                ).properties(
                    width=1200,
                    height=600
                )


                # Highlight predicted values
                param_predicted = alt.Chart(param_df[param_df['Type'].isin(['Predicted', 'Prediction'])]).mark_point(size=1, filled=False).encode(
                    x=alt.X('parameters', title='Model Size (in Billion)'),
                    y=alt.Y('total_energy_7500_prompts_Wh', title='Energy Consumed (in Wh) for 7.500 Prompts'),
                    tooltip=['parameters', 'Type']
                )

                # Highlight predicted values
                param_highlight = alt.Chart(param_df[param_df['Type'].isin(['Prediction'])]).mark_point(size=300, filled=False, strokeWidth=4).encode(
                    x=alt.X('parameters', title='Model Size (in Billion)'),
                    y=alt.Y('total_energy_7500_prompts_Wh', title='Energy Consumed (in Wh) for 7.500 Prompts'),
                    color=alt.Color('Type', title='Input Token Points', 
                                    scale=alt.Scale(
                                    domain=["Prediction"],
                                    range=["#DD7F70", "#A43489"])
                                ),
                    tooltip=['parameters', 'Type']
                )

                param_regression = param_predicted.transform_regression('parameters', 'total_energy_7500_prompts_Wh', method="exp").mark_line()

                # Combine charts
                param_chart = param_highlight + param_base + param_predicted + param_regression

                st.altair_chart(param_chart, use_container_width=True, theme="streamlit")



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

        # Transform back to 7500 prompts and Wh instead of kWh
        df['actual_non_idle_gpu_energy'] = df['non_idle_gpu_energy_per_1M_prompts'] / 1000000 * 7500 * 1000 
        df['actual_idle_gpu_energy'] = df['idle_gpu_energy_per_1M_prompts'] / 1000000 * 7500 * 1000
        df['actual_gpu_energy'] = df['gpu_energy_per_1M_prompts'] / 1000000 * 7500 * 1000
        df['actual_cpu_energy'] = df['cpu_energy_per_1M_prompts'] / 1000000 * 7500 * 1000
        df['actual_ram_energy'] = df['ram_energy_per_1M_prompts'] / 1000000 * 7500 * 1000
        df['actual_total_energy'] = df['actual_gpu_energy'] + df['actual_cpu_energy'] + df['actual_ram_energy']

        df_cleaned = df[['model_setup', 
                        'parameters', 
                        'num_gpus', 
                        'num_prompts', 
                        'total_time',
                        'time_per_prompt', 
                        'tok_per_sec',
                        'total_out_tok', 
                        'total_in_tok', 
                        'avg_out_tok', 
                        'avg_in_tok',
                        'actual_non_idle_gpu_energy',
                        'actual_idle_gpu_energy', 
                        'actual_gpu_energy',
                        'actual_cpu_energy', 
                        'actual_ram_energy',
                        'actual_total_energy']]

        # Map labels back to actual column names
        energy_type_mapping = {
            '3: CPU': 'actual_cpu_energy',
            '4: RAM': 'actual_ram_energy',
            '2: GPU Idle': 'actual_idle_gpu_energy',
            '1: GPU Non-Idle': 'actual_non_idle_gpu_energy',
            '0: GPU': 'actual_gpu_energy',
            '5: Total': 'actual_total_energy',
        }

        # Map selections to the corresponding column names
        selected_columns = [energy_type_mapping[label] for label in energy_type_selections]

        # Filter the dataframe based on selected model sizes
        df_cleaned = df_cleaned[df_cleaned['model_setup'].str.split('_').str[0].isin(param_size_selection)]

        # Prepare the dataframe for the stacked bar chart
        df_melted = df_cleaned.melt(id_vars=[
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
        df_melted['Energy_Consumption_Wh'] = df_melted['Energy_Consumption'].apply(lambda x: f"{x:.3f} Wh")

        # Reverse map the energy type columns back to labels for display
        reverse_energy_type_mapping = {v: k for k, v in energy_type_mapping.items()}
        df_melted['Energy_Type_Order'] = df_melted['Energy_Type'].map(reverse_energy_type_mapping)

        # Color mapping
        color_mapping = {
            'actual_total_energy': '#ff6600',
            'actual_cpu_energy': '#9a86b8',
            'actual_gpu_energy': '#005239',
            'actual_ram_energy': '#86b7b8',
            'actual_idle_gpu_energy': '#5bf0d2',
            'actual_non_idle_gpu_energy': '#16c473'
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
            y=alt.Y('Energy_Consumption:Q', title='Energy Consumed (in Wh)'),
            color=alt.Color('Energy_Type_Order', title='Energy Type', 
                             scale=alt.Scale(
                                domain=energy_type_selections,
                                range=[color_mapping[energy_type_mapping[label]] for label in energy_type_selections])
                             ),
            order=alt.Order('Energy_Type_Order', sort='descending'),
            tooltip=['num_prompts', 'Energy_Type_Order', 'Energy_Consumption_Wh']
        ).properties(
            width=600,
            height=600,
            title=f'Energy Consumption by Model Setup'
        )

        st.altair_chart(chart, use_container_width=True, theme="streamlit")

    st.divider()

    with st.expander("Underlying Data", expanded=False):

        st.dataframe(df_cleaned, hide_index=True)


def vllm_tests(): 
    st.title("LLM Emission Tests 🌍🌱")

    st.caption("This is a dashboard to visualize the results of the LLM emission tests.")

    st.divider()

    #st.subheader("", divider='grey')

    df = load_csv_data('emission_regression_vllm')
    comp_df = load_csv_data('transformers_vs_vllm')
    param_comp_df = load_csv_data('params_test')

    output_comp_df = clean_output_data(load_csv_data('emission_regression_vllm'))
    input_comp_df = clean_input_data(load_csv_data('input_tok_summary_vllm'))
    param_comp_df_cleaned = clean_params_data(load_csv_data('params_test'))['lowest_energy_setup']

    comp, breakdown, model_param_comp, regression  = st.tabs([
        "Transformers vs. vLLM",
        "Energy Type Breakdown", 
        "Model Parameter Comparison",
        "Energy Regression"
        ])
    
    with comp:
        st.write("")
        st.write("")

        comp_page(comp_df)

    with breakdown:

        st.write("")
        st.write("")

        breakdown_page(df)

    with regression:

        st.write("")
        st.write("")

        regression_page(output_comp_df, input_comp_df, param_comp_df_cleaned)
    
    with model_param_comp:
        st.write("")
        st.write("")

        model_param_comp_page(param_comp_df)

if __name__ == "__main__":
    init_session_states()
    
    st.session_state.curr_page = 'vllm_tests'
    sidebar()
    vllm_tests()