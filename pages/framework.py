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





def framework_page(output_comp_df, input_comp_df, param_comp_df):

    # st.subheader("Arena Hard v0.1 Results")
    # st.write("This section provides an overview of the results of the Arena Hard v0.1 Automated Benchmark.")
    # st.write("The Judge Model was GPT-4o and the Baseline Model was GPT-4-0314")

    # st.write("")

    col1, col2 = st.columns([1, 4], gap="large")

    with col1:
        st.write("")
        st.write("")
        st.button("Refresh", use_container_width=True)

    with col2:

        if st.session_state['overall_change']['factor'] < 0.3:
            change = f"{st.session_state['overall_change']['factor']:.2f}x"
        else:
            change = f"{st.session_state['overall_change']['factor']:.1f}x"

        change_percent = f"{st.session_state['overall_change']['percent']:.0f}%"

        st.title(f"All Measures caused an Energy Consumption Change of {change_percent} or {change}")

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
        input_model = fit_regression(X=input_feature, y=input_target, regression_type='linear', polynomial_degree=2)
        output_model = fit_regression(X=output_feature, y=output_target, regression_type='polynomial', polynomial_degree=2)
        param_model = fit_regression(X=param_feature, y=param_target, regression_type='exponential')

        input_predict_values = {'avg_in_tok': [10, 50, 250, 500, 1000, 1500, 2000, 3000, 4000, 5000, 10000, 30000, 55000, 75000]}
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
        

        st.subheader("Changes in Quantization")

        quant_col1, quant_col2, quant_col3 = st.columns([2, 2, 2])

        with quant_col1:
            quant_start = st.select_slider("Precision before Change", options=["int-4", "fp-8", "bf-16"], value="bf-16")
        
            quant_end = st.select_slider("Precision after Change", options=["int-4", "fp-8", "bf-16"], value="bf-16")

        if quant_start == "bf-16" and quant_end == "fp-8":
            quant_energy_change = 0.4
        
        elif quant_start == "bf-16" and quant_end == "int-4":
            quant_energy_change = 0.3188
        
        elif quant_start == "fp-8" and quant_end == "bf-16":
            quant_energy_change = 2.496
        
        elif quant_start == "fp-8" and quant_end == "int-4":
            quant_energy_change = 0.7957
        
        elif quant_start == "int-4" and quant_end == "bf-16":
            quant_energy_change = 3.137
        
        elif quant_start == "int-4" and quant_end == "fp-8":
            quant_energy_change = 1.2567
        
        else:
            quant_energy_change = 1

        quant_energy_change_percent = (quant_energy_change - 1) * 100

        
        with quant_col3:
            st.metric(label="Energy Consumption Change", value=f"{quant_energy_change:.1f}x")
        
        
            st.metric(label="Energy Consumption Change (%)", value=f"{quant_energy_change_percent:.0f}%")


        st.write("")

        st.divider()

        st.write("")

        st.subheader("Changes in the Input Token Count")

        input_col1, input_col2, input_col3 = st.columns([2, 2, 2])

        with input_col1:
            input_tok_start = st.number_input("Input Token Count before Change",  min_value=1, max_value=1000000, value=256, step=64)

            input_tok_end = st.number_input("Input Token Count after Change",  min_value=1, max_value=1000000, value=256, step=64)
        

        # Prepare the results dictionary
        # results = {
        #     'start_val': start_val,
        #     'end_val': end_val,
        #     'energy_start': energy_start,
        #     'energy_end': energy_end,
        #     'energy_change_percent': energy_change_percent,
        #     'energy_change_range': energy_change_range,
        #     'energy_change_factor': energy_change_factor,
        #     'energy_change_factor_range': energy_change_factor_range
        # }

        input_energy_change = calculate_energy_change(input_model, input_tok_start, input_tok_end, feature_name='avg_in_tok', percentage_range = 50, factor_range = 0.5)

        

        with input_col3:
            st.metric(label="Energy Consumption Change", value=input_energy_change['energy_change_factor_range'])
        
        
            st.metric(label="Energy Consumption Change (%)", value=input_energy_change['energy_change_range'])
        
        start_pred_df = predict_values(model=input_model, 
                                pred_value_dict={'avg_in_tok': [input_tok_start]}, 
                                type='linear', 
                                target='total_energy_100k_output_tokens_Wh')
        
        start_pred_df = pd.concat([start_pred_df, pd.DataFrame({'Type': ['Start']})], axis=1, copy=False)
        
        end_pred_df = predict_values(model=input_model, 
                                pred_value_dict={'avg_in_tok': [input_tok_end]}, 
                                type='linear', 
                                target='total_energy_100k_output_tokens_Wh')
        
        end_pred_df = pd.concat([end_pred_df, pd.DataFrame({'Type': ['End']})], axis=1, copy=False)

        input_df = pd.concat([input_df, start_pred_df, end_pred_df], axis=0, ignore_index=True, copy=False)

        with st.expander("Visualize Energy Consumption Change", expanded=False):

            # Create Altair chart
            input_base = alt.Chart(input_df[input_df['Type'] == 'Actual']).mark_point(size=20, filled=True).encode(
                x=alt.X('avg_in_tok', title='Average Input Tokens per Prompt'),
                y=alt.Y('total_energy_100k_output_tokens_Wh', axis=None),
                tooltip=['avg_in_tok', 'Type']
            ).properties(
                width=1200,
                height=600
            )


            # Highlight predicted values
            input_predicted = alt.Chart(input_df[input_df['Type'].isin(['Predicted', 'Start', 'End'])]).mark_point(size=1, filled=False).encode(
                x=alt.X('avg_in_tok', title='Average Input Tokens per Prompt'), 
                y=alt.Y('total_energy_100k_output_tokens_Wh', axis=None),
                tooltip=['avg_in_tok', 'Type']
            )

            # Highlight predicted values
            input_highlight = alt.Chart(input_df[input_df['Type'].isin(['Start', 'End'])]).mark_point(size=300, filled=False, strokeWidth=4).encode(
                x=alt.X('avg_in_tok', title='Average Input Tokens per Prompt'), 
                y=alt.Y('total_energy_100k_output_tokens_Wh', axis=None),
                color=alt.Color('Type', title='Input Token Points', 
                                scale=alt.Scale(
                                domain=["Start", "End"],
                                range=["#DD7F70", "#A43489"])
                             ),
                tooltip=['avg_in_tok', 'Type']
            )

            input_regression = input_predicted.transform_regression('avg_in_tok', 'total_energy_100k_output_tokens_Wh', method="linear").mark_line()

            # Combine charts
            input_chart = input_highlight + input_base + input_predicted + input_regression

            st.altair_chart(input_chart, use_container_width=True, theme="streamlit")


            
        st.divider()

        # st.dataframe(input_pred_df)
        # st.dataframe(output_pred_df)
        # st.dataframe(param_pred_df)


        st.subheader("Changes in the Output Token Count")

        output_col1, output_col2, output_col3 = st.columns([2, 2, 2])

        with output_col1:
            output_tok_start = st.number_input("Output Token Count before Change",  min_value=1, max_value=1000000, value=256, step=64)

            output_tok_end = st.number_input("Output Token Count after Change",  min_value=1, max_value=1000000, value=256, step=64)
        

        # Prepare the results dictionary
        # results = {
        #     'start_val': start_val,
        #     'end_val': end_val,
        #     'energy_start': energy_start,
        #     'energy_end': energy_end,
        #     'energy_change_percent': energy_change_percent,
        #     'energy_change_range': energy_change_range,
        #     'energy_change_factor': energy_change_factor,
        #     'energy_change_factor_range': energy_change_factor_range
        # }

        output_energy_change = calculate_energy_change(output_model, output_tok_start, output_tok_end, feature_name='avg_out_tok', percentage_range = 50, factor_range = 0.5)

        

        with output_col3:
            st.metric(label="Energy Consumption Change", value=output_energy_change['energy_change_factor_range'])
        
        
            st.metric(label="Energy Consumption Change (%)", value=output_energy_change['energy_change_range'])
        
        start_pred_df = predict_values(model=output_model, 
                                pred_value_dict={'avg_out_tok': [output_tok_start]}, 
                                type='polynomial', 
                                target='total_energy_10k_prompts_Wh')
        
        start_pred_df = pd.concat([start_pred_df, pd.DataFrame({'Type': ['Start']})], axis=1, copy=False)
        
        end_pred_df = predict_values(model=output_model, 
                                pred_value_dict={'avg_out_tok': [output_tok_end]}, 
                                type='polynomial', 
                                target='total_energy_10k_prompts_Wh')
        
        end_pred_df = pd.concat([end_pred_df, pd.DataFrame({'Type': ['End']})], axis=1, copy=False)

        output_df = pd.concat([output_df, start_pred_df, end_pred_df], axis=0, ignore_index=True, copy=False)

        with st.expander("Visualize Energy Consumption Change", expanded=False):

            # Create Altair chart
            output_base = alt.Chart(output_df[output_df['Type'] == 'Actual']).mark_point(size=20, filled=True).encode(
                x=alt.X('avg_out_tok', title='Average Input Tokens per Prompt'),
                y=alt.Y('total_energy_10k_prompts_Wh', axis=None),
                tooltip=['avg_out_tok', 'Type']
            ).properties(
                width=1200,
                height=600
            )


            # Highlight predicted values
            output_predicted = alt.Chart(output_df[output_df['Type'].isin(['Predicted', 'Start', 'End'])]).mark_point(size=1, filled=False).encode(
                x=alt.X('avg_out_tok', title='Average Input Tokens per Prompt'), 
                y=alt.Y('total_energy_10k_prompts_Wh', axis=None),
                tooltip=['avg_out_tok', 'Type']
            )

            # Highlight predicted values
            output_highlight = alt.Chart(output_df[output_df['Type'].isin(['Start', 'End'])]).mark_point(size=300, filled=False, strokeWidth=4).encode(
                x=alt.X('avg_out_tok', title='Average Input Tokens per Prompt'), 
                y=alt.Y('total_energy_10k_prompts_Wh', axis=None),
                color=alt.Color('Type', title='Input Token Points', 
                                scale=alt.Scale(
                                domain=["Start", "End"],
                                range=["#DD7F70", "#A43489"])
                             ),
                tooltip=['avg_out_tok', 'Type']
            )

            output_regression = output_predicted.transform_regression('avg_out_tok', 'total_energy_10k_prompts_Wh', method="quad").mark_line()

            # Combine charts
            output_chart = output_highlight + output_base + output_predicted + output_regression

            st.altair_chart(output_chart, use_container_width=True, theme="streamlit")

    
        st.divider()

        # st.dataframe(input_pred_df)
        # st.dataframe(output_pred_df)
        # st.dataframe(param_pred_df)


        st.subheader("Changes in the Model Size")

        param_col1, param_col2, param_col3 = st.columns([2, 2, 2])

        with param_col1:
            param_start = st.number_input("Model Size before Change (in Billion)",  min_value=0.1, max_value=4000.0, value=70.0, step=2.0)

            param_end = st.number_input("Model Size after Change (in Billion)",  min_value=0.1, max_value=4000.0, value=70.0, step=2.0)
        

        # Prepare the results dictionary
        # results = {
        #     'start_val': start_val,
        #     'end_val': end_val,
        #     'energy_start': energy_start,
        #     'energy_end': energy_end,
        #     'energy_change_percent': energy_change_percent,
        #     'energy_change_range': energy_change_range,
        #     'energy_change_factor': energy_change_factor,
        #     'energy_change_factor_range': energy_change_factor_range
        # }

        param_energy_change = calculate_energy_change(param_model, param_start, param_end, feature_name='parameters', type='exponential', percentage_range = 50, factor_range = 0.5)

        

        with param_col3:
            st.metric(label="Energy Consumption Change", value=param_energy_change['energy_change_factor_range'])
        
        
            st.metric(label="Energy Consumption Change (%)", value=param_energy_change['energy_change_range'])
        
        start_pred_df = predict_values(model=param_model, 
                                pred_value_dict={'parameters': [param_start]}, 
                                type='exponential', 
                                target='total_energy_7500_prompts_Wh')
        
        start_pred_df = pd.concat([start_pred_df, pd.DataFrame({'Type': ['Start']})], axis=1, copy=False)
        
        end_pred_df = predict_values(model=param_model, 
                                pred_value_dict={'parameters': [param_end]}, 
                                type='exponential', 
                                target='total_energy_7500_prompts_Wh')
        
        end_pred_df = pd.concat([end_pred_df, pd.DataFrame({'Type': ['End']})], axis=1, copy=False)

        param_df = pd.concat([param_df, start_pred_df, end_pred_df], axis=0, ignore_index=True, copy=False)

        with st.expander("Visualize Energy Consumption Change", expanded=False):

            # Create Altair chart
            param_base = alt.Chart(param_df[param_df['Type'] == 'Actual']).mark_point(size=20, filled=True).encode(
                x=alt.X('parameters', title='Average Input Tokens per Prompt'),
                y=alt.Y('total_energy_7500_prompts_Wh', axis=None),
                tooltip=['parameters', 'Type']
            ).properties(
                width=1200,
                height=600
            )


            # Highlight predicted values
            param_predicted = alt.Chart(param_df[param_df['Type'].isin(['Predicted', 'Start', 'End'])]).mark_point(size=1, filled=False).encode(
                x=alt.X('parameters', title='Average Input Tokens per Prompt'), 
                y=alt.Y('total_energy_7500_prompts_Wh', axis=None),
                tooltip=['parameters', 'Type']
            )

            # Highlight predicted values
            param_highlight = alt.Chart(param_df[param_df['Type'].isin(['Start', 'End'])]).mark_point(size=300, filled=False, strokeWidth=4).encode(
                x=alt.X('parameters', title='Average Input Tokens per Prompt'), 
                y=alt.Y('total_energy_7500_prompts_Wh', axis=None),
                color=alt.Color('Type', title='Input Token Points', 
                                scale=alt.Scale(
                                domain=["Start", "End"],
                                range=["#DD7F70", "#A43489"])
                             ),
                tooltip=['parameters', 'Type']
            )

            param_regression = param_predicted.transform_regression('parameters', 'total_energy_7500_prompts_Wh', method="exp").mark_line()

            # Combine charts
            param_chart = param_highlight + param_base + param_predicted + param_regression

            st.altair_chart(param_chart, use_container_width=True, theme="streamlit")


        overall_factor = (
            param_energy_change['energy_change_factor'] *
            output_energy_change['energy_change_factor'] * 
            input_energy_change['energy_change_factor'] * 
            quant_energy_change
        ) 

        overall_percent = (overall_factor - 1) *100

        st.session_state['overall_change'] = {
            'percent': overall_percent,
            'factor': overall_factor
        }


def framework(): 
    st.title("LLM Emission - Framework 🌍🌱")

    st.caption("""
               This is an interactive Representation of the Framework developed in the Master's Thesis on LLM Emissions. \n
               The Framework is designed to help in the Decision Process of reducing the Energy Consumption of deployed LLMs.
               """)

    st.divider()

    output_comp_df = clean_output_data(load_csv_data('emission_regression_vllm'))
    input_comp_df = clean_input_data(load_csv_data('input_tok_summary_vllm'))
    param_comp_df = clean_params_data(load_csv_data('params_test'))['lowest_energy_setup']

    # arena_results = st.tabs([
    #     "Arena Results",
    #     ])
    

    # with arena_results:
    st.write("")
    st.write("")

    framework_page(output_comp_df, input_comp_df, param_comp_df)



if __name__ == "__main__":
    init_session_states()
    
    st.session_state.curr_page = 'framework'
    sidebar()
    framework()