import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import os
import sys
import json


from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def init_session_states():

    pd.options.mode.chained_assignment = None

    if 'metric' not in st.session_state:
        st.session_state.metric = 'actual_emissions_per_10k_prompts'

    if 'degree_list' not in st.session_state:
        st.session_state.degree_list = [1]

    if 'curr_page' not in st.session_state:
        st.session_state.curr_page = 'vllm_tests'
    
    if 'MTBENCH_CATEGORIES' not in st.session_state:
        st.session_state.MTBENCH_CATEGORIES = ["Writing", "Roleplay", "Reasoning", "Math", "Coding", "Extraction", "STEM", "Humanities"]
    
    # Initialize session state for button click
    if 'qa_selections' not in st.session_state:
        st.session_state.qa_selections = {
                "question": "none",
                "model": "none",
                "quantization": "none",
                "guidance": "none"
            }
        st.session_state.gen_chat_button_clicked = False
        st.session_state.gen_judge_button_clicked = False
        st.session_state.first_gen = True
        st.session_state.first_judge = True
        st.session_state.ans_printed = False
        st.session_state.base_printed = False
    
    if 'overall_change' not in st.session_state:
        st.session_state['overall_change'] = {'percent': 0, 'factor': 1}


@st.cache_data
def load_csv_data(type='emission_regression'):
    """
    Load data from a csv file
    """

    df = pd.read_csv(f'results/data/{type}.csv')

    return df

@st.cache_data
def load_parquet_data(type='results'):
    """
    Load data from a csv file
    """

    df = pd.read_parquet(f'results/data/{type}.parquet')

    return df


@st.cache_data
def get_mt_model_df(file_path):
    q2result = []
    fin = open(f"{file_path}.jsonl", "r")
    for line in fin:
        obj = json.loads(line)
        obj["category"] = st.session_state.MTBENCH_CATEGORIES[(obj["question_id"]-81)//10]
        q2result.append(obj)
    df = pd.DataFrame(q2result)
    return df


@st.cache_data
def get_mt_model_df_pair(file_path):
    fin = open("{file_path}.jsonl", "r")
    q2result = []
    for line in fin:
        obj = json.loads(line)

        result = {}
        result["qid"] = str(obj["question_id"])
        result["turn"] = str(obj["turn"])
        if obj["g1_winner"] == "model_1" and obj["g2_winner"] == "model_1":
            result["result"] = "win"
        elif obj["g1_winner"] == "model_2" and obj["g2_winner"] == "model_2":
            result["result"] = "loss"
        else:
            result["result"] = "tie"
        result["category"] = st.session_state.MTBENCH_CATEGORIES[(obj["question_id"]-81)//10]
        result["model"] = obj["model_1"]
        q2result.append(result)

    df = pd.DataFrame(q2result)

    return df

@st.cache_data
def clean_input_data(df):

    df = df[df['test_type'] == 'Input-tok']

    df['ram_energy_10k_prompts_Wh'] = df['actual_ram_energy_per_10k_prompts'] * 1000
    df['gpu_energy_10k_prompts_Wh'] = df['actual_gpu_energy_per_10k_prompts'] * 1000
    df['cpu_energy_10k_prompts_Wh'] = df['actual_cpu_energy_per_10k_prompts'] * 1000
    df['total_energy_10k_prompts_Wh'] = df['cpu_energy_10k_prompts_Wh'] + df['gpu_energy_10k_prompts_Wh'] + df['ram_energy_10k_prompts_Wh']

    df = df[['test_type', 
            'model_type', 
            'parameters',
            'num_examples', 
            'num_prompts', 
            'total_out_tok', 
            'total_in_tok', 
            'avg_out_tok', 
            'avg_in_tok', 
            'total_energy_10k_prompts_Wh', 
            'ram_energy_10k_prompts_Wh', 
            'gpu_energy_10k_prompts_Wh', 
            'cpu_energy_10k_prompts_Wh']]

    return df

@st.cache_data
def clean_output_data(df):
    # Transform energy values from kWh to Wh
    df['total_energy_10k_prompts_Wh'] = df['actual_total_energy_per_10k_prompts'] * 1000
    df['ram_energy_10k_prompts_Wh'] = df['actual_ram_energy_per_10k_prompts'] * 1000
    df['gpu_energy_10k_prompts_Wh'] = df['actual_gpu_energy_per_10k_prompts'] * 1000
    df['cpu_energy_10k_prompts_Wh'] = df['actual_cpu_energy_per_10k_prompts'] * 1000
    df['gpu_idle_energy_10k_prompts_Wh'] = df['actual_idle_gpu_energy_per_10k_prompts'] * 1000
    df['gpu_non_idle_energy_10k_prompts_Wh'] = df['actual_non_idle_gpu_energy_per_10k_prompts'] * 1000
    df['prompt_per_sec'] = df['num_prompts'] / df['total_time']

    df = df[['test_type', 
                                                            'model_type', 
                                                            'parameters',
                                                            'num_examples', 
                                                            'num_prompts', 
                                                            'total_time', 
                                                            'prompt_per_sec', 
                                                            'total_out_tok', 
                                                            'total_in_tok', 
                                                            'avg_out_tok', 
                                                            'avg_in_tok', 
                                                            'total_energy_10k_prompts_Wh', 
                                                            'ram_energy_10k_prompts_Wh', 
                                                            'gpu_energy_10k_prompts_Wh', 
                                                            'cpu_energy_10k_prompts_Wh',
                                                            'gpu_idle_energy_10k_prompts_Wh', 
                                                            'gpu_non_idle_energy_10k_prompts_Wh']]


    return df


@st.cache_data
def clean_params_data(df):
    # Transform energy values from kWh to Wh
    df['total_energy_7500_prompts_Wh'] = df['total_energy_per_1M_prompts'] / 1000000 * 7500 * 1000
    df['ram_energy_7500_prompts_Wh'] = df['ram_energy_per_1M_prompts'] / 1000000 * 7500 * 1000
    df['gpu_energy_7500_prompts_Wh'] = df['gpu_energy_per_1M_prompts'] / 1000000 * 7500 * 1000
    df['cpu_energy_7500_prompts_Wh'] = df['cpu_energy_per_1M_prompts'] / 1000000 * 7500 * 1000
    df['gpu_idle_energy_7500_prompts_Wh'] = df['idle_gpu_energy_per_1M_prompts'] / 1000000 * 7500 * 1000
    df['gpu_non_idle_energy_7500_prompts_Wh'] = df['non_idle_gpu_energy_per_1M_prompts'] / 1000000 * 7500 * 1000
    df['prompt_per_sec'] = df['num_prompts'] / df['total_time']
    df['model_type'] = 'Code LLaMA'

    df = df[['model_setup', 
                                                            'model_type', 
                                                            'parameters',
                                                            'num_gpus',
                                                            'num_prompts', 
                                                            'total_time', 
                                                            'prompt_per_sec', 
                                                            'total_out_tok', 
                                                            'total_in_tok', 
                                                            'avg_out_tok', 
                                                            'avg_in_tok', 
                                                            'total_energy_7500_prompts_Wh', 
                                                            'ram_energy_7500_prompts_Wh', 
                                                            'gpu_energy_7500_prompts_Wh', 
                                                            'cpu_energy_7500_prompts_Wh',
                                                            'gpu_idle_energy_7500_prompts_Wh', 
                                                            'gpu_non_idle_energy_7500_prompts_Wh']]

    smallest_setup = [
        '7B_1GPUs', '13B_4GPUs', '34B_4GPUs', '70B_8GPUs'
    ]

    largest_setup = [
        '70B_8GPUs', '34B_8GPUs', '13B_8GPUs', '7B_8GPUs'
    ]

    lowest_energy_setup = [
        '7B_4GPUs', '13B_4GPUs', '34B_4GPUs', '70B_8GPUs'
    ]

    df_smallest_setup = df[df['model_setup'].isin(smallest_setup)]
    df_largest_setup = df[df['model_setup'].isin(largest_setup)]
    df_lowest_energy_setup = df[df['model_setup'].isin(lowest_energy_setup)]

    df_dict = {
        'smallest_setup': df_smallest_setup,
        'largest_setup': df_largest_setup,
        'lowest_energy_setup': df_lowest_energy_setup
        }
    
    return df_dict


@st.cache_resource
def fit_regression(X, y, regression_type='linear', polynomial_degree=2):
    if regression_type == 'linear':
        model = LinearRegression()

    elif regression_type == 'exponential':
        # Transform y for exponential regression
        y = np.log(y)
        model = LinearRegression()

    elif regression_type == 'polynomial':
        polynomial_features = PolynomialFeatures(degree=polynomial_degree)
        linear_regression = LinearRegression()
        model = make_pipeline(polynomial_features, linear_regression)

    else:
        raise ValueError("Unsupported regression type")
    
    model.fit(X, y)
    return model


def predict_values(model, pred_value_dict, type='linear', target='total_energy_10k_prompts_Wh'):
    # Define the x values for prediction
    # predicted_values = {'parameters': [7, 10, 25, 45, 50, 70, 75]}

    x_values = pd.DataFrame(pred_value_dict)

    predictions = pred_value_dict

    if type == 'exponential': 
        predictions[target] = np.exp(model.predict(x_values)).flatten()
    else:
        predictions[target] = model.predict(x_values).flatten()
        
    result_df = pd.DataFrame(predictions)

    return result_df


def combine_actual_predict_df(actual_df, predict_df, feature='avg_out_tok', target='total_energy_10k_prompts_Wh'):
    combined_df = pd.concat([actual_df[[feature, target]], predict_df[[feature, target]]], axis=0)
    combined_df['Type'] = ['Actual'] * len(actual_df) + ['Predicted'] * len(predict_df)
    return combined_df


def calculate_energy_change(model, start_val, end_val, feature_name='avg_out_tok', type='linear', percentage_range = 50, factor_range = 0.5):
    """
    Calculate the percentage and factor change in energy consumption when changing input factors.

    Parameters:
    - model: The scikit-learn model used for prediction.
    - start_val (int): The initial number of tokens or parameters.
    - end_val (int): The new number of tokens or parameters.
    - feature_name (str): The name of the feature used for prediction.

    Returns:
    - results (dict): A dictionary containing token counts, predicted energies, percentage change range, and factor change range.
    """
    # Create a DataFrame with the start and end token counts
    x_values = pd.DataFrame({feature_name: [start_val, end_val]})
    
    # Predict the energy consumption for both token counts
    if type == 'exponential':
        y_values = np.exp(model.predict(x_values).flatten())
    else:
        y_values = model.predict(x_values).flatten()
    
    # Extract the predicted energy values
    energy_start = y_values[0]
    energy_end = y_values[1]

    
    # Calculate the percentage change in energy consumption
    energy_change_percent = ((energy_end - energy_start) / energy_start) * 100
    
    # Calculate the factor change in energy consumption
    energy_change_factor = energy_end / energy_start if energy_start != 0 else np.inf
    
    # Calculate the next lower and higher multiples of 50% (or other number) for percentage change
    lower_percent = np.floor(energy_change_percent / percentage_range) * percentage_range
    upper_percent = np.ceil(energy_change_percent / percentage_range) * percentage_range

    # Calculate the next lower and higher multiples of 0.5x for factor change
    lower_factor = np.floor(energy_change_factor / factor_range) * factor_range
    upper_factor = np.ceil(energy_change_factor / factor_range) * factor_range

    # Handle percentage change range formatting

    max_percentage = 1000
    max_upper_factor = 10

    if energy_change_percent == 0:
        energy_change_range = "No Change"
    elif energy_change_percent < 0:
        if abs(energy_change_percent) > max_percentage:
            energy_change_range = f"Decrease of > {int(max_percentage)}%"
        energy_change_range = f"Decrease of {abs(energy_change_percent):.2f}%"
    else:
        if upper_percent > max_percentage:
            energy_change_range = f"> {int(max_percentage)}%"
        else:
            if lower_percent == upper_percent:
                energy_change_range = f"{int(upper_percent)}%"
            else:
                energy_change_range = f"{int(lower_percent)}% – {int(upper_percent)}%"

    # Handle factor change range formatting
    if energy_change_factor == 1:
        energy_change_factor_range = "No Change"
    elif energy_change_factor < 1:
        energy_change_factor_range = f"{energy_change_factor:.2f}x"
    else:
        if upper_factor > max_upper_factor:
            energy_change_factor_range = f"> {max_upper_factor}x"
        else:
            if lower_factor == upper_factor:
                energy_change_factor_range = f"{upper_factor:.1f}x"
            else:
                energy_change_factor_range = f"{lower_factor:.1f}x – {upper_factor:.1f}x"

    # Prepare the results dictionary
    results = {
        'start_val': start_val,
        'end_val': end_val,
        'energy_start': energy_start,
        'energy_end': energy_end,
        'energy_change_percent': energy_change_percent,
        'energy_change_range': energy_change_range,
        'energy_change_factor': energy_change_factor,
        'energy_change_factor_range': energy_change_factor_range
    }
    
    return results


def label_func(input): 

    label_dict = {

        # Emission Metrics
        'actual_emissions_per_10k_prompts': 'Emissions per 10,000 Prompts',
        'actual_cpu_energy_per_10k_prompts': 'CPU Energy per 10,000 Prompts',
        'actual_gpu_energy_per_10k_prompts': 'GPU Energy per 10,000 Prompts',
        'actual_ram_energy_per_10k_prompts': 'RAM Energy per 10,000 Prompts',
        'actual_non_idle_gpu_energy_per_10k_prompts' : 'Idle GPU Energy per 10,000 Prompts', 
        'actual_idle_gpu_energy_per_10k_prompts' : 'Non-Idle GPU Energy per 10,000 Prompts', 
        'actual_emissions_per_1M_out_tok': 'Emissions per 1M Output Tokens',



        # Benchmark Models
        "Llama-2-7b-chat": "LLaMA-2-7B",
        "Llama-2-13b-chat": "LLaMA-2-13B",
        "Llama-2-70b-chat": "LLaMA-2-70B",
        "llama-3-8B-Instruct": "LLaMA-3-8B",
        "gpt-3.5-turbo": "GPT-3.5-Turbo",
        "gpt-4": "GPT-4", 
        "gpt-4-turbo": "GPT-4-Turbo", 
        "gpt-4o": "GPT-4o", 
        "claude-v1": "Claude-v1",
        "vicuna-33b-v1.3": "Vicuna-33B",
        "vicuna-13b-v1.3": "Vicuna-13B",
        "vicuna-7b-v1.3": "Vicuna-7B",
        "Llama-3-8B-Instruct_Orce_plus": "Orca-Plus-8B",
    }


    return label_dict[input]


# Function to create charts for each test type
def create_chart(df, test_type, metric = 'actual_emissions_per_10k_prompts', remove_x_title=False):
    llama2_note = 'Note: Emissions normalized to number of output tokens for Llama2 because the Llama2 and Llama3 output differed drastically'
    chart_data = df.loc[df['test_type'] == test_type]

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
        chart_data.loc[:, 'Note'] = llama2_note
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
    chart_data = df.loc[df['test_type'] == test_type]

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

        st.page_link(page="pages/framework.py", label="Framework", icon="📐")
        st.page_link(page="pages/benchmarks.py", label="Benchmarks", icon="📊")
        st.page_link(page="pages/explain_qa.py", label="Explain Benchmark", icon="💬")
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