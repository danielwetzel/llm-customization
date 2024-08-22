import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import os
import sys
import plotly.express as px
import plotly.graph_objects as go
import re
import time

import concurrent.futures
from streamlit.runtime.scriptrunner import add_script_run_ctx

from pages.utils.streamlit_utils import *

# Function to strip markdown formatting and replace newlines
def strip_markdown_system_prompt(text):
    # Remove bold and italic
    #text = re.sub(r'\*\*|\*', '', text)
    # Remove headings
    text = re.sub(r'#+ ', '', text)
    # Remove backticks
    #text = re.sub(r'`', '', text)
    # Ensure \n is preserved and multiple spaces are allowed for formatting
    #text = re.sub(r'\n+', '\n', text)
    return text


def strip_markdown_user_question(text):
    # Remove bold and italic
    text = re.sub(r'\*\*|\*', '', text)
    # Remove headings
    text = re.sub(r'#+ ', '', text)
    # Remove backticks
    text = re.sub(r'`', '', text)
    # Replace multiple spaces with a single space
    text = re.sub(r' +', ' ', text)
    # Ensure \n is preserved but remove excessive newlines
    #text = re.sub(r'\n+', ' ', text)
    text = text.strip()
    return text

# Function to stream data
def stream_text(text, delay=0.04, seq_len=4):
    """
    Yields text one word at a time with a delay.
    :param text: Full text to stream
    :param delay: Delay between words (in seconds)
    """
    words = text.split(" ")
    seq = ""
    word_count = 0
    for word in words:
        word_count += 1
        seq += word + " "
        if word_count % seq_len == 0:
            yield seq
            seq = ""
        
            time.sleep(delay)
    
    if seq:
        yield seq

def stream_answer_or_baseline(text, col, is_baseline=False, model_name="GPT-4-0314", fast=False):
    if is_baseline:
        st.session_state.base_printed = True
        avatar = "pages/img/robot_2.svg"
        header = f"**Assistant B — {model_name} (Baseline)**"
    else:
        st.session_state.ans_printed = True
        avatar = "pages/img/robot.svg"
        header = f"**Assistant A — {model_name}**"
    
    if fast:
        delay = 0.01
        seq_len = 8
    else:
        delay = 0.05
        seq_len = 4

    with col:
        with st.chat_message("ans", avatar=avatar):
            with st.expander(header, expanded=True):
                st.write_stream(stream_text(text, delay, seq_len))    



def show_chat(df):
    st.subheader("Demonstrate automated Arena Chats")
    st.write("This section provides an overview of how the LLMs were Queried in the automated Arena Benchmark")
    st.write("The Knowledge Embedding was created automatically using GPT-4o. It features Guidance in form of important context, useful information and a step-by-step plan to solve the task.")
    st.write("The Judge Model was GPT-4o and the Baseline Model was GPT-4-0314")

    st.write("")

    with st.container(border=True):
        # Define the cleartext names for the models
        model_names = {
            'gpt-4o-mini': 'GPT-4o Mini',
            'mistral_large_2': 'Mistral Large 2407',
            'mistral_nemo': 'Mistral Nemo',
            'llama3_1_405b': 'Llama 3.1 405B',
            'llama3_1_70b': 'Llama 3.1 70B',
            'llama3_1_8b': 'Llama 3.1 8B',
            'llama3_70b': 'Llama 3.70B'
        }

        quant_models = ['Mistral Nemo', 'Llama 3.1 70B', 'Llama 3.1 8B']

        # Reverse mapping to get model_id from cleartext name
        model_id_mapping = {v: k for k, v in model_names.items()}

        # Create a mapping between numbers 1 to 20 and the actual qid values
        qid_mapping = {i: qid for i, qid in enumerate(df['qid'].unique()[:35])}

        # Create a list of options for the select box where each option is "Question {i+1} - {Cluster Name}"
        question_options = [f"Question {i} - {df.loc[df['qid'] == qid, 'cluster'].values[0]}" for i, qid in qid_mapping.items()]

        ques, mod, quant, guid, skip = st.columns([0.3, 0.2, 0.15, 0.15, 0.2], gap="medium")

        with ques:
            # Selectbox for Question with numbered display
            selected_question_option = st.selectbox("Question", question_options, index=5)

            # Get the corresponding qid based on the selected numbered question
            selected_number = int(selected_question_option.split('-')[0].replace("Question ", ""))
            qid = qid_mapping[selected_number]

            # Filter the DataFrame for the selected question
            selected_row = df[df['qid'] == qid]
        
        with mod:
            # Selectbox for Model
            selected_model_name = st.selectbox("Benchmarked Model", list(model_names.values()), index=2)

        with quant:
            # Selectbox for Quantization
            if selected_model_name in quant_models:
                quant = ['bf16', 'FP8', 'int4']
            else: 
                quant = ['bf16']
            
            quantization = st.selectbox("Quantization", quant)

        with guid:
            # Checkbox for Guided or Not
            is_guided = st.selectbox("Guidance", ['Non-Guided', 'Guided'])

        st.divider()
        st.write("")
        st.write("")

        # Determine the relevant columns based on user selections
        model_id = model_id_mapping[selected_model_name]
        guidance_suffix = "_guided" if is_guided == "Guided" else ""
        quant_suffix = f"_{quantization.lower()}" if quantization != "bf16" else ""

        # Build the column names dynamically
        answer_col = f"{model_id}{quant_suffix}{guidance_suffix}"
        judgment_col = f"{answer_col}_judgment"
        score_col = f"{answer_col}_score"

        if is_guided == "Guided":
            system_prompt_col = "guided_system_prompt"
        else:
            system_prompt_col = "system_prompt"

        # Select the relevant columns
        selected_columns = ['qid', 'question', 'cluster', 'guidance', system_prompt_col, 
                            'judge_prompt_start', 'judge_prompt', 'baseline_answer_gpt4-0314', 
                            answer_col, judgment_col, score_col]

        # Filter the row to include only these columns
        filtered_row = selected_row[selected_columns].rename(columns={
            answer_col: 'answer',
            judgment_col: 'judgment',
            score_col: 'score',
            'baseline_answer_gpt4-0314': 'baseline_answer', 
            system_prompt_col: 'system_prompt'
        })

        # Strip Markdown from system prompt and guidance while preserving formatting
        for col in ['system_prompt', 'guidance', 'answer', 'judgment', 'baseline_answer']:
            filtered_row[col] = filtered_row[col].apply(strip_markdown_system_prompt)

        # Strip Markdown more aggressively from the user question
        filtered_row['question'] = filtered_row['question'].apply(strip_markdown_user_question)

        #t.button(":material/checkbook:  Generate Answer")

        if (st.session_state.qa_selections["question"] != selected_question_option or
                st.session_state.qa_selections["model"] != selected_model_name or
                st.session_state.qa_selections["quantization"] != quantization or
                st.session_state.qa_selections["guidance"] != is_guided):
            
            st.session_state.gen_chat_button_clicked = False
            st.session_state.gen_judge_button_clicked = False
            st.session_state.first_gen = True
            st.session_state.first_judge = True
            st.session_state.ans_printed = False
            st.session_state.base_printed = False

            st.session_state.qa_selections = {
                "question": selected_question_option,
                "model": selected_model_name,
                "quantization": quantization,
                "guidance": is_guided
            }
        
        with skip:
            st.write("")
            skip_animation = st.toggle("Skip Animation")
            if skip_animation:
                st.session_state.first_gen = False
                st.session_state.first_judge = False
                st.session_state.gen_chat_button_clicked = True
                st.session_state.gen_judge_button_clicked = True

        #st.button("Generate Answer :material/text_rotate_vertical:")

        with st.chat_message("system", avatar="pages/img/system_prompt.svg"):
            #st.write("**System Prompt**")
            with st.expander("**System Prompt**", expanded=False):
                st.write(filtered_row['system_prompt'].values[0])

                if is_guided == "Guided":
                    st.divider()
                    st.write("**Guidance**")
                    st.write(filtered_row['guidance'].values[0])



        chat, gen_button = st.columns([0.8, 0.2])


        with chat:
            with st.chat_message("question", avatar="pages/img/person-circle.svg"):
                with st.expander("**Question**", expanded=True):
                    st.write(filtered_row['question'].values[0])
        

        with gen_button:
            st.write("")
            if st.button(":material/draw: Generate Answer", use_container_width=True, type="primary", key="gen_chat_button"):
                st.session_state.gen_chat_button_clicked = True
        
        

        if st.session_state.gen_chat_button_clicked:

            st.divider()

            ans, base = st.columns([2, 2], gap="medium")

            if st.session_state.first_gen:
                st.session_state.first_gen = False
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future_answer = executor.submit(stream_answer_or_baseline, filtered_row['answer'].values[0], ans, False, selected_model_name)
                    future_baseline = executor.submit(stream_answer_or_baseline, filtered_row['baseline_answer'].values[0], base, True, "GPT-4-0314")

                    add_script_run_ctx(future_answer)
                    add_script_run_ctx(future_baseline)

                    for t in executor._threads:
                        add_script_run_ctx(t)
                    
                    for t in executor._threads:
                        add_script_run_ctx(t)


            else:
                with ans:
                    with st.chat_message("assistant", avatar="pages/img/robot.svg"):
                        with st.expander(f"**Assistant A — {selected_model_name}**", expanded=True):
                            st.write(filtered_row['answer'].values[0])     
                with base:
                    with st.chat_message("assistant_b", avatar="pages/img/robot_2.svg"):
                        with st.expander(f"**Assistant B — GPT-4-0314 (Baseline)**", expanded=True):
                            st.write(filtered_row['baseline_answer'].values[0]) 
    

            st.divider()

            judge, gen_judge_button = st.columns([0.8, 0.2])
            with judge:
                with st.chat_message("question", avatar="pages/img/person-circle.svg"):
                    with st.expander("**Judge Prompt**", expanded=False):
                        st.write(filtered_row['judge_prompt_start'].values[0])
                        st.write(filtered_row['judge_prompt'].values[0])
            
            with gen_judge_button:
                st.write("")
                if st.button(":material/gavel: Generate Judgment", use_container_width=True, type="primary", key="gen_judge_button"):
                    st.session_state.gen_judge_button_clicked = True
            
            if st.session_state.gen_judge_button_clicked:

                st.divider()
                

                with st.chat_message("judge", avatar="pages/img/judge.svg"):
                    with st.expander("**Judge — GPT-4o**", expanded=True):
                        if st.session_state.first_judge:
                            st.write_stream(stream_text(filtered_row['judgment'].values[0]))
                            st.session_state.first_judge = False
                        else:
                            st.write(filtered_row['judgment'].values[0])

                gap, met_1, met_2, met_3 = st.columns([1, 4, 3, 4], gap="medium")

                met_1.metric("Model", selected_model_name)
                met_3.metric("Baseline", "GPT-4-0314",)

                if filtered_row['score'].values[0] == 'A>B' or filtered_row['score'].values[0] == 'A>>B':
                    met_2.metric("Score", filtered_row['score'].values[0], f"{selected_model_name} Wins", delta_color="normal")
                elif filtered_row['score'].values[0] == 'B>A':
                    met_2.metric("Score", 'A<B', "GPT-4-0314 Wins", delta_color="inverse")
                elif filtered_row['score'].values[0] == 'B>>A':
                    met_2.metric("Score", 'A<<B', "GPT-4-0314 Wins", delta_color="inverse")
                else:
                    met_2.metric("Score", filtered_row['score'].values[0], "Tie", delta_color="off")



def explain_qa(): 
    st.title("LLM Emission Tests 🌍🌱")

    st.caption("This is a dashboard to visualize the results of the LLM emission tests.")

    st.divider()

    df = load_parquet_data('bench_qa_deepdive')
    
    chat, elo = st.tabs([
        "Arena Chats",
        "Elo Calculations"
        ])
    
    with chat:
        st.write("")
        st.write("")

        show_chat(df)
    


if __name__ == "__main__":
    init_session_states()
    
    st.session_state.curr_page = 'benchmarks'
    sidebar()
    explain_qa()

   