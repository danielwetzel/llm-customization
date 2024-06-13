import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import os
import sys
import plotly.express as px
import plotly.graph_objects as go

from pages.utils.streamlit_utils import *





def mt_bench_page(df): 

    st.subheader("LLM Performance Benchmark - MT-Bench")
    st.write("This section visualizes the MT-Bench Performance Benchmark for LLMs.")

    st.write("")

    with st.container(border=True):

        all_models = df["model"].unique()

        #st.write(all_models)

        scores_all = []
        for model in all_models:
            for cat in st.session_state.MTBENCH_CATEGORIES:
                # filter category/model, and score format error (<1% case)
                res = df[(df["category"]==cat) & (df["model"]==model) & (df["score"] >= 0)]
                score = res["score"].mean()

                scores_all.append({"model": model, "category": cat, "score": score})

        
        models = [
            "Llama-2-7b-chat", 
            "Llama-2-13b-chat", 
            "Llama-2-70b-chat", 
            "llama-3-8B-Instruct", 
            "gpt-3.5-turbo", 
            "gpt-4", 
            "claude-v1", 
            "vicuna-33b-v1.3", 
            "vicuna-13b-v1.3", 
            "vicuna-7b-v1.3"]
        
        pre_select_models = [
            #"Llama-2-7b-chat", 
            #"Llama-2-13b-chat", 
            "Llama-2-70b-chat", 
            "llama-3-8B-Instruct", 
            "gpt-3.5-turbo", 
            "gpt-4"]

        target_models = st.multiselect("Select Models to Review", models, default=pre_select_models, key="mtbench_models", format_func=label_func)

        st.divider()



        scores_target = [scores_all[i] for i in range(len(scores_all)) if scores_all[i]["model"] in target_models]

        # sort by target_models
        scores_target = sorted(scores_target, key=lambda x: target_models.index(x["model"]), reverse=True)

        df_score = pd.DataFrame(scores_target)
        df_score = df_score[df_score["model"].isin(target_models)]

        rename_map = {"Llama-2-7b-chat": "LLaMA-2-7B",
                "Llama-2-13b-chat": "LLaMA-2-13B",
                "Llama-2-70b-chat": "LLaMA-2-70B",
                "llama-3-8B-Instruct": "LLaMA-3-8B",
                "gpt-3.5-turbo": "GPT-3.5-Turbo",
                "gpt-4": "GPT-4", 
                "claude-v1": "Claude-v1",
                "vicuna-33b-v1.3": "Vicuna-33B",
                "vicuna-13b-v1.3": "Vicuna-13B",
                "vicuna-7b-v1.3": "Vicuna-7B",
                }

        for k, v in rename_map.items():
            df_score.replace(k, v, inplace=True)

    

        fig = px.line_polar(df_score, r = 'score', theta = 'category', line_close = True, category_orders = {"category": st.session_state.MTBENCH_CATEGORIES},
                    color = 'model', markers=True, color_discrete_sequence=px.colors.qualitative.Dark2, height=900)
        
        #fig.update_layout(legend=dict(
        #    yanchor="top",
        #    y=0.3,
            #xanchor="left",
            #x=0.99
        #))

        st.plotly_chart(fig, use_container_width=False, theme="streamlit")


def vicuana_bench_page(df): 

    st.subheader("LLM Performance Benchmark - Vicuana Benchmark")
    st.write("This section visualizes the Vicuana Performance Benchmark for LLMs.")

    st.write("")

    with st.container(border=True):

        st.write("...")


def benchmarks(): 
    st.title("LLM Emission Tests 🌍🌱")

    st.caption("This is a dashboard to visualize the results of the LLM emission tests.")

    st.divider()

    #st.subheader("", divider='grey')

    df = get_mt_model_df('llm_judge/results/gpt-4_single')
    #df_pair = get_mt_model_df_pair('llm_judge/results/gpt-4_pair')

    mt_bench, vicuana = st.tabs(["MT-Bench Benchmark", "Vicuana Benchmark"])
    
    with mt_bench:

        st.write("")
        st.write("")

        mt_bench_page(df)
    
    with vicuana:

        st.write("")
        st.write("")

        vicuana_bench_page(df)

if __name__ == "__main__":
    init_session_states()
    
    st.session_state.curr_page = 'benchmarks'
    sidebar()
    benchmarks()