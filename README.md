#  Accompanying Repository for the Master's Thesis on LLM Customization Approaches and their impact on Energy Consumption.

Visit the interactive dashboard for this thesis at [thesis.d-wetzel.de](https://thesis.d-wetzel.de). <br>
Or check out my personal portfolio at [d-wetzel.de](https://d-wetzel.de).

## Abstract

### Green Tech or Digital Polluter? <br> Understanding Emission Drivers of Different Generative AI Customization and Implementation Approaches.

Generative Artificial Intelligence (Gen AI), particularly Large Language Models (LLMs), have rapidly advanced and are widely adopted in various domains. However, these models consume significant energy during training and inference, leading to substantial environmental impact. This thesis investigates methods to reduce the energy consumption of LLM inference without compromising model quality. Specifically, it examines the effectiveness of quantization, prompt engineering, Retrieval-Augmented Generation (RAG), and serving engine optimization. Experiments were conducted using models such as LLaMA~3.1 and Mistral~Nemo across different quantization levels and prompt configurations. Model quality was evaluated using the Arena Hard Auto Benchmark, automated with an LLM-as-a-Judge approach. The results demonstrate that quantization significantly enhances energy efficiency. Reducing model precision from 16-bit to 8-bit yielded substantial energy savings of up to 60\% with minimal impact on model capabilities.
Further quantization to 4-bit led to additional energy reductions, albeit with a more noticeable decrease in quality. Prompt engineering and RAG improved model quality, particularly for models with lower baseline capabilities, but increased energy consumption due to longer input sequences. Serving engine optimization, specifically using the vLLM engine, substantially improved processing speed and energy efficiency compared to traditional implementations. Based on these findings, a decision framework is developed to guide practitioners in optimizing LLM deployments for both efficiency and sustainability. The framework provides practical guidelines to achieve energy-efficient LLM applications without sacrificing quality. This work contributes significantly to advancing sustainable AI practices by offering actionable insights into optimizing LLM inference.


## Repository Structure

The repository is structured as follows:

### Root Directory:

```
streamlit-visualization.py          ### Streamlit Dashboard for the Thesis
requirements.txt                    ### Requirements for the Streamlit Dashboard
dev_requirements_macos.txt          ### Development requirements for MacOS
dev_requirements.txt                ### Development requirements for Linux
Dockerfile                          ### Dockerfile for the Streamlit Dashboard
init.sh                             ### Initialization script for Development on Linux
load_vllm.sh                        ### Script to startup the vLLM Engine with Docker
.streamlit/                         ### Streamlit Configuration Directory
```

### pages/ Directory:

Directory for all Streamlit Pages and Functions
```
benchmarks.py                       ### Benchmarking Streamlit Page 
explain_qa.py                       ### Explainable QA Streamlit Page
framework.py                        ### Framework Streamlit Page
initial_tests.py                    ### Initial Tests Streamlit Page
vllm_tests.py                       ### vLLM Tests Streamlit Page
img/                                ### Image Directory for Streamlit Pages
utils/streamlit_utils.py            ### Utility Functions for Streamlit Pages
```

### notebooks/ Directory:

Various Jupyter notebooks related to data transformation, benchmarking, and initial trials.

```
vllm_openAi_server_showcase.ipynb   ### vLLM with local OpenAI Server Showcase
explain_automatic_benchmark.ipynb   ### Demonstration of the LLM-as-a-Judge Approach

compare_param_sizes.ipynb           ### Comparison of Model Parameter Sizes
vllm_input_token_summary.ipynb      ### Input Token Summary Benchmark
vllm_vs_transformers.ipynb          ### Comparison of vLLM and Transformers

data_transformation_input_summary.ipynb         ### Data Transformation Input Summary
data_transformation_model_size.ipynb            ### Data Transformation Model Sizes
data_transformation_vllm_vs_transformers.ipynb  ### Data Transformation vLLM vs Transformers

initial_trial_phase/                ### Initial Trials done at the Beginning of the Thesis
quantization/                       ### Quantization Trials
```

### results/ Directory:

Directory of all the benchmark results used for the Streamlit Dashboard. 


### llm_judge/ Directory:

Directory containing containing code for the automated LLM-as-a-Judge Benchmarks.

```
batch_model_benchmarks.ipynb        ### Batch Model Benchmarking Notebook
benchmark_results.ipynb             ### Notebook to process the Benchmark Results
emissions_batched_benchmakrs.csv    ### Batched Benchmark CodeCarbon Energy & Emission Data 
emissions_benchmakrs.csv            ### Benchmark CodeCarbon Energy & Emission Data 
model_benchmarks.ipynb              ### Single Model Benchmarking Notebook
process_emissiondata.ipynb          ### Process CodeCarbon Energy & Emission Data 
process_example_questions.ipynb     ### Process Example Questions for the Explain Q&A Page
```

### llm_judge/arena-hard-auto/ Directory:

Submodule with a Fork of the Arena Hard Auto Repo. <br>
The main adjustments are: 
- Added a `gen_guidance.py` script to automatically create the guidance files for the benchmark questions. 
- Added the `config/gen_guidance_config.yaml` file to configure the guidance generation.
- Updated the other present scripts of this benchmark to allow for Azure GPT Batch Execution as well as AWS Bedrock Execution.


### energy_by_region Directory:

Directory containing the data transformations of the historical [electricityMaps](https://app.electricitymaps.com/map) data. 

```
transform_energy_data.ipynb         ### Data Transformation of the electricityMaps Data
data_visualization.py               ### Streamlit Data Visualization of the electricityMaps Data
raw_data/                           ### Raw Data Directory
cleaned_data/                       ### Cleaned Data Directory
```


### open_web_ui Directory:

Directory containing scripts to start Open WebUI with the locally executed models. 

```
litellm_config.yaml                 ### Configuration File for the LiteLLM Engine 
load_liteLLM.sh                     ### Script to start the LiteLLM Engine
load_OpenWebUI_wLiteLLM.sh          ### Script to start OpenWebUI with the LiteLLM Engine
load_OpenWebUI.sh                   ### Script to start OpenWebUI 

pipelines/                          ### Pipeline Scripts for OpenWebUI
    AWS_Bedrock_Pipeline.py
    AzureOpenAI_Pipeline.py
```



## Development

### Setup
To start running your own benchmarks you first need to install the requirements. <br>
To streamline this process I have created a shell script that installs the requirements for you on a linux system (tested on Ubuntu 20.04). <br>
The startup script also initializes and updates the submodules within this repo. <br> 

To run the initialization script, execute the following command in the root directory of this repository:

```bash
./init.sh
```

If you wish setup your environment manually, run the following commands:

```bash
pip install -r dev_requirements.txt
```

```
git submodule init
git submodule update
```

Afterwards, you need to specify the Huggingface & WandB API Keys:

```bash
huggingface-cli login
```

```bash
wandb login
```

### Startup vLLM with Docker exposing a local OpenAI API Server

To start the vLLM Engine with Docker and expose a local OpenAI API Server, run the following command:

(Using LLaMA 3.1 8B Instruct on a setup with 4 Nvidia GPUs)

```bash
./load_vllm.sh
```

or manually execute the following commands:

```bash
docker run --runtime nvidia --gpus all -v ~/.cache/huggingface:/root/.cache/huggingface -p 8000:8000 --ipc=host vllm/vllm-openai:latest --model meta-llama/Meta-Llama-3.1-8B-Instruct --tensor-parallel-size 4 --enable-chunked-prefill --served-model-name llama3_1_8b
```


### Run Arena Hard Auto Benchmarks

To run the Arena Hard Auto Benchmarks, you can use the following notebook:

```
llm_judge/arena-hard-auto/batch_model_benchmarks.ipynb
````

To generate the guidance files for the benchmark questions, you can use the following command: 


```bash
cd llm_judge/arena-hard-auto/

python gen_guidance.py
```

The configuration for the guidance generation can be found in the `config/gen_guidance_config.yaml` file. <br>
The configuration for the model API endpoints can be found in the `config/api_config.yaml` file.


### Run the Input Token Summary Benchmark

To run the Input Token Moby Dick Summary Benchmark, you can use the following notebook:

```
notebooks/vllm_input_token_summary.ipynb 
```


### Run the Streamlit Dashboard 

To run the Streamlit Dashboard, you can execute the following command in the root directory of this repo: 
```
streamlit run streamlit-visualization.py
```