# WSL_Prompting

For running the label aggregation model, we use Wrench library
[1] Install anaconda: Instructions here: https://www.anaconda.com/download/

[2] Clone the wrench repository:

git clone https://github.com/JieyuZ2/wrench.git
cd wrench
[3] Create virtual environment:

conda create -f environment.yml
source activate wrench

If the above environment installation does not succeed, try exported_env.yml file provided in the current repo:
conda create -f exported_env.yml
source activate wrenchj

For running BERT finetuning, run the colab notebook, currently setup for the AGNews dataset.

For using Open AI LLM to annotate/classify on text, refer to the llmparser module and do a standard node installation.
