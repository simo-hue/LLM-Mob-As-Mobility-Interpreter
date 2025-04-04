# ***L***arge ***L***anguage ***M***odels for Human ***Mob***ility Prediction (LLM-Mob)

Starting paper ***[Where Would I Go Next? Large Language Models as Human Mobility Predictors](https://arxiv.org/abs/2308.15197)***.

## My Changes
### 1. Implementing llama3.1 instead of GPT (openAI) Models ( They require payments )
You don't need any OpenAI API KEY but you need to install a local llama model:
```bash
brew install ollama
ollama pull llama3
ollama serve
```

## How To Run It
### 1. Install The Requirements in a Python Enviroment ( myenv )
```bash
python3 -m venv myenv
source myenv/bin/activate
pip install -r requirements.txt
```
### 2. From Terminal To Start the service of your local Model
```bash
OLLAMA_HOST=127.0.0.1:11434 ollama serve
```
### 3. Run the scripts to start the prediction process.
Run `llm-mob.py`, change the parameters in the main function if necessary and start the prediction process by simply running the sripts
```bash
python llm-mob.py
```
The log file will be stored in `/logs` and prediction results will be stored in `/output`.

## Results and evaluation
We provide the actual prediction results obtained in our experiments in `/results`. 
To calculate the evaluation metrics, check the IPython notebook `metrics.ipynb` and run the scripts therein.


## Citation

```bibtex
@article{mattioli2025Thesis,
  title={Large Language Models for Human Mobility Prediction (LLM-Mob)},
  author={Mattioli Simone},
  year={2025}
}
```
```bibtex
@article{wang2023would,
  title={Where would i go next? large language models as human mobility predictors},
  author={Wang, Xinglei and Fang, Meng and Zeng, Zichao and Cheng, Tao},
  journal={arXiv preprint arXiv:2308.15197},
  year={2023}
}
```