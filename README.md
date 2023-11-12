# Auto-Instruct

This is the repository for Auto-Instruct, an automatic solution of generating and selecting instructions for prompting large language models (LLMs). Our method leverages the inherent generative ability of LLMs to produce diverse candidate instructions for a given task, and then ranks them using a scoring model trained on a variety of 575 existing NLP tasks. In experiments on 118 out-of-domain tasks, Auto-Instruct surpasses both human-written instructions and existing baselines of LLM-generated instructions. For more details, please refer to our paper [*"Auto-Instruct: Automatic Instruction Generation and Ranking for Black-Box Language Models"*](https://arxiv.org/pdf/2310.13127.pdf) in EMNLP 2023 Findings.

<center><img src="figures/pipeline.png" alt="Auto-Instruct Pipeline" width="384" height="384"></center>

The repository includes the following contents:

- `data`: the training / testing data files, meta-prompts, downstream prompts, and generated instructions.
- `GPT-3/optimization`: the source code for data, model training and model evaluation.
    - `instruction_generation_templates`: the templates of creating meta-prompts for each task (used for instruction generation)
    - `instruction_generation`: scripts for instruction generation
    - `instruction_labeling`: scripts for label the instructions for training / testing, as well as dataset pre-processing
    - `run.py`: entrance for model training / testing, see `GPT-3/optimization/README.md`
    - `evaluation`: evaluation scripts for the ranking model

### Environment

```bash
pip install -r requirements.txt
```

### Checkpoints

TBA

### Citation

If you find our work useful, please kindly cite our paper:

```
@inproceedings{Auto-Instruct,
  author = {Zhihan Zhang and
                  Shuohang Wang and
                  Wenhao Yu and
                  Yichong Xu and
                  Dan Iter and
                  Qingkai Zeng and
                  Yang Liu and
                  Chenguang Zhu and
                  Meng Jiang},
  title = {Auto-Instruct: Automatic Instruction Generation and 
                  Ranking for Black-Box Language Models},
  booktitle = {Findings of the 2023 Conference on Empirical 
               Methods in Natural Language Processing, {EMNLP} 2023, 
               Singapore, December 6-10, 2023},
  publisher = {Association for Computational Linguistics},
  year = {2023},
  url = {https://doi.org/10.48550/arXiv.2310.13127}
}
```
