# Evaluation

1. If labels of all instructions on the test set are pre-calculated before model training, you can directly call `niv2_calc_rouge.py` to compute the performance of random selection, model selection, etc. In this case, the random selection score would be the mean score of all instructions.
```bash
python GPT-3/optimization/evaluation/niv2_calc_rouge.py \
    -prediction YOUR_SAVE_DIR/91test_predictions.json \
    -evaluation rouge
```

2. If labels are not pre-calculated, we need to call the OpenAI model to test the downstream performance of the selected instructions in `evaluate_selected_instruction.py`. In this case, the random selection score would be a real `random selection`, which means first randomly selecting an instruction and then perform downstream inference. The script is originally intended for testing ChatGPT, so you may need slight modifications if using `text-davinci-003`.
```bash
python GPT-3/optimization/evaluation/evaluate_selected_instruction.py \
    -prediction YOUR_SAVE_DIR/91test_predictions.json \
    -test_data YOUR_DATA_DIR/unseen_test.jsonl \
    -save_interval 20 \
    -output_file YOUR_SAVE_DIR/chatgpt_inference_results.json \
    -model gpt-3.5-turbo \
    -max_tokens 80 \
    -timeout 40 \
    -sleep 2 \
    -api_key YOUR_OPENAI_KEY

# Useful arguments:
# -zero_shot for evaluation with zero-shot prompting (usually also need to specify -append_format to avoid free-form generation)
# -random_baseilne: randomly select an instruction to evaluate
# -naive_baseline: select the human-written seed instruction to evaluate
```
