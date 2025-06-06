import json
import matplotlib.pyplot as plt
import numpy as np

# Load metrics
with open('DiscordBot/evaluation_metrics/llm_metrics.json') as f:
    llm_metrics = json.load(f)
with open('DiscordBot/evaluation_metrics/tuned_llm_metrics.json') as f:
    tuned_llm_metrics = json.load(f)

# Extract per-class metrics
classes = ['0', '1', '2', '3', '4']
metrics = ['precision', 'recall', 'f1-score']

llm_scores = {m: [llm_metrics['classification_report'][c][m] for c in classes] for m in metrics}
tuned_scores = {m: [tuned_llm_metrics['classification_report'][c][m] for c in classes] for m in metrics}

x = np.arange(len(classes))
width = 0.35

for metric in metrics:
    plt.figure(figsize=(8, 5))
    plt.bar(x - width/2, llm_scores[metric], width, label='LLM')
    plt.bar(x + width/2, tuned_scores[metric], width, label='Tuned LLM')
    plt.xticks(x, classes)
    plt.xlabel('Hate Class')
    plt.ylabel(metric.capitalize())
    plt.title(f'{metric.capitalize()} by Class: LLM vs Tuned LLM')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'DiscordBot/evaluation_metrics/{metric}_by_class_llm_vs_tunedllm.png')
    plt.show() 