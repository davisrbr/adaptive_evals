from collections import defaultdict
from inspect_ai.log import list_eval_logs, read_eval_log
import os
import numpy as np
import matplotlib.pyplot as plt

# Assuming you've set INSPECT_LOG_DIR in your environment
# log_dir =  "logs/pair_task_22B_log/" # os.environ.get('INSPECT_LOG_DIR', './logs')
log_dirs =  ["logs/pair_task_adaptive_22B_log/", "logs/pair_task_22B_log/", "pair_task_adaptive_embeddings_22B_log"] # os.environ.get('INSPECT_LOG_DIR', './logs')

log_info = defaultdict(list) 
for log_dir in log_dirs:
    # Get all log files
    log_files = list_eval_logs(log_dir)

    # Dictionary to store vectors of scores for each run
    run_scores = {}

    for log_file in log_files:
        # Read the log file
        eval_log = read_eval_log(log_file)
        
        if eval_log.status == "success":
            # Extract task_id (assuming it's unique for each run)
            task_id = eval_log.eval.task_id
            scores = [sample.score.value for sample in eval_log.samples]
            sample_ids = [sample.id for sample in eval_log.samples]
            # Store the vector of scores and sample IDs
            run_scores[task_id] = list(zip(scores, sample_ids))
            # print(run_scores[task_id])

    # Now run_scores contains a vector of scores for each run
    # You can analyze or compare these vectors as needed

    # Example: Print the mean score for each run
    for task_id, data in run_scores.items():
        max_data = max(data, key=lambda x: x[0])
        # print(f"Max score = {int(max_data[0])}, Sample ID = {max_data[1]}")
        log_info[log_dir].append((int(max_data[0]), max_data[1]))

# sort log_info by the sample id
for log_dir in log_dirs:
    log_info[log_dir].sort(key=lambda x: x[1])
    scores = [x[0] for x in log_info[log_dir]]
    sample_ids = [x[1] for x in log_info[log_dir]]
    log_info[log_dir] = np.array(scores)

log_info_arrays = {l.strip("logs/"): scores for l, scores in log_info.items()}

fig, axes = plt.subplots(len(log_info_arrays), 1, figsize=(10, 2*len(log_info_arrays)), sharex=True)
fig.suptitle('Scores Comparison Across Different Logs')

for i, (log_dir, scores) in enumerate(log_info_arrays.items()):
    scores = np.where(scores >= 7, 1, 0).reshape(10, 3)
    im = axes[i].imshow(scores, aspect='auto', origin='lower', cmap='viridis_r')
    axes[i].set_title(log_dir[:-1])
    fig.colorbar(im, ax=axes[i], orientation='vertical', pad=0.01)

plt.tight_layout()
plt.show()