from sklearn.decomposition import PCA
from eval_dump_inspect import mmlu, truthfulqa
from inspect_ai import eval
from csv import reader
import json
from collections import defaultdict
from inspect_ai.log import list_eval_logs, read_eval_log
import os
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd


if __name__ == "__main__":
    # task = mmlu()
    # # task = truthfulqa()

    # # first, grab the list of models and their families
    # models_list = reader(open("/Users/davisbrown/inspect_attacks/model_observation_list.csv"))
    # model_families = {}
    # for line in models_list:
    #     model_name, model_family = line
    #     model_families.setdefault(model_family, []).append(model_name)

    # # load the model mapping between hugging face and together ai
    # json_file_path = 'model_mapping.json'
    # with open(json_file_path, 'r') as file:
    #     model_mapping = json.load(file)

    # # Evaluate the models
    # for family, models in model_families.items():
    #     print(f"Evaluating models in the {family} family:")
    #     for model in models:
    #         if model in model_mapping and "2-7b" not in model:
    #             model_to_use = f"together/{model_mapping[model]}"
    #         else:
    #             print("Passing on model: ", model)
    #             continue

    #         print(f"Evaluating model: {model_to_use}")
    #         eval(task, max_connections=10000, log_dir="mmlu_log", model=model_to_use)[0]
    #         # eval(task, max_connections=10000, log_dir="truthful_log", model=model_to_use)[0]



    ###################### Process and visualize raw results ######################
    # log_dirs = ["truthful_log"]
    # log_info = {}
    # question_data = {}

    # for log_dir in log_dirs:
    #     # Get all log files
    #     log_files = list_eval_logs(log_dir)

    #     # Dictionary to store vectors of scores for each run
    #     run_scores = {}

    #     for log_file in log_files:
    #         # Read the log file
    #         eval_log = read_eval_log(log_file)
            
    #         if eval_log.status == "success":
    #             for sample in eval_log.samples:
    #                 sample_id = sample.id
    #                 score = 1 if sample.score.value == "C" else 0
    #                 question = sample.input
    #                 answer = sample.score.answer
    #                 explanation = sample.score.explanation
    #                 choices = sample.choices
                    
    #                 # Store the score, question text, answer, and explanation
    #                 run_scores[sample_id] = score
    #                 question_data[sample_id] = {
    #                     "question": question,
    #                     "answer": answer,
    #                     "explanation": explanation,
    #                     "choices": choices
    #                 }

    #             # Convert run_scores to a list of scores, sorted by sample_id
    #             sorted_scores = [run_scores[sample_id] for sample_id in sorted(run_scores.keys())]
    #             log_info[eval_log.samples[0].output.model] = np.array(sorted_scores)

    # # Create a single heatmap with all models stacked
    # fig, ax = plt.subplots(figsize=(12, 8))
    # fig.suptitle('TruthfulQA Scores Comparison')

    # # Prepare data for stacked heatmap
    # stacked_scores = []
    # model_names = []

    # for model, scores in log_info.items():
    #     stacked_scores.append(scores)
    #     model_names.append(model)

    # # Convert to numpy array and transpose
    # stacked_scores = np.array(stacked_scores)

    # # Create the heatmap
    # im = ax.imshow(stacked_scores, aspect='auto', origin='lower', cmap='viridis_r')

    # # Set y-axis labels (model names)
    # ax.set_yticks(np.arange(len(model_names)))
    # ax.set_yticklabels(model_names)

    # # Set x-axis label
    # ax.set_xlabel('Question Index')

    # # Add colorbar
    # cbar = fig.colorbar(im, ax=ax, orientation='vertical', pad=0.01)
    # cbar.set_label('Score (0: Incorrect, 1: Correct)')

    # # Adjust layout and save
    # plt.tight_layout()
    # plt.savefig('truthfulqa_heatmap.png', dpi=300, bbox_inches='tight')
    # plt.close()

    # # Save the question data and scores
    # np.save('truthfulqa_scores.npy', dict(log_info))
    # np.save('truthfulqa_questions.npy', question_data)

    # print("Heatmap saved as 'truthfulqa_heatmap.png'")
    # print("Scores saved as 'truthfulqa_scores.npy'")
    # print("Questions saved as 'truthfulqa_questions.npy'")
    ###############################################################


    ###################### Decompose results ######################

    num_pcs = 4
    scores = np.load('truthfulqa_scores.npy', allow_pickle=True).item()
    questions = np.load('truthfulqa_questions.npy', allow_pickle=True).item()

    # Access scores for mc1
    mc1_scores_stacked = -1* (np.stack([v for _, v in scores.items()]) - 1)
    # pca of mc1 scores, which is a matrix of size (num_questions, num_models)
    # the pca will give us a matrix of size (num_questions, num_principal_components)
    pca = PCA(n_components=num_pcs)
    pca.fit_transform(mc1_scores_stacked)
    pca_components = pca.components_
    print(pca_components.shape)
    print(pca.explained_variance_ratio_)
    # plot the explained variance ratio as a bar chart 
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(pca.explained_variance_ratio_)+1), pca.explained_variance_ratio_)
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Explained Variance Ratio by Principal Component')
    plt.savefig('truthfulqa_pca_explained_variance_ratio.png')

    per_question_accuracy = np.mean(mc1_scores_stacked, axis=0)


    # Create a list of questions for the first 50 items
    questions_list = [questions[i]['question'] for i in range(1, len(questions)+1)]
    choices_list = ["<br>".join(questions[i]['choices']) for i in range(1, len(questions)+1)]

    # Create a DataFrame for Plotly
    df = pd.DataFrame(pca_components[:, :len(questions)].T, columns=[f'PC{i+1}' for i in range(num_pcs)])
    df['question'] = questions_list
    df['choices'] = choices_list
    df['group_accuracy'] = per_question_accuracy
    # Sort the DataFrame by PC1 values
    df_sorted = df.sort_values('group_accuracy', ascending=True)

    # Create the heatmap using Plotly Express with sorted data
    fig = px.imshow(df_sorted.iloc[:, :num_pcs].T,  # Transpose to get PCs as rows
                    labels=dict(x="Question Index", y="Principal Component", color="Value"),
                    x=[f"Q{i+1}" for i in range(1, len(questions)+1)],
                    y=[f'PC{i+1}' for i in range(num_pcs)],
                    aspect="auto",
                    color_continuous_scale="viridis_r")

    # Create a 2D list of questions and choices to match the shape of the heatmap
    combined_texts = [f"{q}<br>{c}<br>acc.: {(1-a)*100:.2f}%" for i, (q, c, a) in enumerate(zip(df_sorted['question'], df_sorted['choices'], df_sorted['group_accuracy']))]
    customdata = [combined_texts for _ in range(num_pcs)]

    # Update hover template to show the question text and choices
    fig.update_traces(hovertemplate="Question Index: %{x}<br>Principal Component: %{y}<br>Value: %{z:.4f}<br>Question and Choices: %{customdata}",
                      customdata=customdata)

    # Update layout
    fig.update_layout(title="PCA Components of TruthfulQA Scores (Sorted by Aggregate Question Accuracy)",
                      xaxis_title="Question Index (Sorted by Aggregate Question Accuracy)",
                      yaxis_title="Principal Component")

    # Show the plot
    fig.show()

    # Optionally, save the plot as an interactive HTML file
    fig.write_html("truthfulqa_pca_heatmap_sorted.html")

    # # Access a specific question's data by its sample ID
    # sample_id = 'some_sample_id'
    # question_info = questions[sample_id]
    # question_text = question_info['question']
    # answer = question_info['answer']
    # explanation = question_info['explanation']
