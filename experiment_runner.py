from sklearn.decomposition import PCA
from inspect_ai import eval, Epochs
from inspect_ai.log import read_eval_log, list_eval_logs
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd

# Import tasks
from tasks.task_adaptive_legal import (
    legalbench_initial,
    legalbench_initial_aggregated,
    adaptive_legal,
)
from tasks.task_adaptive_truthfulqa import (
    truthfulqa_initial,
    adaptive_truthfulqa,
)
from tasks.task_forecasting import forecasting
from tasks.pair_inspect import (
    pair_task,
    pair_task_adaptive,
    jb_behaviors_dataset,
)

if __name__ == "__main__":
    ###################### LegalBench Experiments ######################
    model_list_generator = [
        "openai/gpt-4o",
        "together/mistralai/Mixtral-8x22B-Instruct-v0.1",
        "anthropic/claude-3-5-sonnet",
    ]
    model_list_eval = [
        "openai/gpt-4o",
        "together/mistralai/Mixtral-8x22B-Instruct-v0.1",
        "anthropic/claude-3-5-sonnet",
    ]

    # Specify LegalBench tasks to evaluate
    task_names = [
        'maud_ability_to_consummate_concept_is_subject_to_mae_carveouts',
        'maud_financial_point_of_view_is_the_sole_consideration',
        'maud_accuracy_of_fundamental_target_rws_bringdown_standard',
        'maud_accuracy_of_target_general_rw_bringdown_timing_answer',
    ]

    # Run initial LegalBench aggregated task
    for eval_model in model_list_eval:
        task = legalbench_initial_aggregated(task_names=task_names)
        log_dir = f"logs/initial_legalbench_{eval_model.replace('/', '_')}"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        # Check if initial log exists
        json_files = [f for f in os.listdir(log_dir) if f.endswith('.json')]
        if json_files:
            initial_log_path = os.path.join(log_dir, max(
                json_files,
                key=lambda x: os.path.getctime(os.path.join(log_dir, x))
            ))
            print(f"Initial LegalBench log exists at {initial_log_path}")
            initial_log = read_eval_log(initial_log_path)
            if initial_log.status != "success":
                print("Previous log failed, re-running evaluation")
                task_log = eval(
                    task,
                    epochs=Epochs(1, "max"),
                    max_connections=50,
                    log_dir=log_dir,
                    model=eval_model,
                    log_level="error"
                )[0]
                initial_log_path = os.path.join(log_dir, max(
                    [f for f in os.listdir(log_dir) if f.endswith('.json')],
                    key=lambda x: os.path.getctime(os.path.join(log_dir, x))
                ))
        else:
            task_log = eval(
                task,
                epochs=Epochs(1, "max"),
                max_connections=50,
                log_dir=log_dir,
                model=eval_model,
                log_level="error"
            )[0]
            initial_log_path = os.path.join(log_dir, max(
                [f for f in os.listdir(log_dir) if f.endswith('.json')],
                key=lambda x: os.path.getctime(os.path.join(log_dir, x))
            ))
            if task_log.status == "success":
                print(f"Initial LegalBench task for {eval_model} completed successfully.")
            else:
                print(f"Initial LegalBench task for {eval_model} failed.")
                continue

        # Run Adaptive LegalBench tasks
        for positive_samples in [1]:
            for negative_samples in [4, 8, 16, 32, 64]:
                for generator_model in model_list_generator:
                    log_dir = f"logs/adaptive_legal_{eval_model.replace('/', '_')}"
                    task = adaptive_legal(
                        initial_log_path=initial_log_path,
                        task_name="maud_specific_performance",
                        n_positive_samples=positive_samples,
                        n_negative_samples=negative_samples,
                        generator_model_name=generator_model,
                        eval_model_name=eval_model,
                        use_cot_generator=True,
                        use_cot_evaluator=False,
                        randomize_sampling=False,
                        judge_model_name="openai/o1-preview",
                    )
                    eval(
                        task,
                        epochs=Epochs(30, "mean"),
                        max_connections=1000,
                        log_dir=log_dir,
                        model=eval_model,
                        temperature=0,
                    )

    ###################### TruthfulQA Experiments ######################
    model_list_generator = [
        "together/meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
        "together/mistralai/Mixtral-8x22B-Instruct-v0.1",
        "anthropic/claude-3-5-sonnet",
    ]
    model_list_eval = [
        "openai/gpt-4o-mini",
        "together/mistralai/Mixtral-8x22B-Instruct-v0.1",
        "anthropic/claude-3-5-sonnet",
    ]

    for eval_model in model_list_eval:
        # Run initial TruthfulQA task
        task = truthfulqa_initial(target="mc1")
        log_dir = f"logs/initial_truthfulqa_{eval_model.replace('/', '_')}"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        json_files = [f for f in os.listdir(log_dir) if f.endswith('.json')]
        if json_files:
            initial_log_path = os.path.join(log_dir, max(
                json_files,
                key=lambda x: os.path.getctime(os.path.join(log_dir, x))
            ))
            print(f"Initial TruthfulQA log exists at {initial_log_path}")
            initial_log = read_eval_log(initial_log_path)
            if initial_log.status != "success":
                print("Previous log failed, re-running evaluation")
                task_log = eval(
                    task,
                    epochs=Epochs(1, "max"),
                    max_connections=50,
                    log_dir=log_dir,
                    model=eval_model
                )[0]
                initial_log_path = os.path.join(log_dir, max(
                    [f for f in os.listdir(log_dir) if f.endswith('.json')],
                    key=lambda x: os.path.getctime(os.path.join(log_dir, x))
                ))
        else:
            task_log = eval(
                task,
                epochs=Epochs(1, "max"),
                max_connections=50,
                log_dir=log_dir,
                model=eval_model
            )[0]
            initial_log_path = os.path.join(log_dir, max(
                [f for f in os.listdir(log_dir) if f.endswith('.json')],
                key=lambda x: os.path.getctime(os.path.join(log_dir, x))
            ))
            if task_log.status == "success":
                print(f"Initial TruthfulQA task for {eval_model} completed successfully.")
            else:
                print(f"Initial TruthfulQA task for {eval_model} failed.")
                continue

        # Run Adaptive TruthfulQA tasks
        for generator_model in model_list_generator:
            for positive_samples in [1]:
                for negative_samples in [8]:
                    log_dir = f"logs/adaptive_truthfulqa_{eval_model.replace('/', '_')}"
                    task = adaptive_truthfulqa(
                        initial_log_path=initial_log_path,
                        n_positive_samples=positive_samples,
                        n_negative_samples=negative_samples,
                        generator_model_name=generator_model,
                        eval_model_name=eval_model,
                        target="mc1",
                        use_cot=False,
                    )
                    eval(
                        task,
                        epochs=Epochs(30, "mean"),
                        max_connections=50,
                        log_dir=log_dir,
                        model=eval_model,
                        temperature=0
                    )

                    task = adaptive_truthfulqa(
                        initial_log_path=initial_log_path,
                        n_positive_samples=positive_samples,
                        n_negative_samples=negative_samples,
                        generator_model_name=generator_model,
                        eval_model_name=eval_model,
                        target="mc1",
                        use_cot=True,
                    )
                    eval(
                        task,
                        epochs=Epochs(30, "mean"),
                        max_connections=50,
                        log_dir=log_dir,
                        model=eval_model,
                        temperature=0
                    )

    ###################### Forecasting Experiment ######################
    model_list_eval = [
        "openai/gpt-4",
        "together/mistralai/Mixtral-8x22B-Instruct-v0.1",
        "anthropic/claude-3-5-sonnet",
    ]
    
    for eval_model in model_list_eval:
        task = forecasting()
        log_dir = f"logs/forecasting_{eval_model.replace('/', '_')}"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        eval(
            task,
            epochs=Epochs(1, "max"),
            max_connections=50,
            log_dir=log_dir,
            model=eval_model
        )

    ###################### PAIR Experiments ######################
    model_list_target = [
        "together/meta-llama/Llama-2-7b-chat-hf",
        "together/meta-llama/Llama-3-1-turbo",
    ]
    model_list_judge = [
        "openai/gpt-4o-mini",
    ]
    model_list_attack = [
        "together/mistralai/Mixtral-8x22B-Instruct-v0.1",
        "anthropic/claude-3-5-sonnet",
    ]

    for target_model in model_list_target:
        for judge_model in model_list_judge:
            for attack_model in model_list_attack:
                # Run PAIR task
                task = pair_task(
                    target_model_name=target_model,
                    judge_model_name=judge_model,
                    attack_model_name=attack_model,
                    max_iterations=3,
                    n_last_messages=2,
                    epochs=2,
                    use_strongreject_scorer=True,
                    heirarchal_scorer=False,
                    dataset=jb_behaviors_dataset,
                )
                log_dir = f"logs/pair_task_{target_model.replace('/', '_')}_{judge_model.replace('/', '_')}_{attack_model.replace('/', '_')}"
                if not os.path.exists(log_dir):
                    os.makedirs(log_dir)
                eval(
                    task,
                    epochs=Epochs(1, "max"),
                    max_connections=50,
                    log_dir=log_dir
                )

                # Run adaptive PAIR tasks
                for percentiles_and_samples in [[(1.0, 2), (0.5, 2)]]:
                    task = pair_task_adaptive(
                        target_model_name=target_model,
                        judge_model_name=judge_model,
                        attack_model_name=attack_model,
                        epochs=10,
                        percentiles_and_samples=percentiles_and_samples
                    )
                    log_dir = f"logs/pair_task_adaptive_{target_model.replace('/', '_')}_{judge_model.replace('/', '_')}_{attack_model.replace('/', '_')}_{percentiles_and_samples}"
                    if not os.path.exists(log_dir):
                        os.makedirs(log_dir)
                    eval(
                        task,
                        epochs=Epochs(1, "max"),
                        max_connections=50,
                        log_dir=log_dir
                    )
