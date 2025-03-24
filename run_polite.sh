#!/bin/bash

# Models to use
EVAL_MODELS=(
    "openai/gpt-4o-mini"
)

GENERATOR_MODELS=(
    "openai/gpt-4o-mini"
)

# Datapoint values to sweep over
DATAPOINTS=(5 10 20 40 80 160)

# Base configuration
SIMILARITY_THRESHOLD=0.4
MAX_ATTEMPTS=10

# Loop over datapoints
for n_points in "${DATAPOINTS[@]}"; do
    echo "Running sweep with $n_points datapoints..."

    # Build the eval models and generator models arguments
    EVAL_ARGS=""
    for model in "${EVAL_MODELS[@]}"; do
        EVAL_ARGS="$EVAL_ARGS --eval-models $model"
    done

    GEN_ARGS=""
    for model in "${GENERATOR_MODELS[@]}"; do
        GEN_ARGS="$GEN_ARGS --generator-models $model"
    done

    # Run the experiment
    python experiment_runner_politeness.py \
        $EVAL_ARGS \
        $GEN_ARGS \
        --n-datapoints "$n_points" \
        --similarity-threshold "$SIMILARITY_THRESHOLD" \
        --max-attempts "$MAX_ATTEMPTS" \
        --use-embeddings

done

echo "Sweep completed!"