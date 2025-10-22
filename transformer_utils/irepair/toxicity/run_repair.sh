#!/bin/bash

# RQ3
#SAMPLE_SEED=(43 44 46 48 101)
#PERCENTAGES=(10 20 30 40 60 70 80 90)
##PERCENTAGES=(50)
##SAMPLE_METHOD=kcenter
#for SEED in "${SAMPLE_SEED[@]}"
#do
#    for PCT in "${PERCENTAGES[@]}"
#    do
#        echo "===== Training with sample_seed=$SEED, percentage=$PCT% ====="
#
#        # time
#        python train_dpo/train.py sample_seed=$SEED sample_percentage=$PCT
#
#        # mode 0: toxicity, mode 1: lambada perplexity
#        for MODE in 0 1
#        do
#            echo ">> Evaluation with mode=$MODE, percentage=$PCT%"
#            python eval_interventions/run_evaluations.py --mode $MODE --sample_seed $SEED --sample_percentage $PCT
#        done
#
#        echo "===== Done sample_seed=$SEED, percentage=$PCT%====="
#    done
#done

# RQ4
SAMPLE_SEED=(43 44 46 48 101)
#PERCENTAGES=(10 20 30 40 60 70 80 90)
#PERCENTAGES=(50)
#SAMPLE_ALPHA=(0.0 0.25 0.5 0.75)
SAMPLE_ALPHA=(1.0)
#SAMPLE_METHOD=kcenter
for SEED in "${SAMPLE_SEED[@]}"
do
    for AL in "${SAMPLE_ALPHA[@]}"
    do
        echo "===== Training with sample_seed=$SEED, percentage=$AL% ====="

        # time
        python train_dpo/train.py sample_seed=$SEED sample_alpha=$AL

        # mode 0: toxicity, mode 1: lambada perplexity
        for MODE in 0 1
        do
            echo ">> Evaluation with mode=$MODE, percentage=$AL%"
            python eval_interventions/run_evaluations.py --mode $MODE --sample_seed $SEED --sample_alpha $AL --sample_percentage 50
        done

        echo "===== Done sample_seed=$SEED, percentage=$AL%====="
    done
done

