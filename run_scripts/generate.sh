gpu_id=0
seed=2020

for temperature in 1; do
  for topk in 5 10; do
    for prevent_too_short in 0 230; do
      for prevent_repeat_ngram in 0 2; do
        OMP_NUM_THREADS=8 \
        NUMEXPR_MAX_THREADS=128 \
        CUDA_VISIBLE_DEVICES=${gpu_id} \
          python main.py with task_generate_both2text \
          seed=${seed} \
          topk=${topk} \
          prevent_too_short=${prevent_too_short} \
          prevent_repeat_ngram=${prevent_repeat_ngram} \
          temperature=${temperature}
        done
      done
    done
  done