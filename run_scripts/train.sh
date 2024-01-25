gpu_id=1
seed=2020

OMP_NUM_THREADS=8 \
NUMEXPR_MAX_THREADS=128 \
CUDA_VISIBLE_DEVICES=${gpu_id} \
  python main.py with task_train_text2text \
  seed=${seed} \
  debug=True
