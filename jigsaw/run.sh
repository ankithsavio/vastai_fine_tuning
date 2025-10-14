pip install uv

cd base_train
uv sync
source .venv/bin/activate
cd ..
uv run --active train_bge.py
deactivate

cd awq_train
uv sync
source .venv/bin/activate
cd ..
uv run --active train_deberta.py
uv run --active train.py
uv run --active train_stack_bge.py
uv run --active train_stack_deberta.py
uv run --active train_stack_all.py

