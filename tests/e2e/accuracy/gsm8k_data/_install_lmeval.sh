#!/bin/bash
# Install lm-eval stack on the new pod (per memory lmeval-gsm8k-run) and
# restore fastapi/starlette so the vllm API server keeps working.
set -x
pip install lm-eval tenacity -i https://mirrors.aliyun.com/pypi/simple/ 2>&1 | tail -2
pip install 'pyarrow==16.1.0' 'datasets==3.5.0' -i https://mirrors.aliyun.com/pypi/simple/ 2>&1 | tail -2
# lm-eval pulls incompatible fastapi/starlette; restore vllm-compatible ones
pip show vllm | grep -i requires
pip install --upgrade fastapi starlette -i https://mirrors.aliyun.com/pypi/simple/ 2>&1 | tail -2
python3 -c "import lm_eval; print('LM_EVAL_OK', lm_eval.__version__)"
python3 -c "import fastapi, starlette; print('FASTAPI', fastapi.__version__, 'STARLETTE', starlette.__version__)"
python3 -c "import pyarrow, datasets; print('PYARROW', pyarrow.__version__, 'DATASETS', datasets.__version__)"
