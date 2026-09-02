#!/bin/bash
# One-shot init for the fresh fix251t pod:
# 1) pip install -e afd-plugin (compiles CAM aclnn ops + torch extension)
# 2) lm_eval stack per _install_lmeval.sh
# 3) verification imports
# CAM vendor lib was already restored from NAS backup before this runs.
set -x
cd /a3_inference/itask/workdir/tq02357756/shwstone/code/afd-plugin
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export SOC_VERSION=910c

pip install -e . -v --no-build-isolation 2>&1 | tail -5
python3 -c "import afd_plugin; print('AFD_PLUGIN_OK', afd_plugin.__file__)"

pip install lm-eval tenacity -i https://mirrors.aliyun.com/pypi/simple/ 2>&1 | tail -2
pip install 'pyarrow==16.1.0' 'datasets==3.5.0' -i https://mirrors.aliyun.com/pypi/simple/ 2>&1 | tail -2
pip install --upgrade fastapi starlette -i https://mirrors.aliyun.com/pypi/simple/ 2>&1 | tail -2

python3 -c "import lm_eval; print('LM_EVAL_OK', lm_eval.__version__)"
python3 -c "import fastapi, starlette; print('FASTAPI', fastapi.__version__, 'STARLETTE', starlette.__version__)"
python3 -c "import pyarrow, datasets; print('PYARROW', pyarrow.__version__, 'DATASETS', datasets.__version__)"
python3 -c "import torch, torch_npu; import umdk_cam_op_lib; print('UMDK_OK')"
echo INIT_ALL_DONE
