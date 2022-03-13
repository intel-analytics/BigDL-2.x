# install conda env at client mode

BASE_DIR=/opt/spark/work-dir
PYTHON_VERSION=${PYTHON_VERSION}
CONDA_ENV_NAME=${CONDA_ENV_NAME}
MINICONDA=${MINICONDA}
REQUIREMENTS=${BASE_DIR}/requirements.txt
CURL_TOKEN=${CURL_TOKEN}
PYTHON_DEPS_PATH=${BASE_DIR}/${CONDA_ENV_NAME}".tar.gz"
UPLOAD_URL=${UPLOAD_URL}

export http_proxy=${HTTP_RPOXY}
export https_proxy=${HTTPS_PROXY}

# download conda
cd $BASE_DIR
wget https://repo.anaconda.com/miniconda/${MINICONDA}

# install python packages
/bin/bash ${MINICONDA}
conda create -n ${CONDA_ENV_NAME} python=${PYTHON_VERSION}
source activate ${CONDA_ENV_NAME}
pip install --no-cache-dir --no-deps -r ${REQUIREMENTS}
conda pack -f -o ${CONDA_ENV_NAME}.tar.gz

# upload to http server
curl -v --user ${CURL_TOKEN} --upload-file ${PYTHON_DEPS_PATH} ${UPLOAD_URL}
