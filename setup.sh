set -e

PROXY_ARGS=""
VENV_NAME="./venv"

python3.9 -m venv $VENV_NAME
source $VENV_NAME/bin/activate
which python3
python3 -m pip install $PROXY_ARGS --upgrade pip==21
python3 -m pip install $PROXY_ARGS --upgrade setuptools==65.5.0
python3 -m pip install $PROXY_ARGS --upgrade wheel==0.38.4

python3 -m pip install $PROXY_ARGS -r requirements.txt
python3 -m pip install -e .

echo '-----------------------------------------'
echo '         Great Success!'
echo '-----------------------------------------'

