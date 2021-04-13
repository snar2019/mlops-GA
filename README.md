## This is Revenue prediction project for e-commerce client

### Codes to start with the project:

create env

conda create -n MLOPS python=3.7 -y
activate env

snar2019conda activate MLOPS
created a req file

install the req

pip install -r requirements.txt
download the data from

#https://drive.google.com/drive/folders/18zqQiCJVgF7uzXgfbIJ-04zgz1ItNfF5?usp=sharing

git init
dvc init 
dvc add data_given/train.csv
git add .
git commit -m "first commit"
#oneliner updates for readme

git add . && git commit -m "update Readme.md"
git remote add origin https://github.com/snar2019/mlops-GA.git

git push origin main
tox command -

tox
for rebuilding -

tox -r 
pytest command

pytest -v
setup commands -

pip install -e . 
build your own package commands-

python setup.py sdist bdist_wheel

create an artifcats folder

mlflow server command -

mlflow server \
--backend-store-uri sqlite:///mlflow.db \
--default-artifact-root ./artifacts \
--host 0.0.0.0 -p 1234
