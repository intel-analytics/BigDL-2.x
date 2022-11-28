# Analytics-zoo-doc Documentation

To compile the documentation, run the following commands from this directory.

```
# for reproducing ReadtheDocs deployment environment
pip install --upgrade pip "setuptools<58.3.0"
pip install --upgrade pillow mock==1.0.1 "alabaster>=0.7,<0.8,!=0.7.5" commonmark==0.9.1 recommonmark==0.5.0 sphinx sphinx-rtd-theme "readthedocs-sphinx-ext<2.3"

# for other documentation related dependencies
wget https://raw.githubusercontent.com/analytics-zoo/gha-cicd-env/main/python-requrirements/requirements-zoo-doc.txt
pip install -r requirements-zoo-doc.txt

make html
open _build/html/index.html
```

To test if there are any build errors with the documentation, do the following.

```
sphinx-build -b html -d _build/doctrees source _build/html
```