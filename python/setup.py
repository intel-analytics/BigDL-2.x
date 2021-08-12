#!/usr/bin/env python

import os
import fnmatch
from setuptools import setup, find_packages

exclude_patterns = ["*__pycache__*", "lightning_logs"]
orca_lite_home = os.path.abspath(__file__ + "/../")


def get_orca_lite_packages():
    orca_lite_packages = []
    for dirpath, dirs, files in os.walk(orca_lite_home + "/orca_lite"):
        package = dirpath.split(orca_lite_home + "/")[1].replace('/', '.')
        if any(fnmatch.fnmatchcase(package, pat=pattern)
                for pattern in exclude_patterns):
            print("excluding", package)
        else:
            orca_lite_packages.append(package)
            print("including", package)
    return orca_lite_packages


def load_requirements(file_name="requirements.txt", comment_char='#'):
    """
    Load requirements from a file,
    return install_requires_list
    """
    with open(file_name, 'r') as file:
        lines = [ln.strip() for ln in file.readlines()]
    _install_requires_list = []
    _dependency_links_tmp = ""
    for ln in lines:
        # Filter out urls
        if comment_char in ln:
            text_start = 0
            for s_char in ln:
                if s_char is comment_char:
                    text_start += 1
            ln = ln[text_start:].strip()
            if ln.startswith('http'):
                _dependency_links_tmp = ln
            else:
                _dependency_links_tmp = ""
        elif ln:
            ln = ln + " @ " + _dependency_links_tmp if _dependency_links_tmp else ln
            _dependency_links_tmp = ""
            _install_requires_list.append(ln)
        else:
            _dependency_links_tmp = ""
    return _install_requires_list


def setup_package():
    install_requires_list = load_requirements()

    metadata = dict(
        name='orca-lite',
        version='0.0.1.dev0',
        description='',
        author='',
        author_email='',
        url='https://github.com/analytics-zoo/orca-lite-poc',
        install_requires=install_requires_list,
        packages=get_orca_lite_packages(),
        scripts=['script/orca-lite-run'],
            entry_points={
                'console_scripts': ['orca-lite-init=orca_lite.common.init_orca_lite:main'],
        }
    )

    setup(**metadata)


if __name__ == '__main__':
    setup_package()
