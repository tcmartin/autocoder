# setup.py
from setuptools import setup, find_packages

setup(
    name='devbot',
    version='0.1.0',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        # List your dependencies here
        # e.g., 'requests', 'typer'
    ],
    entry_points='''
        [console_scripts]
        devbot=main:app
    ''',
)