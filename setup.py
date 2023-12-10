from setuptools import setup, find_packages

setup(
    name='autocoder',
    version='0.1.0',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'google.generativeai==0.2.2',
        'weaviate-client==4.3b0',
        'typer==0.9.0',
        'beautifulsoup4==4.12.2',
        'python-dotenv==1.0.0',
        'tiktoken==0.5.1',
        'tenacity==8.2.3',
        'langchain==0.0.339',
        'kor==1.0.0',
        'google-cloud-aiplatform==1.36.4',
        'google-auth==0.4.8'
    ],
    entry_points='''
        [console_scripts]
        devbot=main:app
    ''',
)