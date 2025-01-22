from setuptools import setup, find_packages


def read_requirements(filename):
    with open(filename, 'r') as file:
        return file.read().splitlines()


setup(
    name="neuropsych",
    version="0.0.0",
    author="NeuroLIAA",
    description="A Python package for handling neuro psych data",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/NeuroLIAA/analisis-neuropruebas",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
    install_requires=read_requirements('requirements.txt')
)
