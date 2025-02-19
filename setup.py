from setuptools import setup, find_packages

setup(
    name="rl2023",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'numpy',
        'matplotlib',
        'gym==0.25.2',
        'torch',
        'stable-baselines3==2.1.0',
        'pytest'
    ]
) 