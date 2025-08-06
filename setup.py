from setuptools import setup, find_packages

setup(
    name="FAIRXAI",
    version="0.1",
    author="Paolo Cintia",
    description="FAIRXAI is a platform developed within the Future Artificial Intelligence in Research (FAIR) initiative, designed to support the composition, execution, and explanation of modular AI decision-making processes.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/kdd-lab/FAIRXAI",
    packages=find_packages(),
    install_requires=[
        # List your dependencies here
        # 'numpy>=1.18.0',
        # 'pandas>=1.0.0',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',  # or whatever your minimum Python version is
)