from setuptools import setup, find_packages

version = "0.1"
setup(
    name="fairxai",
    version=version,
    author="Kode srl",
    description="fairxai is a platform developed within the Future Artificial Intelligence in Research (FAIR) initiative, designed to support the composition, execution, and explanation of modular AI decision-making processes.",
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
    python_requires='>=3.9',
    entry_points={
        'console_scripts': [
            'fairxai=fairxai.main:main',
        ],
    },
)
