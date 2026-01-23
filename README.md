# FAIRXAI
## Modular Explainability for AI in Research

*FAIRXAI* is a platform developed within the Future Artificial Intelligence in Research (FAIR) initiative, designed to support the composition, execution, and explanation of modular AI decision-making processes.

Unlike traditional explainable AI tools that focus on individual models, FAIRXAI enables researchers and developers to describe and visualize the entire decision pipeline as a structured composition of interoperable modules. Each module contributes to the overall reasoning process and is associated with its own explanation method.

The platform empowers users to build transparent, traceable AI workflows that are aligned with the needs of scientific research, regulatory clarity, and human interpretability. FAIRXAI is the toolbox for making the next generation of AI systems not only powerful — but understandable.

## How to install

```bash
# 1. Installa il pacchetto e le dipendenze di sviluppo
pip install -e .[all]

# 2. Installa le dipendenze Git separate
pip install -r requirements.txt
```

## Documentation

The documentation is built using Sphinx.
Code documentation is generated automatically from docstrings written in reStructuredText (reST) format.
Docstrings are text blocks enclosed in triple quotes (''' or """) placed immediately below module, class, function, or method definitions.

### Updating the documentation
To build the documentation locally, run:  

```bash
python docs/generate_docs.py
```
This command:

* Generates the .rst sources from the FAIRXAI codebase
* Builds the HTML documentation using Sphinx
* Publishes the generated documentation to the gh-pages branch
* Once the process completes, the online documentation will be automatically available at: https://kdd-lab.github.io/FAIRXAI/

## Support

- 📖 Documentation: [https://kdd-lab.github.io/FAIRXAI/](https://kdd-lab.github.io/FAIRXAI/)
- 🐛 Issue Tracker: https://github.com/kdd-lab/FAIRXAI/issues
- 💬 Discussions: Use GitHub Discussions for questions and discussions