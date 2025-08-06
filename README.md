# FAIRXAI
## Modular Explainability for AI in Research


*FAIRXAI* is a platform developed within the Future Artificial Intelligence in Research (FAIR) initiative, designed to support the composition, execution, and explanation of modular AI decision-making processes.

Unlike traditional explainable AI tools that focus on individual models, FAIRXAI enables researchers and developers to describe and visualize the entire decision pipeline as a structured composition of interoperable modules. Each module contributes to the overall reasoning process and is associated with its own explanation method.

The platform empowers users to build transparent, traceable AI workflows that are aligned with the needs of scientific research, regulatory clarity, and human interpretability. FAIRXAI is the toolbox for making the next generation of AI systems not only powerful — but understandable.

## Documntation

## Documentation

The documentation is based on Sphinx. Documentation of the code is created by simply writing docstrings using reStructuredText markup. Docstrings are comments placed within triple quotes (''' or """) immediately below module, class, function, or method definitions.

The creation of online documentation the features of Sphinx. 
To build the documentation:  

```bash

cd docs
make html

```
Once the documentation is built, the new folder `docs/html` must be committed and pushed to the repository and the documentation is then available here: https://kdd-lab.github.io/FAIRXAI/html/index.html

To update the online documentation, as an instance when new modules or function are added to the LORE_sa library, it is necessary to delete the old folder `docs/html`, build the documentation (see the snippet above)  and copy the greshly created `docs/_build/html` folder into `docs/`. Then, after committing and pushing the folder `docs/html`, the online documentation is updated to the last version.
