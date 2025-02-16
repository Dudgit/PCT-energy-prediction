# Energy Prediction for Proton Computed Tomography
This is a repository made by the Budapest team of the PcT collaboration.  
## configs
This folder contains all the yaml files for the project configurations. It's important to note that the project is structured in a way, that it will run with multiple configurations.
-  In the const.yaml I store data configurations that I usually only change by hand if I overwrite something in the code base.
- In the multivar.yaml I store those configs that I want to run a grid search on. It is fully automatic.

## utils
This folder could have been named to src, it contains helping functions and definitions.  
- filter.py contains some filtering method for the data preprocessing
- model.py contains the model definition with almost all of it's attributes.
- modelfuncs.py contains some external functions called by the model. It might be inculded in the model later
- readdata.py is a legacy python file. It was created to convert Gate output into the output I want to work with.

## eda.ipynb
A simple notebook where I try out some functions. Also it can be used to visualize some results like one would in exploratory data analysis.

## main.py
This is the python file I call to run everything. This handles the different configs.

## vanilla.py
This is a completely separate notebook. Here we tried to fit the data with traditional machine learning algorithms.

## tensorboard_command.txt
Just the command to run to be able to check tensorboard in real time. For this to work one would have to forward a port too.

## Missing folders.
The runs and the data folders are not uploeaded to github.  
- runs contains all the loggings done by tensorboard.
- data contains all the data used for training/evaluation.