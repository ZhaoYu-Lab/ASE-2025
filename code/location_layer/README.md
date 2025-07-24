## Instructions for locating the potential best layer
This is a sample code to locate the potential best layer of the CodeBERT model on the technical debt classification task.

The core file is Evidence.py, which contains the calculation process.

The way to use it is to directly execute setcode to get the maximum evidence value of the corresponding layer.

In the run.py file, modify the number of layers loaded by the model to calculate the maximum evidence score layer by layer. 

The layer with the largest score is the potential best layer.

