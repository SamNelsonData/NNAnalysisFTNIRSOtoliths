# NN Model for Calculating Fish Age  
This repository contains the script and model parameters for estimating the age of Chilipepper rockfish using FTNIRS scans of fish otoliths and metadata.  

# How to Use this Script

## Step 1 - Install Python
This code was built on Python 3.13.3 on MacOS; 3.13.3 is the preferred python version but I expect some others will work. If you use Windows you may have to install Python 3.12.x as PyTorch may not support later versions of Python on Windows. The download can be found [here](https://www.python.org/downloads/).  

## Step 2 - Installing Packages 
I recommend using python virtual environments (a tutorial can be found [here](https://python.land/virtual-environments/virtualenv)). If you're not sure what 'python virtual environments' are and don't have the time to learn skip this (it's not that important).  
Now we will install the necessary packages.

### MacOS/Linux
Run the following command in terminal.
```bash
pip install numpy pandas torch scikit-learn scipy brukeropus
```

### Windows
Run the following commands in Command Prompt.
```bash
cd C:\Python\Scripts\
pip.exe install numpy pandas torch scikit-learn scipy brukeropus
```

## Step 3 - Download Files
Download `OtolithAgeing.py`, `model_params.pth`, and `scalar_params.json`. The second two files are not strictly necessary; they contain model parameters for a pre-trained model. Training a new model with the script will create new versions of these files based on the model you've trained.  
Make sure all three files are in the same directory.

## Step 4 - Run the Program
Within Command Prompt/Terminal, navigate to the folder containing the `OtolithAgeing.py` file the program using the command 
```bash
cd /path/to/file
```
Then run the following command.
```bash
python OtolithAgeing.py
```

## Step 5 - Follow the program's directions
The program will prompt you to provide information as needed to run the model.
