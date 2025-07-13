'''
Script for training and estimating fish age from Otolith scans and metadata

'''

# Imports
from brukeropus import read_opus

import pandas as pd

from json import dump, load

from os import listdir

from re import compile

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler

from copy import deepcopy

from numpy import sqrt, array, round

import time

# Script Parameters
MODEL_FEATURES = ['TrayBienID', 'Latitude', 'Longitude', 'Sex', 'Length (cm)', 'Wgt_g']
MODEL_TARGET = 'age'
CAT_FEATURES = ['Sex']
SAVE_MSE = True

## Model Parameters
SCAN_REDUCTION = 100
TRAIN_EPOCHS = 400
LEARNING_RATE = 6e-5
DECAY_RATE = 0.999  # StepLR
BATCH_SIZE = 16

## Model Validation Parameters
VALIDATION_SEED = 4484
KFOLD_SPLITS = 5

# script-wide variables (Don't change)
MODEL_PARAMS = None
SCALAR_PARAMS = None

# setting up hardware acceleration (hopefully running on GPU/M2)
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f'Training on {device}')

# Helper Functions
def elements_exist(containerA, containerB):
    for element in containerA:
        if element not in containerB:
            return False, element
    return True, None

def shrinkify(x, y, new_count=32, batch_factor=2, func=None):
    from numpy import exp, array, dot, linspace, median
    if func == None:
        func = lambda x, origin: 2 * exp(-((x-origin)/10)**2)
    
    step = len(x) / new_count
    batch_size = int((batch_factor*step) - step)
    indices = linspace(0, len(x) - step, new_count, dtype=int)
    new_x = list()
    new_y = list()
    for start,stop in zip(indices[:-1], indices[1:]):
        weights = array([func(i, x[start]) for i in x[start:stop + batch_size]])
        adjusted_values = dot(y[start:stop + batch_size], weights) / sum(weights)
        new_x += [median(x[start:stop])]
        new_y += [adjusted_values]

    weights = array([func(i, x[indices[-1]]) for i in x[indices[-1]:]])
    adjusted_values = dot(y[indices[-1]:], weights) / sum(weights)
    new_x += [x[indices[-1]]]
    new_y += [adjusted_values]

    return (new_x, new_y)

def auc(x, y):
    answer = 0
    prev_x, prev_y = x[0],y[0]
    for x, y in zip(x[1:], y[1:]):
        height = (prev_y + y) / 2
        width = abs(prev_x - x)
        answer += height*width
        prev_x, prev_y = x, y

    return answer

def format_time(seconds):
    minutes = int(seconds / 60)
    seconds = seconds % 60
    return f'{minutes}m {seconds:.2f}s'

def reset_weights(model):
    """Reset model weights to default initialization"""
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

# Training Mode
def train_mode():
    global MODEL_PARAMS
    global SCALAR_PARAMS
    
    print('------------')
    # load scan data
    otolith_path = input('Enter the path of the directory with the otolith scans (.0)\n  ')

    ## Checks for inserted space after a path
    if otolith_path[-1] == ' ':
        otolith_path = otolith_path[:-1]

    num_vals = SCAN_REDUCTION
    columns = ['specimen_id'] + ['AUC'] + ['FTNIRS_bin_' + str(x+1) for x in range(num_vals)]

    get_id = lambda x: int(compile(r'\d{7,}').findall(x)[0])

    rows = []

    for file in listdir(otolith_path):
        if file[-2:] == '.0':
            file_path = otolith_path + "/" + file
            opus_file = read_opus(filepath=file_path)
            if 'a' in opus_file.data_keys:
                
                id = get_id(file)

                ax = opus_file.a.x
                ay = opus_file.a.y

                AUC = auc(ax, ay)

                new_y = shrinkify(ax, ay, new_count=num_vals)[1]

                rows.append([id] + [AUC] + new_y)

    scan_data = pd.DataFrame(rows, columns=columns)

    # load metadata
    metadata_path = input('Enter the path for the metadata\n  ')

    ## checks for inserted space after path
    if metadata_path[-1] == ' ':
        metadata_path = metadata_path[:-1]

    metadata = pd.read_csv(metadata_path) if '.csv' in metadata_path else pd.read_excel(metadata_path)

    # check existence of variables
    print('------------\nExpected variables:')
    for i in MODEL_FEATURES + [MODEL_TARGET]:
        print('  ' + i)
    
    print('Expected Categorical Variables')
    for i in CAT_FEATURES:
        print('  ' + i)
    print('------------')
    columns = metadata.columns.tolist()

    exists, element = elements_exist(MODEL_FEATURES + [MODEL_TARGET], columns)
    
    if not exists:
        raise Exception(f'Variable "{element}" was not found in the file')
    
    # Select important variables
    metadata = metadata[MODEL_FEATURES + [MODEL_TARGET]]

    # Remove Non-Values
    metadata = metadata.dropna()

    # One-Hot Encoding of categorical variables
    metadata = pd.get_dummies(metadata, columns=CAT_FEATURES)
    metadata = metadata.rename({'TrayBienID':'specimen_id'}, axis=1)

    # combine data
    fish_data = metadata.merge(scan_data, on='specimen_id', how='inner').drop(labels='specimen_id', axis=1)

    print(f"Data Dimensions: {fish_data.shape}")

    # Data Preparation
    X, y = fish_data.drop(labels='age', axis=1).to_numpy(), fish_data[['age']].to_numpy()

    ## Dataset class
    class OtolithDataset(Dataset):
        def __init__(self, X, y):
            self.X = X
            self.y = y

        def __len__(self):
            return len(self.X)
        
        def __getitem__(self, index):
            return self.X[index], self.y[index]
        
    # Model Definition

    ## Neural Network Set Up
    input_length = fish_data.shape[1] - 1 # this represents the number of features
    print(f'Number of Features: {input_length}')
    model = nn.Sequential(
        nn.Linear(input_length, input_length * (input_length - 1)),
        nn.ReLU(),

        nn.Linear(input_length * (input_length - 1), 64),
        nn.ReLU(),

        nn.Linear(64, 1)
    ).to(device)

    ## Optimizer Parameters
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)
    scheduler = StepLR(optimizer=optimizer, step_size=1, gamma=DECAY_RATE)
    
    # Create Train/Test methods
    def train(model, dataloader, optimizer, loss_fn):
        model.train()

        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            # Compute Prediction error
            pred = model(X)
            loss = loss_fn(pred, y)

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    def test(model, dataloader, loss_fn):
        model.eval()

        test_loss = 0
        correct = 0

        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                correct += (abs((y - pred).cpu().numpy()) <= 0.5).sum().item()

        return test_loss / len(dataloader), correct / len(dataloader.dataset)
    
    # Validation

    ## KFold Cross Validation Set Up
    kf_cross_val = KFold(n_splits=KFOLD_SPLITS, shuffle=True, random_state=VALIDATION_SEED)

    sum_rmse = 0
    sum_acc = 0

    best_mse_weights = None
    best_acc_weights = None

    best_mse = 10000.0
    best_acc = 0.0

    print("—————————————————— KFold Cross Validation ————————————————————")
    for i, (train_index, test_index) in enumerate(kf_cross_val.split(X)):
        start_time = time.time()

        # set up fold data
        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        y_test = y[test_index]

        scalar = StandardScaler()
        scalar.fit(X_train)
        X_train = torch.tensor(scalar.transform(X_train), dtype=torch.float32)
        X_test = torch.tensor(scalar.transform(X_test), dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1,1)
        y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1,1)

        # Dataset and Dataloader Creation
        train_dataset = OtolithDataset(X_train, y_train)
        train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

        test_dataset = OtolithDataset(X_test, y_test)
        test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

        # Train

        mse_history = []

        best_fold_mse, best_fold_acc = test(model, test_dataloader, loss_fn)

        for epoch in range(TRAIN_EPOCHS):
            
            # Training on batch
            train(model, train_dataloader, optimizer, loss_fn)

            # adjust learning rate
            scheduler.step()

            mse, accuracy = test(model, test_dataloader, loss_fn)

            mse_history.append(mse)

            if mse < best_fold_mse:
                best_fold_mse = mse
            
            if accuracy > best_fold_acc:
                best_fold_acc = accuracy

            if mse < best_mse:
                best_mse = mse
                best_mse_weights = deepcopy(model.state_dict())
                if SAVE_MSE:
                    SCALAR_PARAMS = {'Means':scalar.mean_.tolist(), 'Variances':scalar.var_.tolist(), 'Scales':scalar.scale_.tolist()}
                    

            if accuracy > best_acc:
                best_acc = accuracy
                best_acc_weights = deepcopy(model.state_dict())
                if not SAVE_MSE:
                    SCALAR_PARAMS = {'Means':scalar.mean_.tolist(), 'Variances':scalar.var_.tolist(), 'Scales':scalar.scale_.tolist()}

        # Reset Weights so model doesn't learn across folds
        reset_weights(model)

        # print fold info
        rmse = sqrt(best_fold_mse)
        fold_time = time.time() - start_time
        m  = f'Fold {i + 1} took {format_time(fold_time)}\n'
        m += f'  RMSE: {rmse:.3f}\n  Accuracy: {best_fold_acc*100:.2f}\n'
        m += "——————————————————————————————————————————————————————————————"
        print(m)

        sum_rmse += rmse
        sum_acc += best_fold_acc
    
    print(f'Average RMSE: {sum_rmse / KFOLD_SPLITS:.3f}\nAverage Accuracy: {sum_acc * 100 / KFOLD_SPLITS:.2f}')

    print(f'Best RMSE: {sqrt(best_mse):.3f}\nBest Accuracy: {best_acc*100:.2f}')

    # Load preferred weights
    if SAVE_MSE:
        model.load_state_dict(best_mse_weights)

    else:
        model.load_state_dict(best_acc_weights)

    print(f'--------------------\n')

    MODEL_PARAMS = best_mse_weights if SAVE_MSE else best_acc_weights

    torch.save(MODEL_PARAMS, 'model_params.pth')

    with open("scalar_params.json", 'w') as f:
        dump(SCALAR_PARAMS, f, indent=4)

    menu()

# Estimation Mode
def estimate_mode():
    device = 'cpu'
    global MODEL_PARAMS
    global SCALAR_PARAMS

    # load scan data
    otolith_path = input('Enter the path of the directory with the otolith scans (.0)\n  ')

    ## Checks for inserted space after a path
    if otolith_path[-1] == ' ':
        otolith_path = otolith_path[:-1]

    num_vals = SCAN_REDUCTION
    columns = ['specimen_id'] + ['AUC'] + ['FTNIRS_bin_' + str(x+1) for x in range(num_vals)]

    get_id = lambda x: int(compile(r'\d{7,}').findall(x)[0])

    rows = []

    for file in listdir(otolith_path):
        if file[-2:] == '.0':
            file_path = otolith_path + "/" + file
            opus_file = read_opus(filepath=file_path)
            if 'a' in opus_file.data_keys:
                
                id = get_id(file)

                ax = opus_file.a.x
                ay = opus_file.a.y

                AUC = auc(ax, ay)

                new_y = shrinkify(ax, ay, new_count=num_vals)[1]

                rows.append([id] + [AUC] + new_y)

    scan_data = pd.DataFrame(rows, columns=columns)

    # load metadata
    metadata_path = input('Enter the path for the metadata file\n  ')

    ## checks for inserted space after path
    if metadata_path[-1] == ' ':
        metadata_path = metadata_path[:-1]

    metadata = pd.read_csv(metadata_path) if '.csv' in metadata_path else pd.read_excel(metadata_path)

    # check existence of variables
    print('------------\nExpected variables:')
    for i in MODEL_FEATURES:
        print('  ' + i)
    
    print('Expected Categorical Variables')
    for i in CAT_FEATURES:
        print('  ' + i)
    print('------------')
    columns = metadata.columns.tolist()

    exists, element = elements_exist(MODEL_FEATURES, columns)
    
    if not exists:
        raise Exception(f'Variable "{element}" was not found in the file')
    
    # Select important variables
    metadata = metadata[MODEL_FEATURES + [MODEL_TARGET]]

    # Remove Non-Values
    metadata = metadata.dropna()

    # One-Hot Encoding of categorical variables
    metadata = pd.get_dummies(metadata, columns=CAT_FEATURES)
    metadata = metadata.rename({'TrayBienID':'specimen_id'}, axis=1)

    # combine data
    fish_data = metadata.merge(scan_data, on='specimen_id', how='inner').drop(labels='specimen_id', axis=1)

    print(f"{fish_data.shape[0]} values to predict")

    if MODEL_PARAMS is None:
        MODEL_PARAMS = torch.load("model_params.pth")
        with open("scalar_params.json", 'r') as f:
            SCALAR_PARAMS = load(f)

    # Data Preparation
    X = fish_data.drop(labels=[MODEL_TARGET], axis=1).to_numpy()

    scalar = StandardScaler()
    scalar.mean_ = array(SCALAR_PARAMS['Means'])
    scalar.var_ = array(SCALAR_PARAMS['Variances'])
    scalar.scale_ = array(SCALAR_PARAMS['Scales'])

    X = torch.tensor(scalar.transform(X), dtype=torch.float32)

    input_length = fish_data.shape[1] - 1 # this represents the number of features
    print(f"Number of Features: {input_length}")

    model = nn.Sequential(
        nn.Linear(input_length, input_length * (input_length - 1)),
        nn.ReLU(),

        nn.Linear(input_length * (input_length - 1), 64),
        nn.ReLU(),

        nn.Linear(64, 1)
    ).to(device)

    model.load_state_dict(MODEL_PARAMS)

    with torch.no_grad():
        model.eval()

        X = X.to(device)

        predictions = model(X).numpy().reshape(-1)
    rounded_preds = round(predictions, 0).astype(int)
    pd.DataFrame({'RawPredictions': predictions, 'RoundedPredictions':rounded_preds}).to_csv('PredictedAges.csv', sep=',', index=False)

    print('--------------------')

    menu()

# Help Mode
def help_mode():
    m = '------------------------------------\n'
    m += 'What would you like help with?\n'
    m += '  (1) Data input formatting\n'
    m += '  (2) How to use the "Train" program to train the model\n'
    m += '  (3) How to use the "Estimate" program to make predictions\n'
    m += '  (4) Description of script variables\n'
    m += '  (5) Return to main menu\n    '

    print(m, end='')
    get_num = lambda x: int(compile(r'\d').findall(x)[0])
    option = get_num(input())
    print('------------------------------------')

    match option:
        
        case 1:
            m = "    There are some important requirements for the directory of the FTNIRS scans and the metadata file. The title of each scan file (.0) must include a 7-10 digit code representing the ID of the fish. This is used to match that scan file with the appropriate row in the metadata file. The metadata file (.csv or excel) must include a column containing that ID so that the correct row may be matched to the correct scan. \n    The current ID is 'TrayBienID'. If the variable name for the ID changes, this must be updated in the MODEL_FEATURES variable at the top of the script to match. See the 'Description of script variables' help option for more information on MODEL_FEATURES.\n    As long as the metadata file has all the necessary columns listed in the MODEL_FEATURES variable (and the MODEL_TARGET variable if you are training) the script can make predictions and train. Extraneous columns will not make a difference.\n"
            m += '------------------------------------'
            print(m)
            input('Press <Enter> to continue')
            help_mode()
        
        case 2:
            m = "    The 'Train' mode is used to train new models. You will be asked to enter the paths for the directory of the scan data and the path for the metadata file (see 'Data input formatting' for more information on requirements for these files). The metadata file MUST include an age variable to train the model. The name of this variable must entered into the script variable MODEL_TARGET. The train mode will train the model on the provided data and print extra information about the cross-validation the model undergoes.\n    If you wish to make changes to how the model trains, increase/decrease the number of features in the model, or otherwise modify the training process, changing the script variables at the top of the script is your easiest option. To find a description of each variable, choose the 'Description of script variables' option in the Help menu.\n    Each fold of the cross validation will take ~1 minute when TRAIN_EPOCHS is 300. With 5 folds, training will generall take ~5 minutes. The program will print a summary of the models performance on the test data after each fold.\n    Once the model has finished training, it will print a summary of the RMSE and Accuracy of the model. It will also create two files in the same directory as the program. One file (.pth) contains the model weights and parameters and the other file (.json) contains the necessary scalar parameters for transforming the input data.\n"
            m += '------------------------------------'
            print(m)
            input('Press <Enter> to continue')
            help_mode()
        
        case 3:
            m = "    The 'Estimate' mode is used to make predictions. This mode will also ask you to provide paths to a directory of scans and to a metadata file containing the remaining features necessary for the model. \n    If the training program has not been run but there  are existing 'model_params.pth' and 'scalar_params.json' files, then the estimate program will attempt to load these files and access the previously trained model.\n    The model will output a .csv named 'PredictedAges.csv' which contains raw and rounded predictions.\n"
            m += '------------------------------------'
            print(m)
            input('Press <Enter> to continue')
            help_mode()
        
        case 4:
            m = "Script variables are the quickest and safest way to modify model behaviour and performance. I have already used a grid search approach to optimize most of the values (e.g. LEARNING_RATE or DECAY_RATE). They can be found at the top of the script below the 'import' statements. They are listed in the order they appear in the script.\n\n"
            m += 'MODEL_FEATURES -- This variable contains a list of the metadata features used to predict fish age (e.g. Sex, Length, weight). These are case-sensitive and the script won\'t include features if the name in the variable differs from the name in the metadata file. Additionally the ID of the fish is included in this variable to assist with merging the metadata and scan data even though the ID is not used in prediction.\n\n'
            m += 'MODEL_TARGET -- This variable holds the name of the target variable (i.e. fish "age") as listed in the metadata file\n\n'
            m += 'CAT_FEATURES -- This variable lists all variables mentioned in MODEL_FEATURES that are categorical. Categorical variables are not ordinal and often not numeric. In order to include these variables the script must do something called "One-Hot encoding". List all categorical variables in CAT_FEATURES.\n\n'
            m += 'SAVE_MSE -- This variable indicates whether to prioritize mean squared error (MSE) or accuracy. While the best model parameters for minimizing MSE are not ALWAYS different from the best model parameters for maximizing accuracy they often can be. Set this variable to "False" to prioritize accuracy over MSE.\n\n'
            m += 'SCAN_REDUCTION -- There is an overabundance of data points in the FTNIRS scans. The "shrinkify" method reduces the number to the size given by SCAN_REDUCTION. "shrinkify" works by using a weighted average to approximate the scanned data.\n\n'
            m += 'TRAIN_EPOCHS -- This variable gives the number of epochs the model should spend optimizing. The model will automatically pick the epoch with the lowest test RMSE or highest accuracy (depends on the SAVE_MSE variable). If you have more data, consider increasing the number of epochs. This will take longer but give the model more time to find an efficient set of parameters. If you lower the LEARNING_RATE  you may also need to increase the number of epochs.\n\n'
            m += 'DECAY_RATE -- This variable reduces the LEARNING_RATE each epoch multiplicatively. This prevents the model from overshooting optimal solutions as it gets closer.\n\n'
            m += 'BATCH_SIZE -- This variable changes the number of samples the model processes before modifying its parameters. Increasing the size means more regularization and less chance of overfitting but slower learning.\n\n'
            m += "VALIDATION_SEED -- This variable sets the random seed so that the sections of data the model trains on are consistent. This is useful for reproducible results. It does NOT affect batch shuffling for gradient descent though so trials will not be completely identical.\n\n"
            m += "KFOLD_SPLITS -- This variable sets the number of sections the data is split into during K-Fold cross-validation. This generally shouldn't fall outside the range of 3-12.\n\n"
            m += '------------------------------------'
            print(m)
            input('Press <Enter> to continue')
            help_mode()
        
        case default:
            menu()

# Main Menu Method
def menu():
    m = 'Enter the number for the preferred menu option:\n  '
    m += '(1) Train    - enter training data to train the model\n  '
    m += '(2) Estimate - enter data from which to estimate fish ages\n  '
    m += '(3) Help     - get a description of the script and how to use it\n  '
    m += '(4) Quit     - Exit the program\n    '

    print(m, end='')
    mode = input()

    get_num = compile(r'\d')
    mode = int(get_num.findall(mode)[0])
    if mode > 4:
        raise Exception("Please restart and select one of the menu options")

    elif mode == 1:
        train_mode()

    elif mode == 2:
        estimate_mode()

    elif mode == 3:
        help_mode()

    exit(0)

menu()

exit(0)

