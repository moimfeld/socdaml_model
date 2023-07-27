import torch
from fc_model import SimpleNet
from tiny_fc_model import TinySimpleNet
from conv_model import ConvNet
from train import train, test, get_datasets
from copy import deepcopy
import utils

# C_HEADER PATHS
C_IMAGES_PATH = 'c_headers/images_lables.h'

# Train Parameters
TRAIN_VAL_SPLIT_RATIO = 0.8
BATCH_SIZE            = 16
EPOCHS                = 10
LEARNING_RATE         = 1e-3
WEIGHT_DECAY          = 1e-5

def main():
    args = utils.get_parser()

    # Set Device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    # Instantiate Model
    if args.model == 'fc_model':
        print("Specified Model: fc_model")
        C_MODEL_PATH = 'c_headers/fc_model.h'
        MODEL_PATH = 'models/fc_model.pth'
        model = SimpleNet()
    elif args.model == 'tiny_fc_model':
        print("Specified Model: tiny_fc_model")
        C_MODEL_PATH = 'c_headers/tiny_fc_model.h'
        MODEL_PATH = 'models/tiny_fc_model.pth'
        model = TinySimpleNet()
    elif args.model == 'conv_model':
        print("Specified Model: conv_model")
        C_MODEL_PATH = 'c_headers/conv_model.h'
        MODEL_PATH = 'models/conv_model.pth'
        model = ConvNet()
    else:
        raise NotImplementedError("Pass a valid model to the --model option. Valid options are: fc_model, tiny_fc_model, conv_model")
    model.to(device)

    # Get Datasets
    train_dataloader, val_dataloader, test_dataloader = get_datasets(train_val_spli_ratio = TRAIN_VAL_SPLIT_RATIO, batch_size = BATCH_SIZE)

    # Define criterion
    criterion = torch.nn.CrossEntropyLoss()

    if args.train:
        print("Training model from scratch")

        # Initialize optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)       

        # Train
        train(model = model, train_dataloader = train_dataloader, val_dataloader = val_dataloader, optimizer = optimizer, criterion = criterion, device = device, epochs = EPOCHS)

        # Test
        test(model = model, test_dataloaader = test_dataloader, criterion = criterion, device = device)

        # Save model
        torch.save(model.to('cpu').state_dict(), MODEL_PATH)

    else:
        print("Loading model from '{0}'".format(MODEL_PATH))
        model.load_state_dict(torch.load(MODEL_PATH))
        if args.eval:
            test(model = model, test_dataloaader = test_dataloader, criterion = criterion, device = device)
    
    if args.convert:
        print("Converting model to C-header file")
        utils.float_model_to_header(model = model, file_name = C_MODEL_PATH)
    
    if args.gen_data:
        print("Converting MNIST images to C-header file ")
        _, _, test_dloader = get_datasets(train_val_spli_ratio = TRAIN_VAL_SPLIT_RATIO, batch_size = 1)
        utils.float_mnist_to_header(dataset = test_dloader, file_name = C_IMAGES_PATH)
    

if __name__ == "__main__":
    main()
