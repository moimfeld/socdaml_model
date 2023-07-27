import torch
from tqdm import tqdm
from torch.nn.functional import one_hot
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def train(model, train_dataloader, val_dataloader, optimizer, criterion, device, epochs=10, print_every_iters = 1000):
    training_loss_per_epoch = []
    validation_loss_per_epoch = []
    for epoch in tqdm(range(epochs)):  # loop over the dataset multiple times
        # First we loop over training dataset
        running_loss = 0.0
        # Set network to train mode before training
        model.train()
        for i, data in enumerate(train_dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)
            labels = one_hot(labels, num_classes=10).type(torch.FloatTensor).to(device) # onehot encode, cast to float and put on device
            # zero the parameter gradients
            optimizer.zero_grad()  # zero the gradients from previous iteration
            # forward + backward + optimize
            outputs = model(inputs.to(device))  # forward pass to obtain network outputs
            loss = criterion(outputs, labels)  # compute loss with respect to labels
            loss.backward()  # compute gradients with backpropagation (autograd)
            optimizer.step()  # optimize network parameters

            # print statistics
            running_loss += loss.item()
            if (i + 1) % print_every_iters == 0:
                tqdm.write(
                    f'[Epoch: {epoch + 1} / {epochs},'
                    f' Iter: {i + 1:5d} / {len(train_dataloader)}]'
                    f' Training loss: {running_loss / (i + 1):.3f}'
                )
        
        mean_loss = running_loss / len(train_dataloader)
        training_loss_per_epoch.append(mean_loss)


        # Next we loop over validation dataset
        running_loss = 0.0

        mean_loss = evaluate(model = model, val_dataloader = val_dataloader, criterion = criterion, device = device, epoch = epoch, epochs = epochs)
        validation_loss_per_epoch.append(mean_loss)

        #early stopping:
        # if mean_loss < best_val_loss:
        #     epochs_no_improve = 0
        #     best_val_loss = mean_loss
        # else:
        #     epochs_no_improve += 1
        # if epoch > min_epochs and epochs_no_improve == patience:
        #     print(f'Early Stopping activated at epoch {epoch}')
        #     early_stop = True
        #     break

def evaluate(model, val_dataloader, criterion, device, epoch = 0, epochs = 0):
    mean_loss, total, correct = get_accuracy(model = model, dataloader = val_dataloader, criterion = criterion, device = device)
    tqdm.write(f'[Epoch: {epoch + 1} / {epochs}] Validation loss: {mean_loss:.3f}')
    tqdm.write(f'Accuracy of the model on the validation images: {100 * correct / total:.2f} %')

def test(model, test_dataloaader, criterion, device):
    mean_loss, total, correct = get_accuracy(model = model, dataloader = test_dataloaader, criterion = criterion, device = device)
    tqdm.write(f'Test loss: {mean_loss:.3f}')
    tqdm.write(f'Accuracy of the model on the test images: {100 * correct / total:.2f} %')

def get_accuracy(model, dataloader, criterion, device):
    running_loss = 0.0
    total = 0
    correct = 0
    # Set network to eval mode before validation
    model.eval()
    for i, data in enumerate(dataloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)

        # on validation dataset, we only do forward, without computing gradients
        with torch.no_grad():
            outputs = model(inputs)  # forward pass to obtain network outputs
            loss = criterion(outputs, labels)  # compute loss with respect to labels

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        # print statistics
        running_loss += loss.item()

    mean_loss = running_loss / len(dataloader)
    return mean_loss, total, correct

def get_datasets(train_val_spli_ratio, batch_size):
    # Download Dataset
    full_train_set = datasets.MNIST("./", train=True, transform=transforms.ToTensor(), download=True)
    test_set  = datasets.MNIST("./", train=False, transform=transforms.ToTensor(), download=True)

    # Instantiate Dataloaders and make train-eval splot
    train_size = int(train_val_spli_ratio * len(full_train_set))
    train_subset, val_subset = random_split(full_train_set, [train_size, len(full_train_set) - train_size], generator=torch.Generator().manual_seed(0))

    train_dataloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True , num_workers=16)
    val_dataloader   = DataLoader(val_subset  , batch_size=batch_size, shuffle=False, num_workers=16)
    test_dataloader  = DataLoader(test_set    , batch_size=batch_size, shuffle=False, num_workers=16)
    return train_dataloader, val_dataloader, test_dataloader