from .. import *
import pandas as pd
from sklearn.datasets import load_boston
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.sampler import SequentialSampler


def get_dataset_from_code(code, batch_size):
    """ interface to get function object
    Args:
        code(str): specific data type
    Returns:
        (torch.utils.data.DataLoader): train loader 
        (torch.utils.data.DataLoader): test loader
    """
    dataset_root = "./assets/data"
    if code == 'mnist':
        train_loader, test_loader = get_mnist_data(batch_size=batch_size,
            data_folder_path=os.path.join(dataset_root, 'mnist-data'))
    elif code == 'cifar10':
        train_loader, test_loader = get_cifar10_data(batch_size=batch_size,
            data_folder_path=os.path.join(dataset_root, 'cifar10-data'))
    elif code == 'fmnist':
        train_loader, test_loader = get_fasionmnist_data(batch_size=batch_size,
            data_folder_path=os.path.join(dataset_root, 'fasionmnist-data'))

    elif code == 'boston':
        train_loader, test_loader = get_boston_data(batch_size=batch_size,
            data_folder_path=os.path.join(dataset_root, 'boston-data'))

    else:
        raise ValueError("Unknown data type : [{}] Impulse Exists".format(data_name))

    return train_loader, test_loader

def get_boston_data(data_folder_path,batch_size=64):
    """
    Args:
        train_batch_size(int): training batch size 
        test_batch_size(int): test batch size
    Returns:
        (torch.utils.data.DataLoader): train loader 
        (torch.utils.data.DataLoader): test loader
    """
    #Dataset Loading
    boston_dataset = load_boston()
    x=boston_dataset.data
    y=boston_dataset.target
    #Convert to Tensors
    inputs = torch.Tensor(x)
    targets = torch.Tensor(y)
    #Create Dataset
    boston_ds = TensorDataset(inputs, targets)
    batch_size = batch_size
    test_split = .2
    shuffle_dataset = True
    random_seed= 42

    # Creating data indices for training and validation splits:
    dataset_size = len(boston_ds)
    indices = list(range(dataset_size))
    split = int(np.floor(test_split * dataset_size))
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, test_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SequentialSampler(train_indices)
    test_sampler = SequentialSampler(test_indices)
    kwargs = {'num_workers': 4, 'pin_memory': True}

    train_loader = torch.utils.data.DataLoader(boston_ds, batch_size=batch_size, 
                                               sampler=train_sampler,shuffle=False, **kwargs)
    test_loader = torch.utils.data.DataLoader(boston_ds, batch_size=batch_size,
                                                    sampler=test_sampler,shuffle=False, **kwargs)

    return train_loader, test_loader

def get_fasionmnist_data(data_folder_path, batch_size=64):
    # Define a transform to normalize the data
    transform = transforms.Compose([transforms.ToTensor(),
                                   #transforms.Normalize((0.2860,), (0.3530,)),
                                 ])
    # Download and load the training data
    trainset = datasets.FashionMNIST(data_folder_path, download=True, train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False)

    # Download and load the test data
    testset = datasets.FashionMNIST(data_folder_path, download=True, train=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

    return trainloader, testloader

def get_mnist_data(data_folder_path, batch_size=64):
    """ mnist data
    Args:
        train_batch_size(int): training batch size 
        test_batch_size(int): test batch size
    Returns:
        (torch.utils.data.DataLoader): train loader 
        (torch.utils.data.DataLoader): test loader
    """
    train_data = datasets.MNIST(data_folder_path, train=True,  download=True, 
        transform=transforms.Compose([
            transforms.ToTensor(), 
            #transforms.Normalize((0.1307,), (0.3081,))
            ])
        )

    test_data  = datasets.MNIST(data_folder_path, train=False, download=True, 
        transform=transforms.Compose([
            transforms.ToTensor(), 
            #transforms.Normalize((0.1307,), (0.3081,))
            ])
        )

    kwargs = {'num_workers': 4, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(train_data, 
        batch_size=batch_size, shuffle=False, **kwargs)
    test_loader  = torch.utils.data.DataLoader(test_data,  
        batch_size=batch_size, shuffle=False, **kwargs)

    return train_loader, test_loader

def get_cifar10_data(data_folder_path, batch_size=64):
    """ cifar10 data
    Args:
        train_batch_size(int): training batch size 
        test_batch_size(int): test batch size
    Returns:
        (torch.utils.data.DataLoader): train loader 
        (torch.utils.data.DataLoader): test loader
    """
    transform_train = transforms.Compose([

        #transforms.RandomCrop(32, padding=4),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        #transforms.Normalize((0.4913, 0.4821, 0.4465), (0.2470, 0.2434, 0.2615)),

    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize((0.4913, 0.4821, 0.4465), (0.2470, 0.2434, 0.2615)),
    ])

    train_data = datasets.CIFAR10(data_folder_path, train=True, 
        download=True, transform=transform_train)
    test_data  = datasets.CIFAR10(data_folder_path, train=False, 
        download=True, transform=transform_test) 

    kwargs = {'num_workers': 4, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(train_data, 
        batch_size=batch_size, shuffle=False, **kwargs)
    test_loader  = torch.utils.data.DataLoader(test_data, 
        batch_size=batch_size, shuffle=False, **kwargs)

    return train_loader, test_loader
