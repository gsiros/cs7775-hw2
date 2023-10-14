from torchvision.datasets import FashionMNIST
import torchvision.transforms as T

fmnist_labels = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot"
}

def load_fmnist_torch(root="./data", transform=None, download=True):
    
    if transform == None:
        transform = T.ToTensor()
    
    train_set = FashionMNIST(root=root,  transform=transform, download=download, train=True)
    test_set = FashionMNIST(root=root,  transform=transform, download=download, train=False)
    
    # Each item in this dictionary is a torch Dataset object
    # To feed the data into a model, you may have to use a DataLoader 
    return {"train": train_set, "test": test_set}