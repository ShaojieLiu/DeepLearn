from pathlib import Path

import matplotlib.pyplot as plt
from torchvision import datasets, transforms


DATA_DIR = Path("data/mnist")

train_dataset = datasets.MNIST(
    root=DATA_DIR,
    train=True,
    download=True,
    transform=transforms.ToTensor(),
)

test_dataset = datasets.MNIST(
    root=DATA_DIR,
    train=False,
    download=True,
    transform=transforms.ToTensor(),
)

print("train size:", len(train_dataset))
print("test size:", len(test_dataset))

x, y = train_dataset[0]
print("sample shape:", tuple(x.shape))
print("sample label:", y)

plt.imshow(x.squeeze(0), cmap="gray")
plt.title(f"label={y}")
plt.axis("off")
plt.savefig("data/mnist/sample.png")
print("saved sample image to data/mnist/sample.png")
