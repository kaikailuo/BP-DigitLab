import matplotlib.pyplot as plt
from torchvision.datasets import EMNIST
from PIL import ImageOps

EMNIST_LETTERS_CLASS_NAMES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

dataset = EMNIST(
    root="./data",
    split="mnist",
    train=True,
    download=True
)

plt.figure(figsize=(10, 4))

for i in range(20, 40):
    img, label = dataset[i]

    # EMNIST letters 的 label 是 1-26，需要转成 0-25
    class_name = EMNIST_LETTERS_CLASS_NAMES[int(label) - 1]

    # 逆时针旋转 90°
    img = ImageOps.mirror(img.rotate(-90, expand=True))

    plt.subplot(2, 10, i - 19)
    plt.imshow(img, cmap="gray")
    plt.title(f"{class_name}\nraw={label}")
    plt.axis("off")

plt.tight_layout()
plt.show()