import torch

def main():
    model = torch.load("models/model.pth")
    test_set = torch.load("data/processed/train.pt")

    accuracy = 0
    for i, (images, labels) in enumerate(test_set):
        output = model.forward(images)
        ps = torch.exp(output)
        equality = labels.data == ps.max(1)[1]
        accuracy += equality.type_as(torch.FloatTensor()).sum()
    print(f"Accuracy: {100*accuracy/len(test_set.dataset)}%")


if __name__ == "__main__":
    main()