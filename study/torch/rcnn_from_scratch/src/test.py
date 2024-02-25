import torch

if __name__ == '__main__':
    a = torch.randn(4, 4)
    print(a)

    b, c = torch.max(a, 1)
    print(b)
    print(c)