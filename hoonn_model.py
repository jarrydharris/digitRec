import torch

#----------------------------------------
# Create Model
class Net(torch.nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1,10,kernel_size = 5) #1st argument is the depth of image, # 2nd argument is how many filters there are?
        self.conv2 = torch.nn.Conv2d(10,20,kernel_size = 5)
        self.mp = torch.nn.MaxPool2d(2)
        self.fc = torch.nn.Linear(320,10)

    def forward(self, x):
        in_size = x.size(0)
        x = torch.nn.functional.relu(self.mp(self.conv1(x)))
        x = torch.nn.functional.relu(self.mp(self.conv2(x)))
        x = x.view(in_size, -1) # flatten the tensor
        return torch.nn.functional.log_softmax(self.fc(x))


