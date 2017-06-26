import torch
from torch.autograd import Variable
class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out, h, w):
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H * h * w, D_out)
    def forward(self, x):
        h_relu = self.linear1(x).clamp(min=0)
        h_relu = torch.unsqueeze(torch.unsqueeze(h_relu, 2), 3) # -> N x H x 1 x 1
        h_expand = h_relu.expand(64, H, h, w).contiguous().view(64, -1) # -> N x H x h x w
        y_pred = self.linear2(h_expand) # -> N x D_out
        return y_pred

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out, h, w = 64, 1000, 100, 10, 6, 6

x = Variable(torch.randn(N, D_in), requires_grad=True)
y = Variable(torch.randn(N, D_out), requires_grad=False)

model = TwoLayerNet(D_in, H, D_out, h, w)

criterion = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
for t in range(500):
    y_pred = model(x)
    loss = criterion(y_pred, y)
    print(t, loss.data[0])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()