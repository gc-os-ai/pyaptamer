

class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape,

    def forward(self, x):
        return x.view(*self.shape)