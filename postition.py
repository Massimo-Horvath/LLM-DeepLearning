class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=50000):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()

        r = max_len**(2/d_model) * torch.ones(d_model)
        div_term = torch.pow(r, torch.floor(torch.arange(0,d_model)/2))

        self.encoding[:, 0::2] = torch.sin(position * div_term[0::2])
        self.encoding[:, 1::2] = torch.cos(position * div_term[1::2])
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        return x + self.encoding[:, :x.size(1)].detach()