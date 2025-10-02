import torch, torch.nn as nn

class ConvSubsampler(nn.Module):
    def __init__(self, in_ch, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, (3,3), stride=(2,2), padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden, hidden, (3,3), stride=(2,2), padding=1),
            nn.ReLU(),
        )
    def forward(self, x):            
        return self.net(x)          

class WakeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = ConvSubsampler(1, 32)
        self.gru = nn.GRU(32*10, 64, batch_first=True, bidirectional=True)
        self.fc  = nn.Linear(64*2, 1)

    def forward(self, x):            
        x = self.cnn(x)
        b,c,t,f = x.shape
        x = x.permute(0,2,1,3).contiguous().view(b,t,-1)
        _, h = self.gru(x)           
        h = torch.cat([h[0],h[1]], dim=-1)
        return torch.sigmoid(self.fc(h))[:,0]  

class CmdNet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.cnn = ConvSubsampler(1, 32)
        self.gru = nn.GRU(32*10, 128, batch_first=True, bidirectional=True)
        self.fc  = nn.Linear(128*2, n_classes)

    def forward(self, x):
        x = self.cnn(x)
        b,c,t,f = x.shape
        x = x.permute(0,2,1,3).contiguous().view(b,t,-1)
        _, h = self.gru(x)
        h = torch.cat([h[0],h[1]], dim=-1)
        return self.fc(h)            
