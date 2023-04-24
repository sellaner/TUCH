import torch
import math
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from swin_t import swin_tiny_patch4_window7_224 as create_model
import settings


class ImgNet(nn.Module):
    def __init__(self, code_len):
        super(ImgNet, self).__init__()
        # self.alexnet = torchvision.models.alexnet(pretrained=True)
        # self.alexnet.classifier = nn.Sequential(*list(self.alexnet.classifier.children())[:6])
        # for param in self.img_encoder.parameters():
        #     param.requires_grad = False

        self.img_encoder = create_model()
        weights_dict = torch.load(settings.WEIGHT_DIR)["model"]

        self.img_encoder.load_state_dict(weights_dict, strict=False)
        for p in self.parameters():
            p.requires_grad = False

        self.fc_encode = nn.Linear(768, code_len)
        self.alpha = 1.0

    def forward(self, x):
        # x = self.alexnet.features(x)
        # x = x.view(x.size(0), -1)
        # feat = self.alexnet.classifier(x)
        # x_inputs = self.feature_extractor(images=x, return_tensors="pt")
        # x = self.img_encoder(x_inputs)
        # feat = x[0][:, 0, :]
        feat = self.img_encoder(x)

        hid = self.fc_encode(feat)
        code = torch.tanh(self.alpha * hid)

        return feat, hid, code

    def set_alpha(self, epoch):
        self.alpha = math.pow((1.0 * epoch + 1.0), 0.5)


class TxtNet(nn.Module):
    def __init__(self, code_len, txt_feat_len):
        super(TxtNet, self).__init__()
        self.fc1 = nn.Linear(txt_feat_len, txt_feat_len)
        self.fc2 = nn.Linear(txt_feat_len, code_len)
        self.alpha = 1.0

    def forward(self, x):
        feat = F.relu(self.fc1(x))
        hid = self.fc2(feat)
        code = torch.tanh(self.alpha * hid)
        return feat, hid, code

    def generate_txt_code(self, x):
        feat = F.relu(self.fc1(x))
        hid = self.fc2(feat)
        code = torch.tanh(self.alpha * hid)
        return code

    def set_alpha(self, epoch):
        self.alpha = math.pow((1.0 * epoch + 1.0), 0.5)


class DIS(torch.nn.Module):
    def __init__(self, hash_dim, txt_feat_len):
        super(DIS, self).__init__()
        self.img_projection = nn.Linear(768, 512, bias=True)
        self.txt_projection = nn.Linear(txt_feat_len, 512, bias=True)
        self.attentionModel = nn.ModuleDict(
            {'img': nn.Linear(512, hash_dim),
             'txt': nn.Linear(512, hash_dim)
             })

        self.weight_init()

    def weight_init(self):
        initializer = self.kaiming_init

        for m in self._modules:
            initializer(m)

    def kaiming_init(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0)
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.weight.data.fill_(1)
            if m.bias is not None:
                m.bias.data.fill_(0)

    def forward(self, F_I, F_T):
    
        x_I = self.img_projection(F_I)
        y_T = self.txt_projection(F_T)

        x_att = self.attentionModel['img'](x_I)
        y_att = self.attentionModel['txt'](y_T)

        x_concat_y = settings.joo * x_att + settings.zoo * y_att

        com_f = torch.tanh(x_concat_y)

        return com_f