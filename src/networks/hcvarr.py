import torch
import torch.nn.functional as F

from .blocks import *

from .swin_model import swinB as create_model

class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(self.shape)

class element_wise_attention(nn.Module):
    def __init__(self,dim):
        super().__init__()
        B,C,H,W = dim
        q = torch.empty(1,C,H,W, requires_grad=True)
        nn.init.kaiming_normal_(q)
        self.q=nn.Parameter(q)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x1, x2):
        stream1, stream2 = x1, x2
        d1 = self.q * x1
        d1 = d1.unsqueeze(0)

        d2 = self.q * x2
        d2 = d2.unsqueeze(0)

        ds = torch.cat((d1, d2), 0)

        temp = torch.softmax(ds, 0)
        weight1 = temp[0,:]
        weight2 = temp[1,:]

        stream1 = weight1* stream1
        stream2 = weight2* stream2

        result = stream1 + stream2

        return result
        
class HCVARR(nn.Module):
    def __init__(self, row_col=True, dropout=True, do_contrast=False, levels='111'):
        super(HCVARR, self).__init__()
        self.do_contrast = do_contrast
        self.levels = levels
        print(f'CONTRAST: {self.do_contrast}')
        print(f'LEVELS: {self.levels}')

        self.high_dim, self.high_dim0 = 64, 32
        self.mid_dim, self.mid_dim0 = 128, 64
        self.low_dim, self.low_dim0 = 256, 128

        self.perception_net_high = nn.Sequential(
            nn.Conv2d(1, self.high_dim0, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.high_dim0),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(self.high_dim0, self.high_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.high_dim),
            nn.ReLU(inplace=True))

        self.perception_net_mid = nn.Sequential(
            nn.Conv2d(self.high_dim, self.mid_dim0, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.mid_dim0),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(self.mid_dim0, self.mid_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.mid_dim),
            nn.ReLU(inplace=True)
            )

        self.perception_net_low = nn.Sequential(
            nn.Conv2d(self.mid_dim, self.low_dim0, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.low_dim0),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(self.low_dim0, self.low_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.low_dim),
            nn.ReLU(inplace=True)
            )

        self.g_function_high = nn.Sequential(Reshape(shape=(-1, 3 * self.high_dim, 20, 20)),
                                             conv3x3(3 * self.high_dim, self.high_dim),
                                             ResBlock(self.high_dim, self.high_dim),
                                             ResBlock(self.high_dim, self.high_dim))
        self.g_function_mid = nn.Sequential(Reshape(shape=(-1, 3 * self.mid_dim, 10, 10)),
                                            conv3x3(3 * self.mid_dim, self.mid_dim),
                                            ResBlock(self.mid_dim, self.mid_dim),
                                            ResBlock(self.mid_dim, self.mid_dim))
        self.g_function_low = nn.Sequential(Reshape(shape=(-1, 3 * self.low_dim, 5, 5)),
                                            conv1x1(3 * self.low_dim, self.low_dim),
                                            ResBlock1x1(self.low_dim, self.low_dim),
                                            ResBlock1x1(self.low_dim, self.low_dim))

        self.conv_row_high = conv3x3(self.high_dim, self.high_dim)
        self.bn_row_high = nn.BatchNorm2d(self.high_dim)
        self.conv_col_high = conv3x3(self.high_dim, self.high_dim) if row_col else self.conv_row_high
        self.bn_col_high = nn.BatchNorm2d(self.high_dim, ) if row_col else self.bn_row_high

        self.conv_row_mid = conv3x3(self.mid_dim, self.mid_dim)
        self.bn_row_mid = nn.BatchNorm2d(self.mid_dim)
        self.conv_col_mid = conv3x3(self.mid_dim, self.mid_dim) if row_col else self.conv_row_mid
        self.bn_col_mid = nn.BatchNorm2d(self.mid_dim) if row_col else self.bn_row_mid

        self.conv_row_low = conv1x1(self.low_dim, self.low_dim)
        self.bn_row_low = nn.BatchNorm2d(self.low_dim)
        self.conv_col_low = conv1x1(self.low_dim, self.low_dim) if row_col else self.conv_row_low
        self.bn_col_low = nn.BatchNorm2d(self.low_dim) if row_col else self.bn_row_low

        self.bn_row_high.register_parameter('bias', None)
        self.bn_col_high.register_parameter('bias', None)
        self.bn_row_mid.register_parameter('bias', None)
        self.bn_col_mid.register_parameter('bias', None)
        self.bn_row_low.register_parameter('bias', None)
        self.bn_col_low.register_parameter('bias', None)

        self.mlp_dim_high = self.mlp_dim_mid = self.mlp_dim_low = 0
        if self.levels[0] == '1':
            self.mlp_dim_high = 128
            self.res1_high = ResBlock(self.high_dim, 2 * self.high_dim, stride=2,
                                      downsample=nn.Sequential(conv1x1(self.high_dim, 2 * self.high_dim, stride=2),
                                                               nn.BatchNorm2d(2 * self.high_dim)
                                                               )
                                      )

            self.res2_high = ResBlock(2 * self.high_dim, self.mlp_dim_high, stride=2,
                                      downsample=nn.Sequential(conv1x1(2 * self.high_dim, self.mlp_dim_high, stride=2),
                                                               nn.BatchNorm2d(self.mlp_dim_high)
                                                               )
                                      )

        if self.levels[1] == '1':
            self.mlp_dim_mid = 128
            self.res1_mid = ResBlock(self.mid_dim, 2 * self.mid_dim, stride=2,
                                     downsample=nn.Sequential(conv1x1(self.mid_dim, 2 * self.mid_dim, stride=2),
                                                              nn.BatchNorm2d(2 * self.mid_dim)
                                                              )
                                     )

            self.res2_mid = ResBlock(2 * self.mid_dim, self.mlp_dim_mid, stride=2,
                                     downsample=nn.Sequential(conv1x1(2 * self.mid_dim, self.mlp_dim_mid, stride=2),
                                                              nn.BatchNorm2d(self.mlp_dim_mid)
                                                              )
                                     )

        if self.levels[2] == '1':
            self.mlp_dim_low = 128
            self.res1_low = nn.Sequential(conv1x1(self.low_dim, self.mlp_dim_low),
                                          nn.BatchNorm2d(self.mlp_dim_low),
                                          nn.ReLU(inplace=True))
            self.res2_low = ResBlock1x1(self.mlp_dim_low, self.mlp_dim_low)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.mlp_dim = self.mlp_dim_high + self.mlp_dim_mid + self.mlp_dim_low
        self.mlp = nn.Sequential(nn.Linear(self.mlp_dim, 256, bias=False),
                                 nn.BatchNorm1d(256),
                                 nn.ReLU(inplace=True),
                                 nn.Dropout(0.5),
                                 nn.Linear(256, 128, bias=False),
                                 nn.BatchNorm1d(128),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(128, 1, bias=True))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.InstanceNorm2d, nn.BatchNorm2d)):
                if m.affine:
                    if m.weight is not None:
                        nn.init.constant_(m.weight, 1)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

        self.F_func_high = nn.Sequential(nn.Conv2d(64*3,64,1), 
                                    nn.ReLU(inplace=True))
        self.F_func_mid = nn.Sequential(nn.Conv2d(128*3,128,1), 
                                    nn.ReLU(inplace=True))
        self.F_func_low = nn.Sequential(nn.Conv2d(256*3,256,1), 
                                    nn.ReLU(inplace=True))

        self.swinT = create_model()
    def triples(self, input_features):
        N, K, C, H, W = input_features.shape
        K0 = K - 8
        choices_features = input_features[:, 8:, :, :, :].unsqueeze(2)  # N, 8, 64, 20, 20 -> N, 8, 1, 64, 20, 20

        row1_features = input_features[:, 0:3, :, :, :]  # N, 3, 64, 20, 20
        row2_features = input_features[:, 3:6, :, :, :]  # N, 3, 64, 20, 20
        row3_pre = input_features[:, 6:8, :, :, :].unsqueeze(1).expand(N, K0, 2, C, H,
                                                                       W)  # N, 2, 64, 20, 20 -> N, 1, 2, 64, 20, 20 -> N, 8, 2, 64, 20, 20
        row3_features = torch.cat((row3_pre, choices_features), dim=2).view(N * K0, 3, C, H, W)

        col1_features = input_features[:, 0:8:3, :, :, :]  # N, 3, 64, 20, 20
        col2_features = input_features[:, 1:8:3, :, :, :]  # N, 3, 64, 20, 20
        col3_pre = input_features[:, 2:8:3, :, :, :].unsqueeze(1).expand(N, K0, 2, C, H,
                                                                         W)  # N, 2, 64, 20, 20 -> N, 1, 2, 64, 20, 20 -> N, 8, 2, 64, 20, 20
        col3_features = torch.cat((col3_pre, choices_features), dim=2).view(N * K0, 3, C, H, W)

        return row1_features, row2_features, row3_features, col1_features, col2_features, col3_features

    def apply_reduce(self, x1, x2, x3):

        # dist12, dist23, dist31 = (x1 - x2).pow(2), (x2 - x3).pow(2), (x3 - x1).pow(2)
        # x = 1 - (dist12 + dist23 + dist31)

        B, _, C, H, W = x1.shape
        fusion = element_wise_attention((B, C, H, W))
        # fusion = torch.nn.DataParallel(fusion)
        fusion = fusion.cuda()
        x_12 = fusion(x1.reshape(-1,C,H,W), x2.reshape(-1,C,H,W))
        x_23 = fusion(x2.reshape(-1,C,H,W), x3.reshape(-1,C,H,W))
        x_13 = fusion(x1.reshape(-1,C,H,W), x3.reshape(-1,C,H,W))

        
        x_cat = torch.cat((x_12,x_23,x_13),dim=1)

        B, C, H, W = x_cat.shape
        x_cat = x_cat.view(-1,C,H,W)
        if C==64*3:
            x_out = self.F_func_high(x_cat)
        if C==128*3:
            x_out = self.F_func_mid(x_cat)
        if C==256*3:
            x_out = self.F_func_low(x_cat)
        x = x_out.view(B//8, 8, C//3, H, W)

        return x

    def reduce(self, row_features, col_features, N, K0):
        _, C, H, W = row_features.shape
        row1 = row_features[:N, :, :, :].unsqueeze(1).expand(N, K0, C, H, W)
        row2 = row_features[N:2 * N, :, :, :].unsqueeze(1).expand(N, K0, C, H, W)
        row3 = row_features[2 * N:, :, :, :].view(N, K0, C, H, W)

        final_row_features = self.apply_reduce(row1, row2, row3)

        col1 = col_features[:N, :, :, :].unsqueeze(1).expand(N, K0, C, H, W)
        col2 = col_features[N:2 * N, :, :, :].unsqueeze(1).expand(N, K0, C, H, W)
        col3 = col_features[2 * N:, :, :, :].view(N, K0, C, H, W)

        final_col_features = self.apply_reduce(col1, col2, col3)

        input_features = final_row_features + final_col_features
        return input_features

    def forward(self, x):
        N, K, H, W = x.shape
        K0 = K - 8
        # assert C==1
        x = x.view(-1, K, 80, 80)

        ### Perception Branch
        input_features_high = self.perception_net_high(x.view(-1, 80, 80).unsqueeze(1))
        input_features_mid = self.perception_net_mid(input_features_high)
        input_features_low = self.perception_net_low(input_features_mid)
        # ((32*16),64,20,20), ((32*16),128,5,5), ((32*16),256,1,1)

        transformer_features = self.swinT(x)

        input_features_high = input_features_high + transformer_features[0].view(-1, 64,20,20) #(32*16,64,20,20)
        input_features_mid = input_features_mid + transformer_features[1].view(-1, 128,10,10) #(32*16,128,10,10)
        input_features_low = input_features_low + transformer_features[2].view(-1, 256,5,5) #(32*16,256,5,5)

        # input_features_high = s_x[0].view(-1, 64,20,20) #(32*16,64,20,20)
        # input_features_mid = s_x[1].view(-1, 128,10,10) #(32*16,128,10,10)
        # input_features_low = s_x[2].view(-1, 256,5,5) #(32*16,256,5,5)

        
        ### Relation Module
        # High res
        if self.levels[0] == '1':
            row1_cat_high, row2_cat_high, row3_cat_high, col1_cat_high, col2_cat_high, col3_cat_high = \
                self.triples(input_features_high.view(N, K, self.high_dim, 20, 20))

            row_feats_high = self.g_function_high(torch.cat((row1_cat_high, row2_cat_high, row3_cat_high), dim=0))
            row_feats_high = self.bn_row_high(self.conv_row_high(row_feats_high))
            col_feats_high = self.g_function_high(torch.cat((col1_cat_high, col2_cat_high, col3_cat_high), dim=0))
            col_feats_high = self.bn_col_high(self.conv_col_high(col_feats_high))

            reduced_feats_high = self.reduce(row_feats_high, col_feats_high, N, K0)  # N, 8, 64, 20, 20

        # Mid res
        if self.levels[1] == '1':
            row1_cat_mid, row2_cat_mid, row3_cat_mid, col1_cat_mid, col2_cat_mid, col3_cat_mid = \
                self.triples(input_features_mid.view(N, K, self.mid_dim, 10, 10))

            row_feats_mid = self.g_function_mid(torch.cat((row1_cat_mid, row2_cat_mid, row3_cat_mid), dim=0))
            row_feats_mid = self.bn_row_mid(self.conv_row_mid(row_feats_mid))
            col_feats_mid = self.g_function_mid(torch.cat((col1_cat_mid, col2_cat_mid, col3_cat_mid), dim=0))
            col_feats_mid = self.bn_col_mid(self.conv_col_mid(col_feats_mid))

            reduced_feats_mid = self.reduce(row_feats_mid, col_feats_mid, N, K0)  # N, 8, 256, 5, 5
        # Low res
        if self.levels[2] == '1':
            row1_cat_low, row2_cat_low, row3_cat_low, col1_cat_low, col2_cat_low, col3_cat_low = \
                self.triples(input_features_low.view(N, K, self.low_dim, 5, 5))

            row_feats_low = self.g_function_low(torch.cat((row1_cat_low, row2_cat_low, row3_cat_low), dim=0))
            row_feats_low = self.bn_row_low(self.conv_row_low(row_feats_low))
            col_feats_low = self.g_function_low(torch.cat((col1_cat_low, col2_cat_low, col3_cat_low), dim=0))
            col_feats_low = self.bn_col_low(self.conv_col_low(col_feats_low))

            reduced_feats_low = self.reduce(row_feats_low, col_feats_low, N, K0)  # N, 8, 256, 5, 5

        ### Combine
        self.final_high = self.final_mid = self.final_low = None
        final = []
        # High
        if self.levels[0] == '1':
            res1_in_high = reduced_feats_high
            if self.do_contrast:
                res1_in_high = res1_in_high - res1_in_high.mean(dim=1).unsqueeze(1)
            res1_out_high = self.res1_high(res1_in_high.view(N * K0, self.high_dim, 20, 20))
            res2_in_high = res1_out_high.view(N, K0, 2 * self.high_dim, 10, 10)
            if self.do_contrast:
                res2_in_high = res2_in_high - res2_in_high.mean(dim=1).unsqueeze(1)
            out_high = self.res2_high(res2_in_high.view(N * K0, 2 * self.high_dim, 10, 10))
            final_high = self.avgpool(out_high)
            final_high = final_high.view(-1, self.mlp_dim_high)
            final.append(final_high)
            self.final_high = final_high

        # Mid
        if self.levels[1] == '1':
            res1_in_mid = reduced_feats_mid
            if self.do_contrast:
                res1_in_mid = res1_in_mid - res1_in_mid.mean(dim=1).unsqueeze(1)
            res1_out_mid = self.res1_mid(res1_in_mid.view(N * K0, self.mid_dim, 10, 10))
            res2_in_mid = res1_out_mid.view(N, K0, 2 * self.mid_dim, 5, 5)
            if self.do_contrast:
                res2_in_mid = res2_in_mid - res2_in_mid.mean(dim=1).unsqueeze(1)
            out_mid = self.res2_mid(res2_in_mid.view(N * K0, 2 * self.mid_dim, 5, 5))
            final_mid = self.avgpool(out_mid)
            final_mid = final_mid.view(-1, self.mlp_dim_mid)
            final.append(final_mid)
            self.final_mid = final_mid

        # Low
        if self.levels[2] == '1':
            res1_in_low = reduced_feats_low
            if self.do_contrast:
                res1_in_low = res1_in_low - res1_in_low.mean(dim=1).unsqueeze(1)
            res1_out_low = self.res1_low(res1_in_low.view(N * K0, self.low_dim, 5, 5))
            res2_in_low = res1_out_low.view(N, K0, self.mlp_dim_low, 5, 5)
            if self.do_contrast:
                res2_in_low = res2_in_low - res2_in_low.mean(dim=1).unsqueeze(1)
            out_low = self.res2_low(res2_in_low.view(N * K0, self.mlp_dim_low, 5, 5))
            final_low = self.avgpool(out_low)
            final_low = final_low.view(-1, self.mlp_dim_low)
            final.append(final_low)
            self.final_low = final_low

        final = torch.cat(final, dim=1) # 32,8,128

        # MLP
        out = self.mlp(final)

        return out.view(-1, K0)
