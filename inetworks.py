import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False),
            nn.LogSigmoid()
        )


    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return out  # (N, C, 1, 1)


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        # 2 channels input (input feature map and attention map), 1 channel output
        self.conv1 = nn.Conv2d(2, 1, kernel_size=7, padding=3) 
        self.sigmoid = nn.LogSigmoid()


    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True) # Calculate average along channels
        max_out, _ = torch.max(x, dim=1, keepdim=True) # Calculate max along channels
        x = torch.cat([avg_out, max_out], dim=1) # Concatenate average and max along channels
        x = self.conv1(x)
        x = self.sigmoid(x)  # (N, 1, H, W)
        return x


# Sliding window function
class SequenceWindows(nn.Module):
    def __init__(self, window_size=1000, window_stride=200, shuffle=0):
        super(SequenceWindows, self).__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.window_size = window_size
        self.window_stride = window_stride
        self.shuffle = shuffle


    def forward(self, x, labels):
        num_samples, channels, h, w = x.size()
        # Number of new samples generated per trial via sliding window
        num_new_samples =int((w-self.window_size)/self.window_stride+1) 
        # Number of new samples after applying a sliding window
        all_samples = num_samples*num_new_samples  
        new_samples =  torch.zeros((all_samples, channels, h, self.window_size)).to(self.device)
        new_labels = torch.zeros((all_samples)).to(self.device)
        sorted_permutation = None 
        if self.shuffle == 0:  # Does not disrupt the order of samples within the window
            for i in range(num_samples):  # Iteration sample size 
                for j in range(num_new_samples):
                    start_point = j*self.window_stride
                    end_point = j*self.window_stride+self.window_size
                    temp_x = x[i, :, :, start_point:end_point]
                    new_samples[i*num_new_samples+j, :, :, :] = temp_x
                    new_labels[i*num_new_samples+j] = labels[i]
        elif self.shuffle == 1:  # Disrupt the order of the samples in the window
            for i in range(num_samples):  
                temp_x = torch.zeros((num_new_samples, channels, h, self.window_size))
                for j in range(num_new_samples):
                    start_point = j*self.window_stride
                    end_point = j*self.window_stride+self.window_size
                    temp_x[j,:,:,:] = x[i, :, :, start_point:end_point]
                start_index = i*num_new_samples   # Starging index
                end_index = (i+1)*num_new_samples # Finishing index 
                np.random.seed(1000)
                permutation = list(np.random.permutation(num_new_samples))  
                temp_x = temp_x[permutation]
                new_samples[start_index:end_index, :, :, :] = temp_x
                new_labels[start_index:end_index] = labels[i]
        elif self.shuffle == 2:  # Disrupting the order of batch samples
            for i in range(num_samples):  # 遍历样本数
                for j in range(num_new_samples):
                    start_point = j*self.window_stride
                    end_point = j*self.window_stride+self.window_size
                    temp_x = x[i, :, :, start_point:end_point]
                    new_samples[i*num_new_samples+j, :, :, :] = temp_x
                    new_labels[i*num_new_samples+j] = labels[i]
            np.random.seed(10000)
            new_nums = new_samples.shape[0]  
            permutation = list(np.random.permutation(new_nums))  # 打乱顺序
            new_samples = new_samples[permutation]
            new_labels = new_labels[permutation]
            # Get the original order of the permutation, and sort the array back to the original order.
            sorted_permutation = np.argsort(permutation)
        return new_samples, new_labels, self.shuffle, sorted_permutation
    

class TemporalAttention(nn.Module):
    def __init__(self, in_channels=64, n_segment=1, window_size=1000, window_stride=200, 
                shuffle=0, kernel_size=3, stride=1, padding=1):
        super(TemporalAttention, self).__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.window_size = window_size
        self.n_segment = n_segment
        
        self.sequenceWindow = SequenceWindows(window_size=self.window_size, window_stride=window_stride, shuffle=shuffle)

        self.GlobalBranch= nn.Sequential(
            nn.Linear(n_segment, n_segment * 2, bias=False),
            nn.BatchNorm1d(n_segment * 2), 
            nn.ReLU(inplace=True),
            nn.Linear(n_segment * 2, kernel_size, bias=False), 
            nn.Softmax(-1))

        self.LocalBranch = nn.Sequential(
            nn.Conv1d(in_channels, in_channels // 4, kernel_size, stride=1, 
                      padding=kernel_size // 2, bias=False), 
            nn.BatchNorm1d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels // 4, in_channels, 1, bias=False),
            nn.Sigmoid())


    def forward(self, x, labels):
        x, new_labels, shuffle, sorted_permu = self.sequenceWindow(x, labels)  # (N, channels, new_samples, 1000)
        nt, c, h, w = x.size()
        t = self.n_segment
        n_batch = nt // t
        new_x = x.view(n_batch, t, c, h, w).permute(0, 2, 1, 3, 4).contiguous()
        out = F.adaptive_avg_pool2d(new_x.view(n_batch * c, t, h, w), (1, 1))
        out = out.view(-1, t)
        conv_kernel = self.GlobalBranch(out.view(-1, t)).view(n_batch * c, 1, -1, 1)
        local_activation = self.LocalBranch(out.view(n_batch, c, t)).view(n_batch, c, t, 1, 1)
        new_x = new_x * local_activation

        out = F.conv2d(new_x.view(1, n_batch * c, t, h * w), conv_kernel, bias=None,
                       stride=(self.stride, 1), padding=(self.padding, 0), groups=n_batch * c)
        
        out = out.view(n_batch, c, t, h, w)
        out = out.permute(0, 2, 1, 3, 4).contiguous().view(nt, c, h, w)
        out = out + x  #  

        return out, new_labels, shuffle, sorted_permu


class MACNet(nn.Module):
    def __init__(self, nClass=3, dropout=0.25):
        super(MACNet, self).__init__()
        self.nClass = nClass
        
        self.convBlock1 = self.ConvBlock1()
        self.ftam = self.FTAM(in_channels=32, n_segment=16, window_size=1000, shuffle=2)
        self.convBlock2 = self.ConvBlock2(dropoutP=dropout)
        self.branchAvgpool = self.BranchAvgpool()
        self.branchConvBlock2 = self.BranchConvBlock2()
        self.branchConvBlock3 = self.BranchConvBlock3()
        self.fcam = self.FCAM(in_channels=384)
        self.fsam = self.FSAM()
        self.convBlock3 = self.ConvBlock3()
        self.fc = self.FC()


    # Temporal Feature 
    def ConvBlock1(self):
        return nn.Sequential(
            # (N, 1, 3, 4000)
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            # (N, 32, 3, 4000)
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )


    def ConvBlock2(self, dropoutP):
        Block1 = nn.Sequential(
            # (N*T, 32, 3, 1000) 
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            # (N*T, 64, 3, 1000)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Dropout(p=dropoutP)
            # (N*T, 64, 3, 500)
        )

        Block2 = nn.Sequential(
            # (N*T, 64, 3, 500)
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            # (N*T, 128, 3, 500)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Dropout(p=dropoutP)
            # (N*T, 64, 3, 250)
        )

        return nn.Sequential(Block1, Block2)


    # Multi Branch
    def BranchAvgpool(self):
        return nn.Sequential(
            nn.AdaptiveAvgPool2d((3, 4))
        )

    
    def BranchConvBlock2(self):
        return nn.Sequential(
            # (N, 128, 3, 250)
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 1), padding=(0, 1), stride=1),
            # (N, 128, 3, 250) 
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((3, 4))
            # (N, 128, 3, 4)
        )


    def BranchConvBlock3(self):
        return nn.Sequential(
            # (N, 128, 3, 250)
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=(1, 1), stride=1),
            # (N, 128, 3, 250)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((3, 4))
            # (N, 128, 3, 4)
        )
        

    # Attention
    def FCAM(self, in_channels, reduction_ratio=16):
        return ChannelAttention(in_channels, reduction_ratio)


    def FSAM(self):
        return SpatialAttention()
    
    
    def FTAM(self, in_channels, n_segment, window_size, shuffle):
        return TemporalAttention(in_channels=in_channels, n_segment=n_segment, 
                                window_size=window_size, shuffle=shuffle)


    def ConvBlock3(self):
        return nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=(1, 1), stride=1),
            # (N, 384, 3, 4)
            nn.BatchNorm2d(384),
            nn.ReLU(),
        )
    

    def FC(self):
        return nn.Sequential(
            # (N, 384, 3, 4)
            nn.Dropout(p=0.25),
            nn.Linear(384*3*4, 256),
            nn.Dropout(p=0.25),
            nn.Linear(256, 128),
            nn.Linear(128, self.nClass),
        )


    def forward(self, x, labels):
        x_temporal1 = self.convBlock1(x)        # (N, 128, 3, 1000) 
        x_temporal2, new_labels, shuffle, sorted_permu = self.ftam(x_temporal1, labels)
        x_temporal = self.convBlock2(x_temporal2)      # (N, 128, 3, 1000) 
        x_branch1 = self.branchAvgpool(x_temporal)     # (N, 128, 3, 4) 
        x_branch2 = self.branchConvBlock2(x_temporal)  # (N, 128, 3, 4)
        x_branch3 = self.branchConvBlock3(x_temporal)  # (N, 128, 3, 4)
        x_concat = torch.cat((x_branch1, x_branch2, x_branch3), dim=1)  # (N, 384, 4, 4)
        x_fcam = self.fcam(x_concat)
        fcam_multiply = x_concat * x_fcam
        x_fsam = self.fsam(fcam_multiply)
        fsam_multiply = fcam_multiply * x_fsam
        x_conv = self.convBlock3(fsam_multiply)
        x_conv = x_conv.view(-1, 384*3*4)  
        out = self.fc(x_conv) 
        # if isinstance(sorted_permu, (int, float))==False:
        if shuffle == 2:
            out = out[sorted_permu] 
            new_labels = new_labels[sorted_permu]
        return out, new_labels
    

if __name__ == '__main__':
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # batch size: 2, sEMG size: (3, 4000), 3 sEMG channels, 4000 timepoints
    input_tensor = torch.randn(2, 1, 3, 4000)
    labels_tensor = torch.Tensor([1, 0])
    input_tensor = input_tensor.to(DEVICE)
    labels_tensor = labels_tensor.type(torch.LongTensor).to(DEVICE)

    print('input_tensor.shape', input_tensor.shape)
    print('labels: ', labels_tensor)

    net = MACNet(nClass=3)
    net.to(DEVICE)
    output_tensor, out_labels = net(input_tensor, labels_tensor)

    print(output_tensor.shape)  
    print(out_labels.shape)  


