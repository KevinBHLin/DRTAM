import torch
import torch.nn as nn
from mmcv.cnn import (PLUGIN_LAYERS, Conv2d, ConvModule, caffe2_xavier_init,
                      normal_init, xavier_init)
import torch.nn.functional as F

class AvgChunkPool(nn.Module):

    def __init__(self, chunk=3, dim = 2):
        super(AvgChunkPool, self).__init__()    
        self.dim = dim
        self.chunk = chunk
        assert (dim ==2 or dim == 3)
        if dim ==2:
            self.pool = nn.AdaptiveAvgPool2d((1, None))
        if dim ==3:
            self.pool = nn.AdaptiveAvgPool2d((None, 1))

    def forward(self, x): 
        x_chunks = torch.chunk(x, chunks=self.chunk, dim=self.dim)
        x_chunks_pool = [self.pool( x_chunk ) for x_chunk in x_chunks]
        x_chunks_pool = torch.cat(x_chunks_pool, dim=self.dim)

        return x_chunks_pool

class AvgStripPool(nn.Module):
    def __init__(self, chunk=3, dim = 2):
        super(AvgStripPool, self).__init__()    
        self.dim = dim
        self.chunk = 1
        assert (dim ==2 or dim == 3)
        if dim ==2:
            self.pool = nn.AdaptiveAvgPool2d((1, None))
        if dim ==3:
            self.pool = nn.AdaptiveAvgPool2d((None, 1))

    def forward(self, x): 
        x_strip_pool = self.pool( x )

        return x_strip_pool

class MaxChunkPool(nn.Module):
    def __init__(self, chunk=3, dim = 2):
        super(MaxChunkPool, self).__init__()    
        self.dim = dim
        self.chunk = chunk
        assert (dim ==2 or dim == 3)
        if dim ==2:
            self.pool = nn.AdaptiveMaxPool2d((1, None))
        if dim ==3:
            self.pool = nn.AdaptiveMaxPool2d((None, 1))

    def forward(self, x): 
        x_chunks = torch.chunk(x, chunks=self.chunk, dim=self.dim)
        x_chunks_pool = [self.pool( x_chunk ) for x_chunk in x_chunks]
        x_chunks_pool = torch.cat(x_chunks_pool, dim=self.dim)

        return x_chunks_pool
    
class MaxStripPool(nn.Module):
    def __init__(self, chunk=3, dim = 2):
        super(MaxStripPool, self).__init__()    
        self.dim = dim
        self.chunk = 1
        assert (dim ==2 or dim == 3)
        if dim ==2:
            self.pool = nn.AdaptiveMaxPool2d((1, None))
        if dim ==3:
            self.pool = nn.AdaptiveMaxPool2d((None, 1))

    def forward(self, x): 
        x_strip_pool = self.pool( x )

        return x_strip_pool

DRTAMPoolModule = {
    'AvgChunkPool': AvgChunkPool,
    'AvgStripPool': AvgStripPool,
    'MaxChunkPool': MaxChunkPool,
    'MaxStripPool': MaxStripPool
}

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
    
class ChannelEmbeddingFilter(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=32):
        super(ChannelEmbeddingFilter, self).__init__()
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
    def forward(self, x):
        c = self.mlp(x).unsqueeze(2).unsqueeze(3)
        c = torch.sigmoid(c)
        return c

class ChannelEncoder(nn.Module):
    def __init__(self, hw_chunk=3, c_pool_list=['AvgChunkPool','MaxChunkPool']):
        super(ChannelEncoder, self).__init__()    

        self.m_PoolLayers = nn.ModuleList()
        poolnum = 0
        for poolname in c_pool_list:
            self.m_PoolLayers.append( DRTAMPoolModule[poolname](chunk = hw_chunk, dim = 2) )
            poolnum = poolnum + self.m_PoolLayers[-1].chunk

        self.c_conv1d = nn.Conv1d(poolnum*2, 1, kernel_size=1, padding=0, bias=False)
        self.bn_c = nn.BatchNorm2d(1)

    def forward(self, x_hc, x_wc): 
        
        #x_hc = x_hc.squeeze(1) #b*h*c
        #x_wc = x_wc.squeeze(1) #b*w*c

        x_hc_pool = [net(x_hc) for net in self.m_PoolLayers] 
        x_wc_pool = [net(x_wc) for net in self.m_PoolLayers]       
        c = torch.cat(x_hc_pool + x_wc_pool, dim=2).squeeze(1) #b*poolnum*c
        c = self.c_conv1d(c).unsqueeze(-1)
        c = self.bn_c(c)     
                     
        return c 


class SpatialEncoder(nn.Module):
    def __init__(self, hw_chunk = 3, c_chunk = 1, 
                 hw_pool_list=['AvgChunkPool'],
                 hwc_pool_list=['AvgChunkPool','MaxChunkPool']):
        super(SpatialEncoder, self).__init__()    

        self.m_hPoolLayers = nn.ModuleList()
        self.m_wPoolLayers = nn.ModuleList()
        wpoolnum = 0
        hpoolnum = 0
        for poolname in hw_pool_list:
            self.m_hPoolLayers.append( DRTAMPoolModule[poolname](chunk = hw_chunk, dim = 2) )
            hpoolnum = hpoolnum +  self.m_hPoolLayers[-1].chunk

            self.m_wPoolLayers.append( DRTAMPoolModule[poolname](chunk = hw_chunk, dim = 3) )
            wpoolnum = wpoolnum + self.m_wPoolLayers[-1].chunk

        self.m_cPoolLayers = nn.ModuleList()
        cpoolnum = 0
        for poolname in hwc_pool_list: 
            self.m_cPoolLayers.append( DRTAMPoolModule[poolname](chunk = c_chunk, dim = 3) )
            cpoolnum = cpoolnum +  self.m_cPoolLayers[-1].chunk
    
        self.h_conv1d = nn.Conv1d(wpoolnum+cpoolnum, 1, kernel_size=3, padding=1, bias=False)
        self.w_conv1d = nn.Conv1d(hpoolnum+cpoolnum, 1, kernel_size=3, padding=1, bias=False)
        self.bn_h = nn.BatchNorm2d(1)
        self.bn_w = nn.BatchNorm2d(1)

    def forward(self, x_hw, x_hc, x_wc):
        x_hw_hpool = [net(x_hw) for net in self.m_hPoolLayers] 
        x_hw_wpool = [net(x_hw) for net in self.m_wPoolLayers] 

        x_hc_cpool = [net(x_hc) for net in self.m_cPoolLayers] 
        x_wc_cpool = [net(x_wc).permute(0,1,3,2) for net in self.m_cPoolLayers] 

        h = torch.cat(x_hc_cpool+x_hw_wpool, dim=3)
        w = torch.cat(x_wc_cpool+x_hw_hpool, dim=2) 
                                
        h = h.squeeze(1).permute(0,2,1)
        h = self.h_conv1d(h).unsqueeze(-1)
        h = self.bn_h(h)
        a_h = torch.sigmoid(h)
        
        w = w.squeeze(1)
        w = self.w_conv1d(w).unsqueeze(2)
        w = self.bn_w(w)
        a_w = torch.sigmoid(w)
        
        return a_h, a_w

@PLUGIN_LAYERS.register_module()
class DRTAMLinv3(nn.Module):
    def __init__(self, in_channels, hw_chunk = 3, c_chunk = 1, reduction_ratio=32, dual_artecture = True,
                 c_pool_list=['AvgChunkPool','MaxChunkPool'],
                 hw_pool_list=['AvgChunkPool'],
                 hwc_pool_list=['AvgChunkPool','MaxChunkPool']
                 ):
        super(DRTAMLinv3, self).__init__()

        self.pool_w = nn.AdaptiveAvgPool2d((None, 1)) #池化得到一个侧面
        self.pool_h = nn.AdaptiveAvgPool2d((1, None)) #池化得到一个侧面
        self.pool_c = nn.AdaptiveAvgPool3d((1,None, None))

        self.ipa_channel_encoder = ChannelEncoder(hw_chunk = hw_chunk, c_pool_list = c_pool_list) 
        self.channel_embedding_filter = ChannelEmbeddingFilter(in_channels, reduction_ratio)
        self.ipa_spatial_encoder = SpatialEncoder(hw_chunk = hw_chunk, c_chunk = c_chunk, hw_pool_list = hw_pool_list, hwc_pool_list = hwc_pool_list)
        
        if dual_artecture:
            self.cpa_channel_encoder = ChannelEncoder(hw_chunk = hw_chunk, c_pool_list = c_pool_list)
            self.cpa_spatial_encoder = SpatialEncoder(hw_chunk = hw_chunk, c_chunk = c_chunk, hw_pool_list = hw_pool_list, hwc_pool_list = hwc_pool_list)

        self.dual_artecture = dual_artecture
 
    def forward(self, x):
        identity = x      

        x_hw = self.pool_c(x)                    #b*1*h*w
        x_hc = self.pool_w(x).permute(0,3,2,1)   #b*1*h*c   
        x_wc = self.pool_h(x).permute(0,2,3,1)   #b*1*w*c
        
        ipa_c = self.ipa_channel_encoder(x_hc,x_wc)  
        ipa_c = self.channel_embedding_filter(ipa_c)
        
        ipa_h, ipa_w = self.ipa_spatial_encoder(x_hw, x_hc, x_wc)
        
        ipa_sa = ipa_h * ipa_w #便于后续正面池化块乘积出cpa
        ipa_tensor = ipa_c * ipa_sa
        
        if self.dual_artecture:
            x_hc_cpa = x_hc * (1-(ipa_c * ipa_h).permute(0,3,2,1))
            x_wc_cpa = x_wc * (1-(ipa_c * ipa_w).permute(0,2,3,1))
            x_hw_cpa = x_hw * (1-ipa_sa) #两个池化面不分开写，tensor乘法计算更快点
            
            cpa_c = self.cpa_channel_encoder(x_hc_cpa, x_wc_cpa)  
            cpa_c = self.channel_embedding_filter(cpa_c)
            
            cpa_h, cpa_w = self.cpa_spatial_encoder(x_hw_cpa, x_hc_cpa, x_wc_cpa)
            cpa_tensor = cpa_c * cpa_h * cpa_w  
            
            tensor = ipa_tensor + (1 - ipa_tensor) * cpa_tensor
        else:
            tensor = ipa_tensor

        out = identity * tensor
       
        return out

if __name__ == "__main__":
    net = DRTAMLinv3(512,3,1,32)
    net2 = DRTAMLinv3(512,3,1,32)
    net.cuda()
    net.eval()
    net2.cuda()
    net2.eval()
    input = torch.randn(3, 512, 56, 56).cuda()
    out = net(input)
    out2= net2(input)
    print(torch.sum(torch.abs(out-out2)))
    #print(torch.sum(torch.abs(x_hw-x_hw2)))
    #print(torch.sum(torch.abs(x_hc-x_hc2)))
    #print(torch.sum(torch.abs(x_wc-x_wc2)))
