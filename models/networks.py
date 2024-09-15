import torch
import torch.nn as nn
from torch.nn import init
import functools, itertools
import numpy as np
from util.util import gkern_2d
import torch.nn.functional as F
from pytorch_msssim import SSIM
from torchvision import models
import math
import skimage
from skimage import measure
from kmeans_pytorch import kmeans



def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

# Spectral normalization base class 
# Projection of x onto y
def proj(x, y):
  return torch.mm(y, x.t()) * y / torch.mm(y, y.t())


# Orthogonalize x wrt list of vectors ys
def gram_schmidt(x, ys):
  for y in ys:
    x = x - proj(x, y)
  return x


# Apply num_itrs steps of the power method to estimate top N singular values.
def power_iteration(W, u_, update=True, eps=1e-12):
  # Lists holding singular vectors and values
  us, vs, svs = [], [], []
  for i, u in enumerate(u_):
    # Run one step of the power iteration
    with torch.no_grad():
      v = torch.matmul(u, W)
      # Run Gram-Schmidt to subtract components of all other singular vectors
      v = F.normalize(gram_schmidt(v, vs), eps=eps)
      # Add to the list
      vs += [v]
      # Update the other singular vector
      u = torch.matmul(v, W.t())
      # Run Gram-Schmidt to subtract components of all other singular vectors
      u = F.normalize(gram_schmidt(u, us), eps=eps)
      # Add to the list
      us += [u]
      if update:
        u_[i][:] = u
    # Compute this singular value and add it to the list
    svs += [torch.squeeze(torch.matmul(torch.matmul(v, W.t()), u.t()))]
    #svs += [torch.sum(F.linear(u, W.transpose(0, 1)) * v)]
  return svs, us, vs


# Convenience passthrough function
class identity(nn.Module):
  def forward(self, input):
    return input
 

class SN(object):
  def __init__(self, num_svs, num_itrs, num_outputs, transpose=False, eps=1e-12):
    # Number of power iterations per step
    self.num_itrs = num_itrs
    # Number of singular values
    self.num_svs = num_svs
    # Transposed?
    self.transpose = transpose
    # Epsilon value for avoiding divide-by-0
    self.eps = eps
    # Register a singular vector for each sv
    for i in range(self.num_svs):
      self.register_buffer('u%d' % i, torch.randn(1, num_outputs))
      self.register_buffer('sv%d' % i, torch.ones(1))
  
  # Singular vectors (u side)
  @property
  def u(self):
    return [getattr(self, 'u%d' % i) for i in range(self.num_svs)]

  # Singular values; 
  # note that these buffers are just for logging and are not used in training. 
  @property
  def sv(self):
   return [getattr(self, 'sv%d' % i) for i in range(self.num_svs)]
   
  # Compute the spectrally-normalized weight
  def W_(self):
    W_mat = self.weight.view(self.weight.size(0), -1)
    if self.transpose:
      W_mat = W_mat.t()
    # Apply num_itrs power iterations
    for _ in range(self.num_itrs):
      svs, us, vs = power_iteration(W_mat, self.u, update=self.training, eps=self.eps) 
    # Update the svs
    if self.training:
      with torch.no_grad(): # Make sure to do this in a no_grad() context or you'll get memory leaks!
        for i, sv in enumerate(svs):
          self.sv[i][:] = sv     
    return self.weight / svs[0]


# 2D Conv layer with spectral norm
class SNConv2d(nn.Conv2d, SN):
  def __init__(self, in_channels, out_channels, kernel_size, stride=1,
             padding=0, dilation=1, groups=1, bias=True, 
             num_svs=1, num_itrs=1, eps=1e-12):
    nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, stride, 
                     padding, dilation, groups, bias)
    SN.__init__(self, num_svs, num_itrs, out_channels, eps=eps)    
  def forward(self, x):
    return F.conv2d(x, self.W_(), self.bias, self.stride, 
                    self.padding, self.dilation, self.groups)


# Linear layer with spectral norm
class SNLinear(nn.Linear, SN):
  def __init__(self, in_features, out_features, bias=True,
               num_svs=1, num_itrs=1, eps=1e-12):
    nn.Linear.__init__(self, in_features, out_features, bias)
    SN.__init__(self, num_svs, num_itrs, out_features, eps=eps)
  def forward(self, x):
    return F.linear(x, self.W_(), self.bias)

################################SN#######################

#######Positional Encoding module, borrowed from https://github.com/open-mmlab/mmgeneration
class SinusoidalPositionalEmbedding(nn.Module):
    """Sinusoidal Positional Embedding 1D or 2D (SPE/SPE2d).
    This module is a modified from:
    https://github.com/pytorch/fairseq/blob/master/fairseq/modules/sinusoidal_positional_embedding.py # noqa
    Based on the original SPE in single dimension, we implement a 2D sinusoidal
    positional encodding (SPE2d), as introduced in Positional Encoding as
    Spatial Inductive Bias in GANs, CVPR'2021.
    Args:
        embedding_dim (int): The number of dimensions for the positional
            encoding.
        padding_idx (int | list[int]): The index for the padding contents. The
            padding positions will obtain an encoding vector filling in zeros.
        init_size (int, optional): The initial size of the positional buffer.
            Defaults to 1024.
        div_half_dim (bool, optional): If true, the embedding will be divided
            by :math:`d/2`. Otherwise, it will be divided by
            :math:`(d/2 -1)`. Defaults to False.
        center_shift (int | None, optional): Shift the center point to some
            index. Defaults to None.
    """

    def __init__(self,
                 embedding_dim,
                 padding_idx,
                 init_size=1024,
                 div_half_dim=False,
                 center_shift=None):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.div_half_dim = div_half_dim
        self.center_shift = center_shift

        self.weights = SinusoidalPositionalEmbedding.get_embedding(
            init_size, embedding_dim, padding_idx, self.div_half_dim)

        self.register_buffer('_float_tensor', torch.FloatTensor(1))

        self.max_positions = int(1e5)

    @staticmethod
    def get_embedding(num_embeddings,
                      embedding_dim,
                      padding_idx=None,
                      div_half_dim=False):
        """Build sinusoidal embeddings.
        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        assert embedding_dim % 2 == 0, (
            'In this version, we request '
            f'embedding_dim divisible by 2 but got {embedding_dim}')

        # there is a little difference from the original paper.
        half_dim = embedding_dim // 2
        if not div_half_dim:
            emb = np.log(10000) / (half_dim - 1)
        else:
            emb = np.log(1e4) / half_dim
        # compute exp(-log10000 / d * i)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(
            num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)],
                        dim=1).view(num_embeddings, -1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0

        return emb

    def forward(self, input, **kwargs):
        """Input is expected to be of size [bsz x seqlen].
        Returned tensor is expected to be of size  [bsz x seq_len x emb_dim]
        """
        assert input.dim() == 2 or input.dim(
        ) == 4, 'Input dimension should be 2 (1D) or 4(2D)'

        if input.dim() == 4:
            return self.make_grid2d_like(input, **kwargs)

        b, seq_len = input.shape
        max_pos = self.padding_idx + 1 + seq_len

        if self.weights is None or max_pos > self.weights.size(0):
            # recompute/expand embedding if needed
            self.weights = SinusoidalPositionalEmbedding.get_embedding(
                max_pos, self.embedding_dim, self.padding_idx)
        self.weights = self.weights.to(self._float_tensor)

        positions = self.make_positions(input, self.padding_idx).to(
            self._float_tensor.device)

        return self.weights.index_select(0, positions.view(-1)).view(
            b, seq_len, self.embedding_dim).detach()

    def make_positions(self, input, padding_idx):
        mask = input.ne(padding_idx).int()
        return (torch.cumsum(mask, dim=1).type_as(mask) *
                mask).long() + padding_idx

    def make_grid2d(self, height, width, num_batches=1, center_shift=None):
        h, w = height, width
        # if `center_shift` is not given from the outside, use
        # `self.center_shift`
        if center_shift is None:
            center_shift = self.center_shift

        h_shift = 0
        w_shift = 0
        # center shift to the input grid
        if center_shift is not None:
            # if h/w is even, the left center should be aligned with
            # center shift
            if h % 2 == 0:
                h_left_center = h // 2
                h_shift = center_shift - h_left_center
            else:
                h_center = h // 2 + 1
                h_shift = center_shift - h_center

            if w % 2 == 0:
                w_left_center = w // 2
                w_shift = center_shift - w_left_center
            else:
                w_center = w // 2 + 1
                w_shift = center_shift - w_center

        # Note that the index is started from 1 since zero will be padding idx.
        # axis -- (b, h or w)
        x_axis = torch.arange(1, w + 1).unsqueeze(0).repeat(num_batches,
                                                            1) + w_shift
        y_axis = torch.arange(1, h + 1).unsqueeze(0).repeat(num_batches,
                                                            1) + h_shift

        # emb -- (b, emb_dim, h or w)
        x_emb = self(x_axis).transpose(1, 2)
        y_emb = self(y_axis).transpose(1, 2)

        # make grid for x/y axis
        # Note that repeat will copy data. If use learned emb, expand may be
        # better.
        x_grid = x_emb.unsqueeze(2).repeat(1, 1, h, 1)
        y_grid = y_emb.unsqueeze(3).repeat(1, 1, 1, w)

        # cat grid -- (b, 2 x emb_dim, h, w)
        grid = torch.cat([x_grid, y_grid], dim=1)
        return grid.detach()

    def make_grid2d_like(self, x, center_shift=None):
        """Input tensor with shape of (b, ..., h, w) Return tensor with shape
        of (b, 2 x emb_dim, h, w)
        Note that the positional embedding highly depends on the the function,
        ``make_positions``.
        """
        h, w = x.shape[-2:]

        grid = self.make_grid2d(h, w, x.size(0), center_shift)

        return grid.to(x)


class CatersianGrid(nn.Module):
    """Catersian Grid for 2d tensor.
    The Catersian Grid is a common-used positional encoding in deep learning.
    In this implementation, we follow the convention of ``grid_sample`` in
    PyTorch. In other words, ``[-1, -1]`` denotes the left-top corner while
    ``[1, 1]`` denotes the right-botton corner.
    """

    def forward(self, x, **kwargs):
        assert x.dim() == 4
        return self.make_grid2d_like(x, **kwargs)

    def make_grid2d(self, height, width, num_batches=1, requires_grad=False):
        h, w = height, width
        grid_y, grid_x = torch.meshgrid(torch.arange(0, h), torch.arange(0, w))
        grid_x = 2 * grid_x / max(float(w) - 1., 1.) - 1.
        grid_y = 2 * grid_y / max(float(h) - 1., 1.) - 1.
        grid = torch.stack((grid_x, grid_y), 0)
        grid.requires_grad = requires_grad

        grid = torch.unsqueeze(grid, 0)
        grid = grid.repeat(num_batches, 1, 1, 1)

        return grid

    def make_grid2d_like(self, x, requires_grad=False):
        h, w = x.shape[-2:]
        grid = self.make_grid2d(h, w, x.size(0), requires_grad=requires_grad)

        return grid.to(x)

#######################################Positional Encoding############

##########Central Difference Convolution, borrowed from https://github.com/ZitongYu/CDCN/
class CDC2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7):

        super(CDC2d, self).__init__() 
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):
        out_normal = self.conv(x)

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal 
        else:
            #pdb.set_trace()
            [C_out,C_in, kernel_size,kernel_size] = self.conv.weight.shape
            kernel_diff = self.conv.weight.sum(2).sum(2)
            kernel_diff = kernel_diff[:, :, None, None]
            out_diff = F.conv2d(input=x[:, :, 1:-1, 1:-1], weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0, groups=self.conv.groups)
            # print(out_normal.size())
            # print(out_diff.size())
            # print(x.size())

            return out_normal - self.theta * out_diff
###############

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        return functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        return functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)


def define_G(input_nc, output_nc, ngf, net_Gen_type, n_blocks, n_blocks_shared, n_domains, norm='batch', use_dropout=False, gpu_ids=[]):
    norm_layer = get_norm_layer(norm_type=norm)
    if type(norm_layer) == functools.partial:
        use_bias = norm_layer.func == nn.InstanceNorm2d
    else:
        use_bias = norm_layer == nn.InstanceNorm2d

    n_blocks -= n_blocks_shared
    n_blocks_enc = n_blocks // 2
    n_blocks_dec = n_blocks - n_blocks_enc

    dup_args = (ngf, norm_layer, use_dropout, gpu_ids, use_bias)
    enc_args = (input_nc, n_blocks_enc) + dup_args
    dec_args = (output_nc, n_blocks_dec) + dup_args

    if net_Gen_type == 'gen_v1':
        plex_netG = G_Plexer(n_domains, ResnetGenEncoder, enc_args, ResnetGenDecoderv1, dec_args)
    elif net_Gen_type == 'gen_SGPA':
        plex_netG = G_Plexer(n_domains, ResnetGenEncoderv2, enc_args, ResnetGenDecoderv1, dec_args) 
    else:
        raise NotImplementedError('Generation Net [%s] is not found' % net_Gen_type)

    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        plex_netG.cuda(gpu_ids[0])

    plex_netG.apply(weights_init)
    return plex_netG


def define_D(input_nc, ndf, netD_n_layers, n_domains, tensor, norm='batch', gpu_ids=[]):
    norm_layer = get_norm_layer(norm_type=norm)

    model_args = (input_nc, ndf, netD_n_layers, tensor, norm_layer, gpu_ids)
    plex_netD = D_Plexer(n_domains, NLayerDiscriminatorSN, model_args)

    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        plex_netD.cuda(gpu_ids[0])

    plex_netD.apply(weights_init)
    return plex_netD

def define_S(input_nc, ngf, n_blocks, n_domains, num_classes=19, norm='batch', use_dropout=False, gpu_ids=[]):
    norm_layer = get_norm_layer(norm_type=norm)

    model_args = (input_nc, n_blocks, ngf, num_classes, norm_layer, use_dropout, gpu_ids)
    plex_netS = S_Plexer(n_domains, SegmentorHeadv2, model_args)

    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        plex_netS.cuda(gpu_ids[0])

    plex_netS.apply(weights_init)
    return plex_netS

##############################################################################
# Classes
##############################################################################


# Defines the GAN loss which uses the Relativistic LSGAN
def GANLoss(inputs_real, inputs_fake, is_discr):
    if is_discr:
        y = -1
    else:
        y = 1
        inputs_real = [i.detach() for i in inputs_real]
    loss = lambda r,f : torch.mean((r-f+y)**2)
    losses = [loss(r,f) for r,f in zip(inputs_real, inputs_fake)]
    multipliers = list(range(1, len(inputs_real)+1));  multipliers[-1] += 1
    losses = [m*l for m,l in zip(multipliers, losses)]
    return sum(losses) / (sum(multipliers) * len(losses))
######Optional added by lfy

def PixelConsistencyLoss(inputs_img, GT_img, ROI_mask, ssim_winsize):
    "Pixel-wise Consistency Loss. inputs_img and GT_img are 4D tensors range [-1, 1], while ROI_mask is a 2D tensor."
    input_masked = inputs_img.mul(ROI_mask.expand_as(inputs_img))
    GT_masked = GT_img.mul(ROI_mask.expand_as(GT_img))
    # print(len(ROI_mask.size()))
    if len(ROI_mask.size()) == 4:
        _, _, h, w = ROI_mask.size()
        area_ROI = torch.sum(ROI_mask[0, 0, :, :])
    elif len(ROI_mask.size()) == 3:
        _, h, w = ROI_mask.size()
        area_ROI = torch.sum(ROI_mask[0, :, :])
    else:
        h, w = ROI_mask.size()
        area_ROI = torch.sum(ROI_mask)

    criterionSSIM = SSIM_Loss(win_size=ssim_winsize, data_range=1.0, size_average=True, channel=3)
    criterionL1 = torch.nn.SmoothL1Loss()
    lambda_L1 = 10.0
    if area_ROI > 0:
        losses = ((h * w) / area_ROI) * (lambda_L1 * criterionL1(input_masked, GT_masked.detach()) + \
                    criterionSSIM((input_masked + 1) / 2, (GT_masked.detach() + 1) / 2))
    else:
        losses = 0.0

    return losses

def UpdateSegGT(seg_tensor, ori_seg_GT, prob_th):
    "Use the high confidence predicted class to update original segmentation GT."

    sm = torch.nn.Softmax(dim = 1)
    pred_sm = sm(seg_tensor.detach())
    pred_max_tensor = torch.max(pred_sm, dim=1)
    pred_max_value = pred_max_tensor[0]
    pred_max_category = pred_max_tensor[1]
    seg_HP_mask = torch.zeros_like(pred_max_value)
    seg_HP_mask = torch.where(pred_max_value > prob_th, torch.ones_like(pred_max_value), seg_HP_mask)
    seg_GT_float = ori_seg_GT.float()
    segGT_UC_mask = torch.zeros_like(seg_GT_float)
    segGT_UC_mask = torch.where(seg_GT_float == 255.0, torch.ones_like(seg_GT_float), segGT_UC_mask)
    seg_HP_mask_UC = seg_HP_mask.mul(segGT_UC_mask)
    mask_new_GT = seg_HP_mask_UC.mul(pred_max_category.float()) + (torch.ones_like(seg_HP_mask_UC) - seg_HP_mask_UC).mul(seg_GT_float)

    return mask_new_GT.detach()

def OnlSemDisModule(seg_tensor1, seg_tensor2, ori_seg_GT, input_IR, prob_th):
    "Online semantic distillation module: Use the common high confidence predicted class to update original segmentation GT."

    sm = torch.nn.Softmax(dim = 1)
    pred_sm1 = sm(seg_tensor1.detach())
    pred_sm2 = sm(seg_tensor2.detach())
    pred_max_tensor1 = torch.max(pred_sm1, dim=1)
    pred_max_tensor2 = torch.max(pred_sm2, dim=1)
    pred_max_value1 = pred_max_tensor1[0]
    pred_max_value2 = pred_max_tensor2[0]
    pred_max_category1 = pred_max_tensor1[1]
    pred_max_category2 = pred_max_tensor2[1]

    mask_category1 = pred_max_category1.float()
    mask_category2 = pred_max_category2.float()
    mask_sub = mask_category1 - mask_category2
    mask_inter = torch.zeros_like(mask_category1)
    mask_inter = torch.where(mask_sub == 0.0, torch.ones_like(mask_category1), mask_inter)

    seg_GT_float = ori_seg_GT.float()
    seg_HP_mask1 = torch.zeros_like(pred_max_value1)
    seg_HP_mask1 = torch.where(pred_max_value1 > prob_th, torch.ones_like(pred_max_value1), seg_HP_mask1)
    seg_HP_mask2 = torch.zeros_like(pred_max_value1)
    seg_HP_mask2 = torch.where(pred_max_value2 > prob_th, torch.ones_like(pred_max_value1), seg_HP_mask2)
    mask_inter_HP = seg_HP_mask1.mul(seg_HP_mask1)

    segGT_UC_mask = torch.zeros_like(seg_GT_float)
    segGT_UC_mask = torch.where(seg_GT_float == 255.0, torch.ones_like(seg_GT_float), segGT_UC_mask)
    seg_inter_mask_UC = mask_inter.mul(segGT_UC_mask)

    seg_inter_mask_UC_HP = mask_inter_HP.mul(seg_inter_mask_UC)

    mask_new_GT = seg_inter_mask_UC_HP.mul(mask_category1) + (torch.ones_like(seg_inter_mask_UC_HP) - seg_inter_mask_UC_HP).mul(seg_GT_float)
    mask_final = RefineIRMask(torch.squeeze(mask_new_GT), input_IR)
    ###Removal of vegetated areas from supervision
    mask_Bkg_all = torch.zeros_like(mask_final)
    mask_Bkg_all = torch.where(mask_final < 11.0, torch.ones_like(mask_Bkg_all), torch.zeros_like(mask_Bkg_all))
   
    mask_Build_new = torch.zeros_like(mask_final)
    mask_Build_new = torch.where(mask_final == 2.0, torch.ones_like(mask_Build_new), torch.zeros_like(mask_Build_new))
    mask_Sign_new = torch.zeros_like(mask_final)
    mask_Sign_new = torch.where(mask_final == 6.0, torch.ones_like(mask_Sign_new), torch.zeros_like(mask_Sign_new))
    mask_Light_new = torch.zeros_like(mask_final)
    mask_Light_new = torch.where(mask_final == 7.0, torch.ones_like(mask_Light_new), torch.zeros_like(mask_Light_new))
    mask_Car_new = torch.zeros_like(mask_final)
    mask_Car_new = torch.where(mask_final == 13.0, torch.ones_like(mask_Car_new), torch.zeros_like(mask_Car_new))
    mask_Bkg_stuff = mask_Bkg_all - mask_Build_new - mask_Sign_new - mask_Light_new

    "Before the parameters of the segmentation network are fixed, the threshold for the background category is set to 0.99; "
    "conversely, the threshold for all categories is set to 0.95."
    if torch.mean(mask_sub) == 0.0:
        High_th = prob_th
    else:
        High_th = prob_th + 0.04
    # High_th = 0.99
    LHP_mask = torch.zeros_like(pred_max_value1)
    LHP_mask = torch.where(pred_max_value1 < High_th, torch.ones_like(LHP_mask), torch.zeros_like(LHP_mask))
    # VegRoad_LP_mask = LHP_mask.mul(mask_Veg_new) + LHP_mask.mul(mask_Road_new)
    VegRoad_LP_mask = LHP_mask.mul(mask_Bkg_stuff)
    ####Confusing categories Mask
    
    mask_CurtVeg = (torch.ones_like(mask_Bkg_stuff) - VegRoad_LP_mask).mul(mask_final) + VegRoad_LP_mask * 255.0

    return mask_CurtVeg.expand_as(ori_seg_GT).detach()

def UpdateIRSegGTv3(seg_tensor1, seg_tensor2, ori_seg_GT, input_IR, prob_th):
    "Combining the online semantic distillation module with masks (predicted offline) of object categories to update "
    "the segmentation pseudo-labels of NTIR images."
    "ori_seg_GT: 1 * h * w."

    sm = torch.nn.Softmax(dim = 1)
    pred_sm1 = sm(seg_tensor1.detach())
    pred_sm2 = sm(seg_tensor2.detach())
    pred_max_tensor1 = torch.max(pred_sm1, dim=1)
    pred_max_tensor2 = torch.max(pred_sm2, dim=1)
    pred_max_value1 = pred_max_tensor1[0]
    pred_max_value2 = pred_max_tensor2[0]
    pred_max_category1 = pred_max_tensor1[1]
    pred_max_category2 = pred_max_tensor2[1]

    mask_category1 = pred_max_category1.float()
    mask_category2 = pred_max_category2.float()
    mask_sub = mask_category1 - mask_category2
    mask_inter = torch.zeros_like(mask_category1)
    mask_inter = torch.where(mask_sub == 0.0, torch.ones_like(mask_category1), mask_inter)

    seg_HP_mask1 = torch.zeros_like(pred_max_value1)
    seg_HP_mask1 = torch.where(pred_max_value1 > prob_th, torch.ones_like(pred_max_value1), seg_HP_mask1)
    seg_HP_mask2 = torch.zeros_like(pred_max_value1)
    seg_HP_mask2 = torch.where(pred_max_value2 > prob_th, torch.ones_like(pred_max_value1), seg_HP_mask2)
    mask_inter_HP = seg_HP_mask1.mul(seg_HP_mask1)
    
    seg_inter_mask_UC_HP = mask_inter_HP.mul(mask_inter)

    mask_new_GT = seg_inter_mask_UC_HP.mul(mask_category1) + (torch.ones_like(seg_inter_mask_UC_HP) - seg_inter_mask_UC_HP) * 255.0
    mask_final = RefineIRMask(torch.squeeze(mask_new_GT), input_IR)
    ###Removal of vegetated areas from supervision
    mask_Bkg_all = torch.zeros_like(mask_final)
    mask_Bkg_all = torch.where(mask_final < 11.0, torch.ones_like(mask_Bkg_all), torch.zeros_like(mask_Bkg_all))
    mask_Build_new = torch.zeros_like(mask_final)
    mask_Build_new = torch.where(mask_final == 2.0, torch.ones_like(mask_Build_new), torch.zeros_like(mask_Build_new))
    mask_Sign_new = torch.zeros_like(mask_final)
    mask_Sign_new = torch.where(mask_final == 6.0, torch.ones_like(mask_Sign_new), torch.zeros_like(mask_Sign_new))
    mask_Light_new = torch.zeros_like(mask_final)
    mask_Light_new = torch.where(mask_final == 7.0, torch.ones_like(mask_Light_new), torch.zeros_like(mask_Light_new))
    mask_Car_new = torch.zeros_like(mask_final)
    mask_Car_new = torch.where(mask_final == 13.0, torch.ones_like(mask_Car_new), torch.zeros_like(mask_Car_new))
    mask_Bkg_stuff = mask_Bkg_all - mask_Build_new - mask_Sign_new - mask_Light_new

    "Before the parameters of the segmentation network are fixed, the threshold for the background category is set to 0.99; "
    "conversely, the threshold for all categories is set to 0.95."
    if torch.mean(mask_sub) == 0.0:
        High_th = prob_th
    else:
        High_th = prob_th + 0.04
    
    # High_th = 0.99
    LHP_mask = torch.zeros_like(pred_max_value1)
    LHP_mask = torch.where(pred_max_value1 < High_th, torch.ones_like(LHP_mask), torch.zeros_like(LHP_mask))
    # VegRoad_LP_mask = LHP_mask.mul(mask_Veg_new) + LHP_mask.mul(mask_Road_new)
    VegRoad_LP_mask = LHP_mask.mul(mask_Bkg_stuff)
    ####Confusing categories Mask
    
    mask_CurtVeg = (torch.ones_like(mask_Bkg_stuff) - VegRoad_LP_mask).mul(mask_final) + VegRoad_LP_mask * 255.0

    ###Fusion with original GT masks of thing classes

    seg_GT_float = torch.squeeze(ori_seg_GT).float()
    segGT_obj_mask = torch.zeros_like(seg_GT_float)
    segGT_obj_mask = torch.where(seg_GT_float < 255.0, torch.ones_like(seg_GT_float), segGT_obj_mask)
    out_mask = (torch.ones_like(segGT_obj_mask) - segGT_obj_mask).mul(mask_CurtVeg) + segGT_obj_mask.mul(seg_GT_float)

    return out_mask.expand_as(ori_seg_GT).detach()

def RefineIRMask(ori_mask, input_IR):
    "Use original IR image to refine segmentaton mask for specific categories, i.e., Sky, Vegetation, Pole, and Person."
    "ori_mask: h * w,  input_IR: 1 * 3 * h * w"

    x_norm = (input_IR - torch.min(input_IR)) / (torch.max(input_IR) - torch.min(input_IR))
    IR_gray = torch.squeeze(.299 * x_norm[:,0:1,:,:] + .587 * x_norm[:,1:2,:,:] + .114 * x_norm[:,2:3,:,:])

    Pole_mask = torch.zeros_like(ori_mask)
    Veg_mask = torch.zeros_like(ori_mask)
    Sky_mask = torch.zeros_like(ori_mask)
    Person_mask = torch.zeros_like(ori_mask)
    Pole_mask = torch.where(ori_mask == 5.0, torch.ones_like(ori_mask), torch.zeros_like(ori_mask))
    Veg_mask = torch.where(ori_mask == 8.0, torch.ones_like(ori_mask), torch.zeros_like(ori_mask))
    Sky_mask = torch.where(ori_mask == 10.0, torch.ones_like(ori_mask), torch.zeros_like(ori_mask))
    Person_mask = torch.where(ori_mask == 11.0, torch.ones_like(ori_mask), torch.zeros_like(ori_mask))
    cnt_Pole = torch.sum(Pole_mask)
    cnt_Veg = torch.sum(Veg_mask)
    cnt_Sky = torch.sum(Sky_mask)
    cnt_Person = torch.sum(Person_mask)
    region_Pole = Pole_mask.mul(IR_gray)
    region_Veg = Veg_mask.mul(IR_gray)
    region_Sky = Sky_mask.mul(IR_gray)
    region_Person = Person_mask.mul(IR_gray)

    if cnt_Pole > 0:
        Pole_region_mean = torch.sum(region_Pole) / cnt_Pole
        if cnt_Sky > 0:
            Sky_region_mean = torch.sum(region_Sky) / cnt_Sky
            #####Corrected Pole region mean.
            Pole_region_Corr_mean = (Pole_region_mean + Sky_region_mean) * 0.5
            Pole_intradis = Pole_mask.mul(torch.pow((region_Pole - Pole_region_Corr_mean), 2))
        else:
            Pole_intradis = Pole_mask.mul(torch.pow((region_Pole - Pole_region_mean), 2))

    if cnt_Veg > 0:
        Veg_region_mean = torch.sum(region_Veg) / cnt_Veg
        Veg_intradis = Veg_mask.mul(torch.pow((region_Veg - Veg_region_mean), 2))

    if cnt_Sky > 0:
        Sky_region_mean = torch.sum(region_Sky) / cnt_Sky
        Sky_intradis = Sky_mask.mul(torch.pow((region_Sky - Sky_region_mean), 2))

    if cnt_Person > 0:
        Person_region_mean = torch.sum(region_Person) / cnt_Person
        Person_intradis = Person_mask.mul(torch.pow((region_Person - Person_region_mean), 2))

    ######Denoised for Sky
    if (cnt_Sky * cnt_Veg) > 0:
        Sky_Veg_dis = Sky_mask.mul(torch.pow((region_Sky - Veg_region_mean), 2))
        Sky_Veg_dis_err = Sky_intradis - Sky_Veg_dis
        Sky2Veg_mask = torch.zeros_like(ori_mask)
        Sky2Veg_mask = torch.where(Sky_Veg_dis_err > 0, torch.ones_like(ori_mask), torch.zeros_like(ori_mask))
        mask_Sky_refine = Sky2Veg_mask * 255.0 + (Sky_mask - Sky2Veg_mask) * 10.0

        new_Sky_mask = Sky_mask - Sky2Veg_mask
        cnt_Sky_new = torch.sum(new_Sky_mask)
        region_Sky_new = new_Sky_mask.mul(IR_gray)
        if cnt_Sky_new > 0:
            Sky_region_mean_new = torch.sum(region_Sky_new) / cnt_Sky_new
        else:
            Sky_region_mean_new = Sky_region_mean
    elif cnt_Sky > 0:
        Sky_region_mean_new = Sky_region_mean
        mask_Sky_refine = Sky_mask * 10.0
    else:
        mask_Sky_refine = Sky_mask * 10.0
        # Sky_region_mean_new = Sky_region_mean

    ######Denoised for Pole
    if (cnt_Pole * cnt_Sky) > 0:
        Pole_Sky_dis = Pole_mask.mul(torch.pow((region_Pole - Sky_region_mean_new), 2))
        Pole_Sky_dis_err = Pole_intradis - Pole_Sky_dis
        Pole2Sky_mask = torch.zeros_like(ori_mask)
        Pole2Sky_mask = torch.where(Pole_Sky_dis_err > 0, torch.ones_like(ori_mask), torch.zeros_like(ori_mask))
        mask_Pole_refine = Pole2Sky_mask * 255.0 + (Pole_mask - Pole2Sky_mask) * 5.0
    else:
        mask_Pole_refine = Pole_mask * 5.0

    ######Denoised for Person
    if (cnt_Person * cnt_Veg) > 0:
        Person_Veg_dis = Person_mask.mul(torch.pow((region_Person - Veg_region_mean), 2))
        Person_Veg_dis_err = Person_intradis - Person_Veg_dis
        Person2Veg_mask = torch.zeros_like(ori_mask)
        Person2Veg_mask = torch.where(Person_Veg_dis_err > 0, torch.ones_like(ori_mask), torch.zeros_like(ori_mask))
        mask_Person_refine = Person2Veg_mask * 255.0 + (Person_mask - Person2Veg_mask) * 11.0

        new_Person_mask = Person_mask - Person2Veg_mask
        cnt_Person_new = torch.sum(new_Person_mask)
        region_Person_new = new_Person_mask.mul(IR_gray)
        if cnt_Person_new > 0:
            Person_region_mean_new = torch.sum(region_Person_new) / cnt_Person_new
        else:
            Person_region_mean_new = Person_region_mean
    elif cnt_Person > 0:
        Person_region_mean_new = Person_region_mean
        mask_Person_refine = Person_mask * 11.0
    else:
        mask_Person_refine = Person_mask * 11.0
        # Person_region_mean_new = Person_region_mean

    ######Denoised for Vegetation
    if (cnt_Veg * cnt_Sky * cnt_Person) > 0:
        Veg_Sky_dis = Veg_mask.mul(torch.pow((region_Veg - Sky_region_mean_new), 2))
        Veg_Sky_dis_err = Veg_intradis - Veg_Sky_dis
        Veg2Sky_mask = torch.zeros_like(ori_mask)
        Veg2Sky_mask = torch.where(Veg_Sky_dis_err > 0, torch.ones_like(ori_mask), torch.zeros_like(ori_mask))

        Veg_Person_dis = Veg_mask.mul(torch.pow((region_Veg - Person_region_mean_new), 2))
        Veg_Person_dis_err = Veg_intradis - Veg_Person_dis
        Veg2Person_mask = torch.zeros_like(ori_mask)
        Veg2Person_mask = torch.where(Veg_Person_dis_err > 0, torch.ones_like(ori_mask), torch.zeros_like(ori_mask))

        uncertain_mask_veg = torch.zeros_like(ori_mask)
        fuse_uncer = Veg2Sky_mask + Veg2Person_mask
        uncertain_mask_veg = torch.where(fuse_uncer > 0, torch.ones_like(ori_mask), torch.zeros_like(ori_mask))

        mask_Veg_refine = uncertain_mask_veg * 255.0 + (Veg_mask - uncertain_mask_veg) * 8.0
    elif (cnt_Veg * cnt_Sky) > 0:
        Veg_Sky_dis = Veg_mask.mul(torch.pow((region_Veg - Sky_region_mean_new), 2))
        Veg_Sky_dis_err = Veg_intradis - Veg_Sky_dis
        Veg2Sky_mask = torch.zeros_like(ori_mask)
        Veg2Sky_mask = torch.where(Veg_Sky_dis_err > 0, torch.ones_like(ori_mask), torch.zeros_like(ori_mask))
        mask_Veg_refine = Veg2Sky_mask * 255.0 + (Veg_mask - Veg2Sky_mask) * 8.0
    elif (cnt_Veg * cnt_Person) > 0:
        Veg_Person_dis = Veg_mask.mul(torch.pow((region_Veg - Person_region_mean_new), 2))
        Veg_Person_dis_err = Veg_intradis - Veg_Person_dis
        Veg2Person_mask = torch.zeros_like(ori_mask)
        Veg2Person_mask = torch.where(Veg_Person_dis_err > 0, torch.ones_like(ori_mask), torch.zeros_like(ori_mask))
        mask_Veg_refine = Veg2Person_mask * 255.0 + (Veg_mask - Veg2Person_mask) * 8.0
    else:
        mask_Veg_refine = Veg_mask * 8.0

    mask_refine = mask_Sky_refine + mask_Pole_refine + mask_Person_refine + mask_Veg_refine + \
                    (torch.ones_like(ori_mask) - Pole_mask - Veg_mask - Sky_mask - Person_mask).mul(ori_mask)

    return mask_refine.detach()

def ClsMeanFea(input_tensor, SegMask, num_class, gpu_ids=[]):
    "Computing mean feafure for each category."

    GAP = nn.AdaptiveAvgPool2d(1)
    b, c, h, w = input_tensor.size()
    _, seg_h, seg_w = SegMask.size()
    mask_resize = F.interpolate(SegMask.expand(1, 1, seg_h, seg_w).float(), size=[h, w], mode='nearest')
    seg_mask = mask_resize[0]
    # _, c_a, _, _ = att_maps.size()
    out_tensor = torch.zeros(b, 1, num_class, c).cuda(gpu_ids)
    out_cls_tensor = torch.zeros(num_class, 1).cuda(gpu_ids)
    # att_maps_max_value = torch.zeros(b, 1, c_a, 1).cuda(gpu_ids)
    if b == 1:
        for i in range(num_class):
            ###The similarity between sidewalks and other categories was excluded from the calculation 
            ### because of the extremely high similarity between sidewalks and roads.
            if i != 1:
                temp_tensor = torch.zeros_like(seg_mask)
                temp_tensor = torch.where(seg_mask == i, torch.ones_like(temp_tensor), torch.zeros_like(temp_tensor))
                # temp_tensor = att_maps[0, i, :, :]
                if (torch.sum(temp_tensor)).item() > 0:
                    # print((torch.sum(temp_tensor)).item())
                    out_cls_tensor[i, 0] = 1.0
                    cls_fea_map = (temp_tensor.detach().expand_as(input_tensor)).mul(input_tensor)
                    # if torch.isnan(cls_fea_map).any().cpu().numpy():
                    #     print('NaN is existing in cls_fea_map. ')

                    ave_fea = (torch.squeeze(GAP(cls_fea_map) * h * w)) / torch.sum(temp_tensor)    # b * c * 1 * 1
                    # ave_fea = (torch.squeeze(GAP(cls_fea_map) * h * w))
                    out_tensor[0, 0, i, :] = ave_fea

    else:
        raise NotImplementedError('ChannelSoftmax for batchsize larger than 1 is not implemented.')

    out_tensor_L2norm = torch.nn.functional.normalize(out_tensor, p=2, dim=3)

    return out_tensor_L2norm, out_cls_tensor


def ClsMeanPixelValue(input_tensor, SegMask, num_class, gpu_ids=[]):
    "Computing mean feafure for each category."

    GAP = nn.AdaptiveAvgPool2d(1)
    b, c, h, w = input_tensor.size()
    _, seg_h, seg_w = SegMask.size()
    mask_resize = F.interpolate(SegMask.expand(1, 1, seg_h, seg_w).float(), size=[h, w], mode='nearest')
    seg_mask = mask_resize[0]
    # _, c_a, _, _ = att_maps.size()
    out_tensor = torch.zeros(num_class, c).cuda(gpu_ids)
    out_cls_tensor = torch.zeros(num_class, 1).cuda(gpu_ids)
    out_cls_ratio_tensor = torch.zeros(num_class, 1).cuda(gpu_ids)
    # att_maps_max_value = torch.zeros(b, 1, c_a, 1).cuda(gpu_ids)
    if b == 1:
        for i in range(num_class):
            ###The similarity between sidewalks and other categories was excluded from the calculation 
            ### because of the extremely high similarity between sidewalks and roads.
            if i != 1:
                temp_tensor = torch.zeros_like(seg_mask)
                temp_tensor = torch.where(seg_mask == i, torch.ones_like(temp_tensor), torch.zeros_like(temp_tensor))
                # temp_tensor = att_maps[0, i, :, :]
                if (torch.sum(temp_tensor)).item() > 0:
                    # print((torch.sum(temp_tensor)).item())
                    out_cls_tensor[i, 0] = 1.0
                    out_cls_ratio_tensor[i, 0] = torch.sum(temp_tensor) / (h * w)
                    cls_fea_map = (temp_tensor.detach().expand_as(input_tensor)).mul(input_tensor)

                    out_tensor[i, :] = (torch.squeeze(GAP(cls_fea_map) * h * w)) / torch.sum(temp_tensor)    # b * c * 1 * 1
                    # ave_fea = (torch.squeeze(GAP(cls_fea_map) * h * w))
                    # out_tensor[i, :] = ave_fea

    else:
        raise NotImplementedError('ChannelSoftmax for batchsize larger than 1 is not implemented.')

    # out_tensor_L2norm = torch.nn.functional.normalize(out_tensor, p=2, dim=3)

    return out_tensor, out_cls_tensor, out_cls_ratio_tensor

def ClsMeanPixelValuev2(input_tensor, SegMask, num_class, gpu_ids=[]):
    "Computing mean feafure for each category."

    GAP = nn.AdaptiveAvgPool2d(1)
    b, c, h, w = input_tensor.size()
    _, seg_h, seg_w = SegMask.size()
    mask_resize = F.interpolate(SegMask.expand(1, 1, seg_h, seg_w).float(), size=[h, w], mode='nearest')
    seg_mask = mask_resize[0]
    # _, c_a, _, _ = att_maps.size()
    out_tensor = torch.zeros(num_class, c).cuda(gpu_ids)
    out_cls_tensor = torch.zeros(num_class, 1).cuda(gpu_ids)
    out_cls_ratio_tensor = torch.zeros(num_class, 1).cuda(gpu_ids)
    # att_maps_max_value = torch.zeros(b, 1, c_a, 1).cuda(gpu_ids)
    if b == 1:
        for i in range(num_class):
            ###The similarity between sidewalks and other categories was excluded from the calculation 
            ### because of the extremely high similarity between sidewalks and roads.
            if i != 1:
                temp_tensor = torch.zeros_like(seg_mask)
                temp_tensor = torch.where(seg_mask == i, torch.ones_like(temp_tensor), torch.zeros_like(temp_tensor))

                if (torch.sum(temp_tensor)).item() > 0:
                    out_cls_tensor[i, 0] = 1.0
                    out_cls_ratio_tensor[i, 0] = torch.sum(temp_tensor) / (h * w)
                    cls_fea_map = (temp_tensor.detach().expand_as(input_tensor)).mul(input_tensor)

                    out_tensor[i, :] = (torch.squeeze(GAP(cls_fea_map) * h * w)) / torch.sum(temp_tensor)    # b * c * 1 * 1

        temp_tensor_person = torch.zeros_like(seg_mask)
        temp_tensor_person = torch.where(seg_mask == 11, torch.ones_like(temp_tensor_person), torch.zeros_like(temp_tensor_person))
        temp_tensor_uncertain = torch.zeros_like(seg_mask)
        temp_tensor_uncertain = torch.where(seg_mask == 255, torch.ones_like(temp_tensor_person), torch.zeros_like(temp_tensor_person))
        temp_tensor_nonperson = torch.ones_like(temp_tensor_person) - temp_tensor_person - temp_tensor_uncertain

        if (torch.sum(temp_tensor_nonperson)).item() > 0:
            # out_cls_tensor[i, 0] = 1.0
            # out_cls_ratio_tensor[i, 0] = torch.sum(temp_tensor) / (h * w)
            nonperson_fea_map = (temp_tensor_nonperson.detach().expand_as(input_tensor)).mul(input_tensor)

            nonperson_fea_mean = (torch.squeeze(GAP(nonperson_fea_map) * h * w)) / torch.sum(temp_tensor_nonperson)
        else:
            nonperson_fea_mean = torch.zeros(1, c).cuda(gpu_ids)
    else:
        raise NotImplementedError('ChannelSoftmax for batchsize larger than 1 is not implemented.')

    # out_tensor_L2norm = torch.nn.functional.normalize(out_tensor, p=2, dim=3)

    return out_tensor, out_cls_tensor, out_cls_ratio_tensor, nonperson_fea_mean


def getRoadDarkRegionMean(input_img, input_mask, gpu_ids=[]):
    "Obtain the mean value of the below-average brightness portion of the road area."
    img_gray = torch.squeeze(.299 * input_img[:,0:1,:,:] + .587 * input_img[:,1:2,:,:] + .114 * input_img[:,2:3,:,:])
    Road_mask = torch.zeros_like(input_mask)
    Road_mask = torch.where(input_mask < 2.0, torch.ones_like(Road_mask), torch.zeros_like(Road_mask))
    if torch.sum(Road_mask) > 0:
        Road_region = img_gray.mul(Road_mask.detach())
        Road_region_mean = torch.sum(Road_region) / torch.sum(Road_mask)
        Road_region_filling_one = Road_region + (torch.ones_like(input_mask) - Road_mask).mul(torch.ones_like(Road_region))
        Road_Dark_Region_Mask = torch.zeros_like(input_mask)
        Road_Dark_Region_Mask = torch.where(Road_region_filling_one < Road_region_mean, torch.ones_like(Road_mask), torch.zeros_like(Road_mask))
        out = torch.sum(Road_Dark_Region_Mask.mul(img_gray)) / torch.sum(Road_Dark_Region_Mask)
    else:
        out = torch.zeros(1).cuda(gpu_ids)

    return out

def getLightDarkRegionMean(cls_idx, input_img, input_mask, ref_img, gpu_ids=[]):
    "Obtain the mean value of the below-average brightness portion of the traffic light area."
    "The dark region mask of the traffic light region is first obtained using the reference image, and then "
    "the mean value of the corresponding region of the input image is calculated."

    _, _, h, w = input_img.size()
    input_img_gray = torch.squeeze(.299 * input_img[:,0:1,:,:] + .587 * input_img[:,1:2,:,:] + .114 * input_img[:,2:3,:,:])
    ref_img_gray = torch.squeeze(.299 * ref_img[:,0:1,:,:] + .587 * ref_img[:,1:2,:,:] + .114 * ref_img[:,2:3,:,:])
    light_mask_ori = torch.zeros_like(input_mask)
    light_mask_ori = torch.where(input_mask == cls_idx, torch.ones_like(light_mask_ori), torch.zeros_like(light_mask_ori))
    max_pool_k3 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
    Light_mask = torch.squeeze(-max_pool_k3(- light_mask_ori.expand(1, 1, h, w)))
    light_region_area = torch.sum(Light_mask)
    if light_region_area > 1:
        "In order to avoid that the bright and dark regions cannot be divided, the threshold is set to 1."
        Light_region = ref_img_gray.mul(Light_mask.detach())
        Light_region_mean = torch.sum(Light_region) / light_region_area

        Light_region_filling_one = Light_region + (torch.ones_like(input_mask) - Light_mask).mul(torch.ones_like(Light_region))
        Light_Dark_Region_Mask = torch.zeros_like(input_mask)
        Light_Dark_Region_Mask = torch.where(Light_region_filling_one < Light_region_mean, torch.ones_like(Light_mask), torch.zeros_like(Light_mask))
        Light_Dark_Region_Mean = torch.sum(Light_Dark_Region_Mask.mul(input_img_gray)) / torch.sum(Light_Dark_Region_Mask)

        Light_Bright_Region_Mask = Light_mask - Light_Dark_Region_Mask
        
        #####Light Bright region min
        Light_BR_filling_one = Light_Bright_Region_Mask.mul(input_img_gray) + (torch.ones_like(input_mask) - Light_Bright_Region_Mask)
        Light_Bright_Region_Min = torch.min(Light_BR_filling_one)
        ###Compute channle mean.
        input_img_3dim = torch.squeeze(input_img)
        input_img_DR_Masked = input_img_3dim.mul(Light_Dark_Region_Mask.expand_as(input_img_3dim))
        input_img_DR_mean_3dim = torch.sum(input_img_DR_Masked, dim=0, keepdim=True) / 3.0 #3*h*w
        input_img_DR_submean = (input_img_DR_Masked - input_img_DR_mean_3dim) ** 2
        input_img_DR_var = torch.max(torch.sum(input_img_DR_submean, dim=0))

    else:
        Light_Dark_Region_Mean = torch.zeros(1).cuda(gpu_ids)
        Light_Bright_Region_Min = torch.zeros(1).cuda(gpu_ids)
        input_img_DR_var = torch.zeros(1).cuda(gpu_ids)
        # Light_region_ref_var = torch.zeros(1).cuda(gpu_ids)

    return Light_Dark_Region_Mean, light_region_area, Light_Bright_Region_Min, input_img_DR_var


def CarIntraClsVarLoss(input_IR, fake_vis, SegMask, num_class, gpu_ids=[]):
    "Encouraging intra-class feature variability in foreground categories in IR images."

    GAP = nn.AdaptiveAvgPool2d(1)
    b, c, h, w = fake_vis.size()
    _, seg_h, seg_w = SegMask.size()
    mask_resize = F.interpolate(SegMask.expand(1, 1, seg_h, seg_w).float(), size=[h, w], mode='nearest')
    seg_mask = mask_resize[0]

    IR_gray = torch.squeeze(.299*input_IR[:,0:1,:,:] + .587*input_IR[:,1:2,:,:] + .114*input_IR[:,2:3,:,:])

    mask_intensity_low = torch.zeros_like(IR_gray)
    mask_intensity_low = torch.where(IR_gray < 0.3, torch.ones_like(IR_gray), torch.zeros_like(IR_gray))
    mask_intensity_high = torch.zeros_like(IR_gray)
    mask_intensity_high = torch.where(IR_gray > 0.6, torch.ones_like(IR_gray), torch.zeros_like(IR_gray))
    # _, c_a, _, _ = att_maps.size()
    out_tensor = torch.zeros(num_class, 1).cuda(gpu_ids)
    out_cls_tensor = torch.zeros(num_class, 1).cuda(gpu_ids)

    if b == 1:
        for i in range(13, 16):
            temp_tensor = torch.zeros_like(seg_mask)
            temp_tensor = torch.where(seg_mask == i, torch.ones_like(temp_tensor), torch.zeros_like(temp_tensor))
            temp_tensor_low = temp_tensor.mul(mask_intensity_low)
            temp_tensor_high = temp_tensor.mul(mask_intensity_high)
            # temp_tensor = att_maps[0, i, :, :]
            if (torch.sum(temp_tensor_low)).item() > 0:
                # print((torch.sum(temp_tensor)).item())
                out_cls_tensor[i, 0] = 1.0
                # out_cls_ratio_tensor[i, 0] = torch.sum(temp_tensor) / (h * w)
                fea_map_low = (temp_tensor_low.detach().expand_as(fake_vis)).mul(fake_vis)
                fea_low = (torch.squeeze(GAP(fea_map_low) * h * w)) / torch.sum(temp_tensor_low)    # b * c * 1 * 1

                out_tensor[i, 0] = torch.mean(fea_low)
    else:
        raise NotImplementedError('ChannelSoftmax for batchsize larger than 1 is not implemented.')

    if torch.sum(out_cls_tensor).item() > 0:
        out_loss = F.relu((torch.sum(out_cls_tensor.mul(out_tensor)) / torch.sum(out_cls_tensor)) - 0.3)
    else:
        out_loss = torch.zeros(1).cuda(gpu_ids)

    return out_loss

def CondGradRepaLoss(fake_img, fake_mask, real_IR, gpu_ids=[]):
    "Conditional Gradient Repair loss for background categories. fake_img: fake vis image. fake_mask: IR seg mask."
    
    ###Conditional Gradient Repair loss for background categories
    _, _, h, w = fake_img.size()
    _, seg_h, seg_w = fake_mask.size()
    fake_mask_resize = F.interpolate(fake_mask.expand(1, 1, seg_h, seg_w).float(), size=[h, w], mode='nearest')

    seg_mask_fake = fake_mask_resize[0]
    IR_bkg_mask = torch.zeros_like(seg_mask_fake)
    IR_bkg_mask = torch.where(seg_mask_fake < 11.0, torch.ones_like(seg_mask_fake), torch.zeros_like(seg_mask_fake))
    IR_UC_mask = torch.zeros_like(seg_mask_fake)
    IR_UC_mask = torch.where(seg_mask_fake == 255.0, torch.ones_like(seg_mask_fake), torch.zeros_like(seg_mask_fake))
    IR_bkg_fuse_mask = IR_bkg_mask + IR_UC_mask
    getgradmap = Get_gradmag_gray()
    IR_grad = getgradmap(real_IR)
    fake_vis_grad = getgradmap(fake_img)
    IR_grad_bkg = IR_grad.mul(IR_bkg_fuse_mask.expand_as(IR_grad))
    vis_grad_bkg = fake_vis_grad.mul(IR_bkg_fuse_mask.expand_as(fake_vis_grad))
    IR_grad_bkg_sum = torch.sum(IR_grad_bkg)
    
    if IR_grad_bkg_sum > 0:
        # bkg_EC_loss = torch.sum(F.relu(IR_grad_bkg.detach() - vis_grad_bkg)) / IR_grad_bkg_sum.detach()

        IR_grad_bkg_mean = IR_grad_bkg_sum / torch.sum(IR_bkg_fuse_mask)
        IR_grad_bkg_high_mask = torch.zeros_like(IR_grad_bkg)
        IR_grad_bkg_high_mask = torch.where(IR_grad_bkg > IR_grad_bkg_mean, torch.ones_like(IR_grad_bkg), torch.zeros_like(IR_grad_bkg))
        IR_grad_bkg_high = IR_grad_bkg_high_mask.mul(IR_grad_bkg)
        vis_grad_bkg_high = IR_grad_bkg_high_mask.mul(vis_grad_bkg)
        IR_grad_bkg_high_sum = torch.sum(IR_grad_bkg_high)
        if IR_grad_bkg_high_sum > 0:
            bkg_EC_loss = torch.sum(F.relu(IR_grad_bkg_high.detach() - vis_grad_bkg_high)) / IR_grad_bkg_high_sum.detach()
        else:
            bkg_EC_loss = torch.zeros(1).cuda(gpu_ids)
    else:
        bkg_EC_loss = torch.zeros(1).cuda(gpu_ids)

    return bkg_EC_loss

def TrafLighLumiLoss(fake_img, fake_mask, real_IR, gpu_ids=[]):
    "Traffic Light Luminance Loss. fake_img: fake vis image. fake_mask: IR seg mask. real_mask: Vis seg mask."
    _, _, h, w = fake_img.size()
    _, seg_h, seg_w = fake_mask.size()
    fake_mask_resize = F.interpolate(fake_mask.expand(1, 1, seg_h, seg_w).float(), size=[h, w], mode='nearest')

    fake_img_norm = (fake_img + 1.0) * 0.5
    real_IR_norm = (real_IR + 1.0) * 0.5

    fake_vis_Light_DR_Mean, fake_vis_Light_area, fake_vis_Light_BR_Min, _ = \
        getLightDarkRegionMean(6.0, fake_img_norm, torch.squeeze(fake_mask_resize), real_IR_norm.detach(), gpu_ids)
    
    if fake_vis_Light_area > 100:
        losses = F.relu(fake_vis_Light_DR_Mean - fake_vis_Light_BR_Min) / (fake_vis_Light_BR_Min.detach() + 1e-6)
    else:
        losses = torch.zeros(1).cuda(gpu_ids)

    return losses

def MaskedCGRLoss(input_mask, real_IR, fake_vis, gpu_ids):
    "Conditional Gradient Repairing loss for input binary mask."

    getgradmap = Get_gradmag_gray()
    IR_grad = getgradmap(real_IR)
    fake_vis_grad = getgradmap(fake_vis)
    IR_grad_bkg = IR_grad.mul(input_mask.expand_as(IR_grad))
    vis_grad_bkg = fake_vis_grad.mul(input_mask.expand_as(fake_vis_grad))
    IR_grad_bkg_sum = torch.sum(IR_grad_bkg)
    
    if IR_grad_bkg_sum > 0:
        # bkg_EC_loss = torch.sum(F.relu(IR_grad_bkg.detach() - vis_grad_bkg)) / IR_grad_bkg_sum.detach()

        IR_grad_bkg_mean = IR_grad_bkg_sum / torch.sum(input_mask)
        IR_grad_bkg_high_mask = torch.zeros_like(IR_grad_bkg)
        IR_grad_bkg_high_mask = torch.where(IR_grad_bkg > IR_grad_bkg_mean, torch.ones_like(IR_grad_bkg), torch.zeros_like(IR_grad_bkg))
        IR_grad_bkg_high = IR_grad_bkg_high_mask.mul(IR_grad_bkg)
        vis_grad_bkg_high = IR_grad_bkg_high_mask.mul(vis_grad_bkg)
        IR_grad_bkg_high_sum = torch.sum(IR_grad_bkg_high)
        if IR_grad_bkg_high_sum > 0:
            losses = torch.sum(F.relu(IR_grad_bkg_high.detach() - vis_grad_bkg_high)) / IR_grad_bkg_high_sum.detach()
        else:
            losses = torch.zeros(1).cuda(gpu_ids)
    else:
        losses = torch.zeros(1).cuda(gpu_ids)

    return losses

def ComIRCGRLoss(FG_mask, FG_mask_flip, ori_Seg_GT, real_IR, fake_vis, gpu_ids):
    "Conditional Gradient Repairing loss for composite IR."
    _, _, h, w = fake_vis.size()
    _, seg_h, seg_w = ori_Seg_GT.size()
    IR_mask_resize = F.interpolate(ori_Seg_GT.expand(1, 1, seg_h, seg_w).float(), size=[h, w], mode='nearest')
    seg_mask_fake = torch.squeeze(IR_mask_resize)  #h * w

    FG_mask_fused_4d = FG_mask + FG_mask_flip
    FG_mask_fused = FG_mask_fused_4d[0, 0, :, :]
    IR_bkg_mask = torch.zeros_like(seg_mask_fake)
    IR_bkg_mask = torch.where(seg_mask_fake < 11.0, torch.ones_like(seg_mask_fake), torch.zeros_like(seg_mask_fake))
    IR_UC_mask = torch.zeros_like(seg_mask_fake)
    IR_UC_mask = torch.where(seg_mask_fake == 255.0, torch.ones_like(seg_mask_fake), torch.zeros_like(seg_mask_fake))
    IR_bkg_ori_mask = IR_bkg_mask + IR_UC_mask
    # mask_overlap = FG_mask_fused.mul(IR_bkg_ori_mask)
    ComIR_bkg_mask = IR_bkg_ori_mask - FG_mask_fused.mul(IR_bkg_ori_mask)

    losses = MaskedCGRLoss(ComIR_bkg_mask, real_IR, fake_vis, gpu_ids)

    return losses

def BiasCorrLoss(Seg_mask, fake_IR, real_vis, rec_vis, real_vis_edgemap, gpu_ids=[]):
    "Bias correction loss includes the artifact bias correction loss and the color bias correction loss. "
    "The category index for streetlights is defined as 12."

    _, _, h, w = fake_IR.size()
    _, seg_h, seg_w = Seg_mask.size()
    GT_mask_resize = F.interpolate(Seg_mask.expand(1, 1, seg_h, seg_w).float(), size=[h, w], mode='nearest')
    GT_mask = torch.squeeze(GT_mask_resize[0])

    light_mask_ori = torch.zeros_like(GT_mask)
    light_mask_ori = torch.where(GT_mask == 6.0, torch.ones_like(GT_mask), torch.zeros_like(GT_mask))
    veg_mask = torch.zeros_like(GT_mask)
    veg_mask = torch.where(GT_mask == 8.0, torch.ones_like(GT_mask), torch.zeros_like(GT_mask))
    SLight_mask_ori = torch.zeros_like(GT_mask)
    SLight_mask_ori = torch.where(GT_mask == 12.0, torch.ones_like(GT_mask), torch.zeros_like(GT_mask))
    light_mask = RefineLightMask(GT_mask, real_vis, gpu_ids)

    fake_img_norm = (fake_IR + 1.0) * 0.5
    real_img_norm = (real_vis + 1.0) * 0.5
    fake_IR_gray = torch.squeeze(.299 * fake_img_norm[:,0:1,:,:] + .587 * fake_img_norm[:,1:2,:,:] + .114 * fake_img_norm[:,2:3,:,:])
    real_vis_gray = torch.squeeze(.299 * real_img_norm[:,0:1,:,:] + .587 * real_img_norm[:,1:2,:,:] + .114 * real_img_norm[:,2:3,:,:])

    ###########Artifact bias correction loss
    ####Street light luminance adjustment loss
    "Excluding the noise at the periphery and the street light area less than 25."
    max_pool_k3 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    SLight_mask = torch.squeeze(-max_pool_k3(- SLight_mask_ori.expand(1, 1, h, w)))

    if torch.sum(SLight_mask) > 25:
        real_vis_SLight_region = SLight_mask.mul(real_vis_gray)
        real_vis_SLight_mean = torch.sum(real_vis_SLight_region) / torch.sum(SLight_mask)
        SLight_high_mask = torch.zeros_like(SLight_mask)
        SLight_high_mask = torch.where(real_vis_SLight_region > real_vis_SLight_mean, torch.ones_like(GT_mask), torch.zeros_like(GT_mask))
        fake_IR_SLight_region_high = SLight_high_mask.mul(fake_IR_gray) + (torch.ones_like(SLight_high_mask) - SLight_high_mask)
        
        # SLight_loss = F.relu(0.5 - torch.min(fake_IR_SLight_region_high))
        if torch.sum(veg_mask) > 0:
            fake_IR_veg_region = veg_mask.mul(fake_IR_gray)
            fake_IR_veg_mean = torch.sum(fake_IR_veg_region) / torch.sum(veg_mask)
            fake_IR_veg_max = torch.max(fake_IR_veg_region)
            "Avoid the maximum value of the brightness of the vegetation area is too high."
            SLight_loss = F.relu(fake_IR_veg_mean.detach() + 0.25 - torch.min(fake_IR_SLight_region_high))
        else:
            SLight_loss = F.relu(0.7 - torch.min(fake_IR_SLight_region_high))
    else:
        SLight_loss = 0.0

    ########Light region SGA loss
    light_mask_all = light_mask_ori + SLight_mask_ori
    if torch.sum(light_mask_all) > 100:
        real_vis_EM = torch.squeeze(real_vis_edgemap)
        gradmag_com = Get_gradmag_gray()
        fake_IR_GM = torch.squeeze(gradmag_com(fake_IR))

        EM_masked = light_mask_all.mul(real_vis_EM)
        GM_masked = light_mask_all.mul(fake_IR_GM)

        sum_edge_pixels = torch.sum(EM_masked)
        if sum_edge_pixels > 0:
            fake_grad_norm = GM_masked / (torch.max(GM_masked) + 1e-4)
            loss_sga_light = 0.5 * (torch.sum(F.relu(0.8 * EM_masked - fake_grad_norm))) / sum_edge_pixels
        else:
            loss_sga_light = 0.0
    else:
        loss_sga_light = 0.0
    #################

    ####Traffic light luminance adjustment loss
    if torch.sum(light_mask) > 100:
        
        real_vis_light_region = light_mask.mul(real_vis_gray)
        real_vis_light_mean = torch.sum(real_vis_light_region) / torch.sum(light_mask)
        real_vis_light_region_submean = light_mask.mul(real_vis_light_region - real_vis_light_mean)
        real_vis_light_region_norm2 = torch.sqrt(torch.sum(real_vis_light_region_submean ** 2))
        real_vis_light_norm = real_vis_light_region_submean / (real_vis_light_region_norm2 + 1e-4)

        light_high_mask = torch.zeros_like(light_mask)
        light_high_mask = torch.where(real_vis_light_region > real_vis_light_mean, torch.ones_like(GT_mask), torch.zeros_like(GT_mask))

        high_area_ratio = torch.sum(light_high_mask) / torch.sum(light_mask)

        if high_area_ratio > 0.1:
            fake_IR_light_region = light_mask.mul(fake_IR_gray)
            fake_IR_light_mean = torch.sum(fake_IR_light_region) / torch.sum(light_mask)
            fake_IR_light_region_submean = light_mask.mul(fake_IR_light_region - fake_IR_light_mean)
            fake_IR_light_region_norm2 = torch.sqrt(torch.sum(fake_IR_light_region_submean ** 2))
            fake_IR_light_norm = fake_IR_light_region_submean / (fake_IR_light_region_norm2 + 1e-4)

            TLight_loss = F.relu(0.8 - torch.sum(fake_IR_light_norm.mul(real_vis_light_norm.detach())))
        else:
            TLight_loss = 0.0
    else:
        TLight_loss = 0.0

    ABC_losses = TLight_loss + SLight_loss + loss_sga_light

    ##########Color bias correction loss
    ####Traffic sign reconstruction loss
    sign_mask = torch.zeros_like(GT_mask)
    sign_mask = torch.where(GT_mask == 7.0, torch.ones_like(GT_mask), torch.zeros_like(GT_mask))
    if torch.sum(sign_mask) > 10:
        sign_rec_loss = PixelConsistencyLoss(rec_vis, real_vis, sign_mask, 3)
    else:
        sign_rec_loss = 0.0
    ####Traffic light reconstruction loss
    if torch.sum(light_mask) > 10:
        light_rec_loss = PixelConsistencyLoss(rec_vis, real_vis, light_mask, 3)
    else:
        light_rec_loss = 0.0
    ####Motorcycle reconstruction loss
    motorcycle_mask = torch.zeros_like(GT_mask)
    motorcycle_mask = torch.where(GT_mask == 17.0, torch.ones_like(GT_mask), torch.zeros_like(GT_mask))
    if torch.sum(motorcycle_mask) > 10:
        motorcycle_rec_loss = PixelConsistencyLoss(rec_vis, real_vis, motorcycle_mask, 3)
    else:
        motorcycle_rec_loss = 0.0
    
    CBC_losses = sign_rec_loss + light_rec_loss + motorcycle_rec_loss

    out_losses = ABC_losses + CBC_losses

    return out_losses

def LightMaskDenoised(Seg_mask, real_vis, Avg_KernelSize, gpu_ids=[]):
    "Denoising of the traffic light mask region with given kernel size of average pooling."
    light_mask_ori = torch.zeros_like(Seg_mask)
    light_mask_ori = torch.where(Seg_mask == 6.0, torch.ones_like(Seg_mask), torch.zeros_like(Seg_mask))
    sky_mask = torch.zeros_like(Seg_mask)
    sky_mask = torch.where(Seg_mask == 10.0, torch.ones_like(Seg_mask), torch.zeros_like(Seg_mask))

    real_img_norm = (real_vis + 1.0) * 0.5
    real_vis_gray = torch.squeeze(.299 * real_img_norm[:,0:1,:,:] + .587 * real_img_norm[:,1:2,:,:] + .114 * real_img_norm[:,2:3,:,:])
    real_vis_sky_region = sky_mask.mul(real_vis_gray)
    h, w = real_vis_gray.size()
    padsize = Avg_KernelSize // 2
    AvgPool_k5 = nn.AvgPool2d(Avg_KernelSize, stride=1, padding=padsize)
    real_vis_light_region = light_mask_ori.mul(real_vis_gray)
    real_vis_pooled = torch.squeeze(AvgPool_k5(real_vis_light_region.expand(1, 1, h, w)))

    "In a noisy traffic light region, if the distance between a given pixel and the average feature of the sky region is "
    "less than the distance between it and the average feature of the neighborhood in which it is located, the pixel has a "
    "high probability of belonging to the sky category, and therefore is set as a noisy pixel."
    if torch.sum(sky_mask) > 0:
        real_vis_sky_mean = torch.sum(real_vis_sky_region) / torch.sum(sky_mask)

        light_sky_dis = light_mask_ori.mul((real_vis_light_region - real_vis_sky_mean) ** 2)
        light_local_dis = light_mask_ori.mul((real_vis_light_region - real_vis_pooled) ** 2)
        sky_local_diff = light_local_dis - light_sky_dis
        sky_noise = torch.zeros_like(Seg_mask)
        sky_noise = torch.where(sky_local_diff > 0, torch.ones_like(Seg_mask), torch.zeros_like(Seg_mask))
        sky_noise_mask = light_mask_ori.mul(sky_noise)
    else:
        sky_noise_mask = torch.zeros_like(Seg_mask)

    light_mask_denoised = F.relu(light_mask_ori - sky_noise_mask)

    ####Filling the cavities inside the mask that are smaller than a certain area
    area_th = torch.sum(light_mask_ori) - torch.sum(light_mask_denoised)
    pre_mask = light_mask_denoised.cpu().numpy()
    pre_mask_rever = pre_mask<=0
    pre_mask_rever = skimage.morphology.remove_small_objects(pre_mask_rever, min_size=area_th.item())
    pre_mask[pre_mask_rever<=0] = 1
    light_mask_refine_tensor = torch.tensor(pre_mask).cuda(gpu_ids)
    out_mask = (torch.ones_like(Seg_mask) - light_mask_ori).mul(Seg_mask) + 6.0 * light_mask_refine_tensor + \
                255.0 * (light_mask_ori - light_mask_refine_tensor)

    return out_mask

def RefineLightMask(Seg_mask, real_vis, gpu_ids=[]):
    "Denoising of the traffic light mask region."
    Segmask_Light_DN_k5 = LightMaskDenoised(Seg_mask, real_vis, 5, gpu_ids)
    Segmask_Light_DN_k3 = LightMaskDenoised(Segmask_Light_DN_k5, real_vis, 3, gpu_ids)
    
    light_mask_DN = torch.zeros_like(Seg_mask)
    light_mask_DN = torch.where(Segmask_Light_DN_k3 == 6.0, torch.ones_like(Seg_mask), torch.zeros_like(Seg_mask))


    return light_mask_DN

def FakeIRPersonLossv2(Seg_mask, fake_IR, real_vis, gpu_ids=[]):
    "Temperature regularization term: Encouraging the min value of the pedestrian region in the fake IR image "
    "to be larger than the mean value of the road region. "

    # b, c, h, w = fake_IR.size()

    b, c, h, w = fake_IR.size()
    _, seg_h, seg_w = Seg_mask.size()
    GT_mask_resize = F.interpolate(Seg_mask.expand(1, 1, seg_h, seg_w).float(), size=[h, w], mode='nearest')
    # real_mask_resize = F.interpolate(real_mask.expand(1, 1, seg_h, seg_w).float(), size=[h, w], mode='nearest')
    GT_mask = GT_mask_resize[0]
    person_mask = torch.zeros_like(GT_mask_resize)
    person_mask = torch.where(GT_mask_resize == 11, torch.ones_like(GT_mask_resize), torch.zeros_like(GT_mask_resize))

    fake_img_norm = (fake_IR + 1.0) * 0.5
    real_img_norm = (real_vis + 1.0) * 0.5
    fake_IR_gray = .299 * fake_img_norm[:,0:1,:,:] + .587 * fake_img_norm[:,1:2,:,:] + .114 * fake_img_norm[:,2:3,:,:]
    fake_mean_fea, fake_cls_tensor, _ = ClsMeanPixelValue(fake_IR_gray, Seg_mask.detach(), 19, gpu_ids)
    if (fake_cls_tensor[11, :] * fake_cls_tensor[0, :]) > 0 :
        person_region = (person_mask.expand_as(fake_IR_gray)).mul(fake_IR_gray)
        non_person_mask = torch.ones_like(person_mask) - person_mask
        person_region_padding1 = person_region + non_person_mask ####To get person region min value
        road_mean_value = (fake_mean_fea[0, :]).detach()
        person_min_value = torch.min(person_region_padding1)
        person_dis_loss = F.relu(road_mean_value - person_min_value) / (road_mean_value + 1e-4)
    else:
        person_dis_loss = 0.0

    return person_dis_loss
def PatchNormFea(input_array, sqrt_patch_num, gpu_ids=[]):
    "Calculate the L2 normalized features for each patch."
    h, w = input_array.size()

    crop_size = h // sqrt_patch_num
    pos_list = list(range(0, h, crop_size))
    patch_num = sqrt_patch_num * sqrt_patch_num
    patch_pixel = crop_size * crop_size
    out_fea_array = torch.zeros(patch_num, patch_pixel).cuda(gpu_ids)
    for p in range(sqrt_patch_num):
        for q in range(sqrt_patch_num):
            idx = p * sqrt_patch_num + q
            pos_h = pos_list[p]
            pos_w = pos_list[q]
            temp_patch = input_array[pos_h:(pos_h + crop_size), pos_w:(pos_w + crop_size)]
            temp_patch_view = temp_patch.reshape(1, patch_pixel)
            out_fea_array[idx, :] = torch.div(temp_patch_view, (torch.norm(temp_patch_view) + 1e-4))

    return out_fea_array

def GetImgHieFea(input_rgb, input_gray, value_th_list, th_num, gpu_ids=[]):
    "Get the hierarchical features of RGB images with different value intervals."
    _, c, h, w = input_rgb.size()
    # th_mask_tensor = torch.zeros(th_num, h, w).cuda(gpu_ids)
    out_rgb_fea_tensor = torch.zeros(th_num, c).cuda(gpu_ids)
    out_dist_pixels_tensor = torch.zeros(th_num, 1).cuda(gpu_ids)
    # mask_squeeze = torch.squeeze(input_mask)
    gray_img_squeeze = torch.squeeze(input_gray)
    GAP = nn.AdaptiveAvgPool2d(1)
    for i in range(th_num):
        temp_mask1 = torch.zeros_like(gray_img_squeeze)
        temp_mask1 = torch.where(gray_img_squeeze < value_th_list[i], torch.ones_like(gray_img_squeeze), torch.zeros_like(gray_img_squeeze))
        temp_mask2 = torch.zeros_like(gray_img_squeeze)
        temp_mask2 = torch.where(gray_img_squeeze < value_th_list[i + 1], torch.ones_like(gray_img_squeeze), torch.zeros_like(gray_img_squeeze))
        th_mask = temp_mask2 - temp_mask1
        mask_pixels = torch.sum(th_mask)
        out_dist_pixels_tensor[i, :] = mask_pixels 
        if mask_pixels != 0:
            rgb_mask = input_rgb.mul(th_mask.expand_as(input_rgb))
            out_rgb_fea_tensor[i, :] = (torch.squeeze(GAP(rgb_mask) * h * w)) / mask_pixels

    return out_rgb_fea_tensor, out_dist_pixels_tensor

def bhw_to_onehot(bhw_tensor1, num_classes, gpu_ids):
    """
    Args:
        bhw_tensor: b,h,w
        num_classes: 20 (19 + uncertain_clsidx)
    Returns: b,num_classes,h,w
    """
    assert bhw_tensor1.ndim == 3, bhw_tensor1.shape
    # assert num_classes > bhw_tensor.max(), torch.unique(bhw_tensor)
    # bhw_tensor = bhw_tensor1
    # bhw_tensor[(bhw_tensor == 255)] = 5

    bhw_tensor = torch.zeros_like(bhw_tensor1).cuda(gpu_ids)
    uncertain_clsidx = num_classes - 1
    bhw_tensor = torch.where(bhw_tensor1 == 255, uncertain_clsidx, bhw_tensor1)
    # one_hot = torch.eye(num_classes).index_select(dim=0, index=bhw_tensor.reshape(-1)).cuda(gpu_ids)
    one_hot = torch.eye(num_classes).cuda(gpu_ids).index_select(dim=0, index=bhw_tensor.reshape(-1))
    one_hot = one_hot.reshape(*bhw_tensor.shape, num_classes)
    out_tensor = one_hot.permute(0, 3, 1, 2)

    return out_tensor[:, :-1, :, :]

def SemEdgeLoss(seg_tensor, GT_mask, num_classes, gpu_ids):
    "Encourage semantic edge prediction consistent with GT."

    sm = torch.nn.Softmax(dim = 1)
    pred_sm = sm(seg_tensor)
    GT2onehot = bhw_to_onehot(GT_mask, num_classes+1, gpu_ids)
    AvgPool_k3 = nn.AvgPool2d(3, stride=1, padding=1)
    pred_semedge = torch.abs(pred_sm - AvgPool_k3(pred_sm))
    GT_semedge = torch.abs(GT2onehot - AvgPool_k3(GT2onehot))
    if torch.sum(GT_semedge) > 0:
        losses = torch.sum(torch.abs(GT_semedge.detach() - pred_semedge)) / torch.sum(GT_semedge.detach())
    else:
        losses = 0.0

    return losses

def GetFeaMatrixCenter(fea_array, cluster_num, max_iter, gpu_ids):
    "Obtain the central features of each cluster of the feature matrix."
    if gpu_ids == 0:
        # kmeans
        _, cluster_centers = kmeans(
            X=fea_array, num_clusters=cluster_num, distance='cosine', device=torch.device('cuda:0'), tqdm_flag=False, iter_limit=max_iter
        )
    elif gpu_ids == 1:
        # kmeans
        _, cluster_centers = kmeans(
            X=fea_array, num_clusters=cluster_num, distance='cosine', device=torch.device('cuda:1'), tqdm_flag=False, iter_limit=max_iter
        )
    elif gpu_ids == 2:
        # kmeans
        _, cluster_centers = kmeans(
            X=fea_array, num_clusters=cluster_num, distance='cosine', device=torch.device('cuda:2'), tqdm_flag=False, iter_limit=max_iter
        )
    elif gpu_ids == 3:
        # kmeans
        _, cluster_centers = kmeans(
            X=fea_array, num_clusters=cluster_num, distance='cosine', device=torch.device('cuda:3'), tqdm_flag=False, iter_limit=max_iter
        )
    else:
        # kmeans
        _, cluster_centers = kmeans(
            X=fea_array, num_clusters=cluster_num, distance='cosine', device=torch.device('cuda:4'), tqdm_flag=False, iter_limit=max_iter
        )

    return cluster_centers.cuda(gpu_ids)

def ClsACALoss(real_vis_fea, cls_mask_real, fake_vis_fea, cls_mask_fake, fea_dim, cluster_num, max_iter, gpu_ids):
    "Calculating adaptive collaborative attention loss for a single class."

    real_fea_cls_masked = real_vis_fea.mul(cls_mask_real.expand_as(real_vis_fea))
    real_fea_cls_matrix = (real_fea_cls_masked.view(fea_dim, -1)).t() ### N * c
    nonZeroRows = torch.abs(real_fea_cls_matrix).sum(dim=1) > 0
    real_fea_cls_matrix = real_fea_cls_matrix[nonZeroRows]
    cls_cluster_center_array = GetFeaMatrixCenter(real_fea_cls_matrix, cluster_num, max_iter, gpu_ids) ###Cn * c
    cls_center_fea_norm = F.normalize(cls_cluster_center_array, p=2, dim=1)
    real_fea_cls_matrix_norm = F.normalize(real_fea_cls_matrix, p=2, dim=1)
    cls_sim_map_real = torch.mm(real_fea_cls_matrix_norm, cls_center_fea_norm.t()) ### Np * Cn
    cls_sim_map_max_real = torch.max(cls_sim_map_real, dim=1)
    cls_fea_sim_mean_real = torch.mean(cls_sim_map_max_real[0])
    sim_map_clustermax_real = torch.max(cls_sim_map_real, dim=0)
    fea_sim_clustermean_real = torch.mean(sim_map_clustermax_real[0])

    fake_fea_cls_masked = fake_vis_fea.mul(cls_mask_fake.expand_as(fake_vis_fea))
    fake_fea_cls_matrix = (fake_fea_cls_masked.view(fea_dim, -1)).t() ### N * c
    fake_fea_cls_matrix_norm = F.normalize(fake_fea_cls_matrix, p=2, dim=1)
    cls_sim_map_fake = torch.mm(fake_fea_cls_matrix_norm, cls_center_fea_norm.t()) ### N * Cn
    cls_sim_map_max_fake = torch.max(cls_sim_map_fake, dim=1)
    cls_fea_sim_mean_fake = torch.sum(cls_sim_map_max_fake[0]) / torch.sum(cls_mask_fake.detach())
    sim_map_clustermax_fake = torch.max(cls_sim_map_fake, dim=0)
    fea_sim_clustermean_fake = torch.mean(sim_map_clustermax_fake[0])
    loss_cls_sim = F.relu(0.9 * cls_fea_sim_mean_real.detach() - cls_fea_sim_mean_fake)
    loss_cls_div = F.relu(0.9 * fea_sim_clustermean_real.detach() - fea_sim_clustermean_fake)
    losses = loss_cls_sim + loss_cls_div

    return losses

def AdaColAttLoss(real_vis_mask, real_vis_fea, fake_vis_mask, fake_vis_fea, cluster_num, max_iter, gpu_ids):
    "Adaptive Collaborative Attention Loss."
    
    _, c, h, w = real_vis_fea.size()
    real_vis_mask_resize = F.interpolate(real_vis_mask.expand(1, 1, 256, 256).float(), size=[h, w], mode='nearest')
    fake_vis_mask_resize = F.interpolate(fake_vis_mask.expand(1, 1, 256, 256).float(), size=[h, w], mode='nearest')
    Light_mask_real = torch.zeros_like(real_vis_mask_resize)
    Sign_mask_real = torch.zeros_like(real_vis_mask_resize)
    Person_mask_real = torch.zeros_like(real_vis_mask_resize)
    Vehicle_mask_real = torch.zeros_like(real_vis_mask_resize)
    Motor_mask_real = torch.zeros_like(real_vis_mask_resize)

    Light_mask_real = torch.where(real_vis_mask_resize == 6.0, torch.ones_like(real_vis_mask_resize), torch.zeros_like(real_vis_mask_resize))
    Sign_mask_real = torch.where(real_vis_mask_resize == 7.0, torch.ones_like(real_vis_mask_resize), torch.zeros_like(real_vis_mask_resize))
    Person_mask_real = torch.where(real_vis_mask_resize == 11.0, torch.ones_like(real_vis_mask_resize), torch.zeros_like(real_vis_mask_resize))
    Vehicle_mask_real = torch.where((real_vis_mask_resize > 12.0) & (real_vis_mask_resize < 17.0), torch.ones_like(real_vis_mask_resize), torch.zeros_like(real_vis_mask_resize))
    Motor_mask_real = torch.where(real_vis_mask_resize == 17.0, torch.ones_like(real_vis_mask_resize), torch.zeros_like(real_vis_mask_resize))

    Light_mask_fake = torch.zeros_like(fake_vis_mask_resize)
    Sign_mask_fake = torch.zeros_like(fake_vis_mask_resize)
    Person_mask_fake = torch.zeros_like(fake_vis_mask_resize)
    Vehicle_mask_fake = torch.zeros_like(fake_vis_mask_resize)
    Motor_mask_fake = torch.zeros_like(fake_vis_mask_resize)

    Light_mask_fake = torch.where(fake_vis_mask_resize == 6.0, torch.ones_like(fake_vis_mask_resize), torch.zeros_like(fake_vis_mask_resize))
    Sign_mask_fake = torch.where(fake_vis_mask_resize == 7.0, torch.ones_like(fake_vis_mask_resize), torch.zeros_like(fake_vis_mask_resize))
    Person_mask_fake = torch.where(fake_vis_mask_resize == 11.0, torch.ones_like(fake_vis_mask_resize), torch.zeros_like(fake_vis_mask_resize))
    Vehicle_mask_fake = torch.where((fake_vis_mask_resize > 12.0) & (fake_vis_mask_resize < 17.0), torch.ones_like(fake_vis_mask_resize), torch.zeros_like(fake_vis_mask_resize))
    Motor_mask_fake = torch.where(fake_vis_mask_resize == 17.0, torch.ones_like(fake_vis_mask_resize), torch.zeros_like(fake_vis_mask_resize))

    if (torch.sum(Light_mask_real) > cluster_num) & (torch.sum(Light_mask_fake) > cluster_num):
        loss_light = ClsACALoss(real_vis_fea, Light_mask_real, fake_vis_fea, Light_mask_fake, c, cluster_num, max_iter, gpu_ids)
        idx_light = 1.0
    else:
        loss_light = 0.0
        idx_light = 0.0

    if (torch.sum(Sign_mask_real) > cluster_num) & (torch.sum(Sign_mask_fake) > cluster_num):
        loss_sign = ClsACALoss(real_vis_fea, Sign_mask_real, fake_vis_fea, Sign_mask_fake, c, cluster_num, max_iter, gpu_ids)
        idx_sign = 1.0
    else:
        loss_sign = 0.0
        idx_sign = 0.0

    if (torch.sum(Person_mask_real) > cluster_num) & (torch.sum(Person_mask_fake) > cluster_num):
        loss_person = ClsACALoss(real_vis_fea, Person_mask_real, fake_vis_fea, Person_mask_fake, c, cluster_num, max_iter, gpu_ids)
        idx_person = 1.0
    else:
        loss_person = 0.0
        idx_person = 0.0

    if (torch.sum(Vehicle_mask_real) > cluster_num) & (torch.sum(Vehicle_mask_fake) > cluster_num):
        loss_vehicle = ClsACALoss(real_vis_fea, Vehicle_mask_real, fake_vis_fea, Vehicle_mask_fake, c, cluster_num, max_iter, gpu_ids)
        idx_vehicle = 1.0
    else:
        loss_vehicle = 0.0
        idx_vehicle = 0.0

    if (torch.sum(Motor_mask_real) > cluster_num) & (torch.sum(Motor_mask_fake) > cluster_num):
        loss_motor = ClsACALoss(real_vis_fea, Motor_mask_real, fake_vis_fea, Motor_mask_fake, c, cluster_num, max_iter, gpu_ids)
        idx_motor = 1.0
    else:
        loss_motor = 0.0
        idx_motor = 0.0

    obj_cls_num = idx_light + idx_sign + idx_person + idx_vehicle + idx_motor
    # obj_cls_num = idx_sign + idx_person + idx_vehicle + idx_motor
    if obj_cls_num > 0:
        losses = (loss_light + loss_sign + loss_person + loss_vehicle + loss_motor) / obj_cls_num
        # losses = (loss_sign + loss_person + loss_vehicle + loss_motor) / obj_cls_num
    else:
        losses = 0.0

    return losses

def FakeIRFGMergeMask(vis_segmask, IR_seg_tensor, gpu_ids):
    "Selecting a suitable foreground mask from the fake IR image and fuse it with the real IR image."

    sm = torch.nn.Softmax(dim = 1)
    pred_sm1 = sm(IR_seg_tensor.detach())
    pred_max_tensor1 = torch.max(pred_sm1, dim=1)
    pred_max_category1 = pred_max_tensor1[1]

    IR_segmask = pred_max_category1.float()
    vis_FG_idx_list = [6, 7, 11, 13, 14, 15, 16, 17]
    large_FG_list = [15, 16]
    traffic_sign_list = [6, 7]
    vis_GT_segmask = torch.squeeze(vis_segmask).float().detach().cpu().numpy()
    real_IR_segmask = torch.squeeze(IR_segmask).float().detach().cpu().numpy()
    IR_road_mask = np.zeros_like(real_IR_segmask)
    IR_road_mask = np.where(real_IR_segmask < 2.0, 1.0, 0.0)
    output_FG_Mask = np.zeros_like(real_IR_segmask)
    for i in range(len(vis_FG_idx_list)):
        temp_mask = np.zeros_like(vis_GT_segmask)
        temp_mask = np.where(vis_GT_segmask == vis_FG_idx_list[i], 1.0, 0.0)
        label_connect, num = measure.label(temp_mask, connectivity=2, background=0, return_num=True)
        for j in range(1, num+1):
            "Since background index is 0, the num is num+1."
            temp_connect_mask = np.zeros_like(label_connect)
            temp_connect_mask = np.where(label_connect == j, 1.0, 0.0)
            road_mask_prod = temp_connect_mask * IR_road_mask
            # IoU_th = 0.5 * np.sum(temp_connect_mask)
            if np.sum(temp_connect_mask) > 50:
                if vis_FG_idx_list[i] in traffic_sign_list:
                    output_FG_Mask += temp_connect_mask
                elif vis_FG_idx_list[i] in large_FG_list:
                    IoU_th = 0.1 * np.sum(temp_connect_mask)
                    if np.sum(road_mask_prod) > IoU_th:
                        output_FG_Mask += temp_connect_mask
                else:
                    IoU_th = 0.1 * np.sum(temp_connect_mask)
                    if np.sum(road_mask_prod) > IoU_th:
                        output_FG_Mask += temp_connect_mask
    # print(np.sum(output_FG_Mask))

    return torch.tensor(output_FG_Mask).cuda(gpu_ids).expand(1, 3, 256, 256)


class Vgg16(nn.Module):
    def __init__(self, gpu_ids=[]):
        super(Vgg16, self).__init__()
        features = models.vgg16(pretrained=True).features.cuda(gpu_ids)
        self.to_relu_1_2 = nn.Sequential() 
        self.to_relu_2_2 = nn.Sequential() 
        self.to_relu_3_3 = nn.Sequential()
        self.to_relu_4_3 = nn.Sequential()
        self.to_relu_5_3 = nn.Sequential()

        for x in range(4):
            self.to_relu_1_2.add_module(str(x), features[x])
        for x in range(4, 9):
            self.to_relu_2_2.add_module(str(x), features[x])
        for x in range(9, 16):
            self.to_relu_3_3.add_module(str(x), features[x])
        for x in range(16, 23):
            self.to_relu_4_3.add_module(str(x), features[x])
        # for x in range(23, 30):
        #     self.to_relu_5_3.add_module(str(x), features[x])
        
        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        h = self.to_relu_1_2(x)
        h_relu_1_2 = h
        h = self.to_relu_2_2(h)
        h_relu_2_2 = h
        h = self.to_relu_3_3(h)
        h_relu_3_3 = h
        h = self.to_relu_4_3(h)
        h_relu_4_3 = h
        # h = self.to_relu_5_3(h)
        # h_relu_5_3 = h
        out = (h_relu_1_2, h_relu_2_2, h_relu_3_3, h_relu_4_3)
        return out

def compute_vgg_loss(img, target, gpu_ids=[]):
    # img_vgg = vgg_preprocess(img)
    # target_vgg = vgg_preprocess(target)
    vgg = Vgg16(gpu_ids)
    vgg.eval()
    for param in vgg.parameters():
        param.requires_grad = False
    loss_mse = torch.nn.MSELoss()
    img_fea = vgg(img)
    target_fea = vgg(target)
    content_loss = 0.0
    for j in range(4):
        content_loss += loss_mse(img_fea[j], target_fea[j])

    return content_loss * 0.25

class Get_gradmag_gray(nn.Module):
    "To obtain the magnitude values of the gradients at each position."
    def __init__(self):
        super(Get_gradmag_gray, self).__init__()
        kernel_v = [[0, -1, 0], 
                    [0, 0, 0], 
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0], 
                    [-1, 0, 1], 
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data = kernel_h, requires_grad = False).cuda()
        self.weight_v = nn.Parameter(data = kernel_v, requires_grad = False).cuda()

    def forward(self, x):
        x_norm = (x + 1) / 2
        x_norm = (.299*x_norm[:,0:1,:,:] + .587*x_norm[:,1:2,:,:] + .114*x_norm[:,2:3,:,:])
        x0_v = F.conv2d(x_norm, self.weight_v, padding = 1)
        x0_h = F.conv2d(x_norm, self.weight_h, padding = 1)

        x_gradmagn = torch.sqrt(torch.pow(x0_v, 2) + torch.pow(x0_h, 2) + 1e-6)

        return x_gradmagn

def StruGradAligLoss(real_IR_edgemap, fake_vis_gradmap, sqrt_patch_num, gradient_th):
    "SGA Loss. The ratio of the gradient at the edge location to the maximum gradient in "
    "its neighborhood is encouraged to be greater than a given threshold."

    b, c, h, w = fake_vis_gradmap.size()
    # patch_num = sqrt_patch_num * sqrt_patch_num
    AAP_module = nn.AdaptiveAvgPool2d(sqrt_patch_num)
    real_IR_edgemap_pooling = AAP_module(real_IR_edgemap.expand_as(fake_vis_gradmap))
    if torch.sum(real_IR_edgemap) > 0:
        pooling_array = real_IR_edgemap_pooling[0].detach().cpu().numpy()
        h_nonzero, w_nonzero = np.nonzero(pooling_array[0])
        patch_idx_rand = np.random.randint(0, len(h_nonzero))
        patch_idx_x = h_nonzero[patch_idx_rand]
        patch_idx_y = w_nonzero[patch_idx_rand]
        crop_size = h // sqrt_patch_num
        pos_list = list(range(0, h, crop_size))

        pos_h = pos_list[patch_idx_x]
        pos_w = pos_list[patch_idx_y]
        # rand_patch = self.Tensor(b, c, crop_size, crop_size)
        rand_edgemap_patch = real_IR_edgemap[:, pos_h:(pos_h + crop_size), pos_w:(pos_w + crop_size)]
        rand_gradmap_patch = fake_vis_gradmap[:, :, pos_h:(pos_h + crop_size), pos_w:(pos_w + crop_size)]

        sum_edge_pixels = torch.sum(rand_edgemap_patch) + 1
        # print('Sum_edge_pixels of IR edge map is: ', sum_edge_pixels.detach().cpu().numpy())
        fake_grad_norm = rand_gradmap_patch / torch.max(rand_gradmap_patch)
        losses = (torch.sum(F.relu(gradient_th * rand_edgemap_patch - fake_grad_norm))) / sum_edge_pixels
    else:
        losses = 0

    return losses

def FakeIRFGMergeMaskv3(vis_segmask, IR_seg_tensor, real_vis, fake_IR, gpu_ids):
    "Selecting a suitable foreground mask from the fake IR image and fusing it with the real IR image, "
    "and keeping the original foreground area unchanged. Vertical flipping of traffic light areas that "
    "meet the conditions to reduce the wrong color of traffic lights due to uneven distribution of red "
    "and green lights."

    sm = torch.nn.Softmax(dim = 1)
    pred_sm1 = sm(IR_seg_tensor.detach())
    pred_max_tensor1 = torch.max(pred_sm1, dim=1)
    pred_max_category1 = pred_max_tensor1[1]

    IR_segmask = pred_max_category1.float()
    _, _, h, w = IR_seg_tensor.size()
    vis_FG_idx_list = [6, 7, 17]
    large_FG_list = [15, 16]
    traffic_sign_list = [6, 7]
    vis_GT_segmask = torch.squeeze(vis_segmask).float().detach().cpu().numpy()
    real_IR_segmask = torch.squeeze(IR_segmask).float().detach().cpu().numpy()
    real_vis_numpy = torch.squeeze(real_vis).detach().cpu().numpy()
    fake_IR_numpy = torch.squeeze(fake_IR).detach().cpu().numpy()
    IR_road_mask = np.zeros_like(real_IR_segmask)
    IR_road_mask = np.where(real_IR_segmask < 2.0, 1.0, 0.0)
    IR_FG1_mask = np.zeros_like(real_IR_segmask)
    IR_FG1_mask = np.where(real_IR_segmask > 10.0, 1.0, 0.0)
    IR_light_mask = np.zeros_like(real_IR_segmask)
    IR_light_mask = np.where(real_IR_segmask == 6.0, 1.0, 0.0)
    IR_sign_mask = np.zeros_like(real_IR_segmask)
    IR_sign_mask = np.where(real_IR_segmask == 7.0, 1.0, 0.0)
    IR_FG_mask = IR_FG1_mask + IR_light_mask + IR_sign_mask
    output_FG_Mask = np.zeros_like(real_IR_segmask)
    output_FG_Mask_ori = np.zeros_like(real_IR_segmask)
    output_HL_Mask = np.zeros_like(real_IR_segmask)
    output_Light_TopMask = np.zeros_like(real_IR_segmask)
    output_Light_BottomMask = np.zeros_like(real_IR_segmask)
    output_FG_FakeIR = np.zeros_like(fake_IR_numpy)
    output_FG_RealVis = np.zeros_like(real_vis_numpy)
    for i in range(len(vis_FG_idx_list)):
        
        #######erode
        temp_mask_ori = np.zeros_like(vis_GT_segmask)
        temp_mask_ori = np.where(vis_GT_segmask == vis_FG_idx_list[i], 1.0, 0.0)
        max_pool_k3 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        temp_mask_erode = -max_pool_k3(-torch.Tensor(temp_mask_ori).unsqueeze(0).unsqueeze(0))
        temp_mask = torch.squeeze(temp_mask_erode).numpy()
        #########
        label_connect, num = measure.label(temp_mask, connectivity=2, background=0, return_num=True)
        for j in range(1, num+1):
            "Since background index is 0, the num is num+1."
            temp_connect_mask = np.zeros_like(label_connect)
            temp_connect_mask = np.where(label_connect == j, 1.0, 0.0)
            road_mask_prod = temp_connect_mask * IR_road_mask
            FG_mask_overlap = temp_connect_mask * IR_FG_mask
            fake_IR_masked = temp_connect_mask * fake_IR_numpy
            real_vis_masked = temp_connect_mask * real_vis_numpy
            # IoU_th = 0.5 * np.sum(temp_connect_mask)
            if np.sum(FG_mask_overlap) == 0:
                if np.sum(temp_connect_mask) > 50:
                    if vis_FG_idx_list[i] in traffic_sign_list:
                        if i == 0:
                            temp_FG_Mask, temp_FG_FakeIR, temp_FG_RealVis, temp_highlight_mask, temp_TopMask, temp_BottomMask = ObtainTLightMixedMask(temp_connect_mask, fake_IR_masked, real_vis_masked, h)
                            output_FG_Mask += temp_FG_Mask
                            output_FG_FakeIR += temp_FG_FakeIR
                            output_FG_RealVis += temp_FG_RealVis
                            output_HL_Mask += temp_highlight_mask
                            output_Light_TopMask += temp_TopMask
                            output_Light_BottomMask += temp_BottomMask
                        else:
                            output_FG_Mask += temp_connect_mask
                            output_FG_FakeIR += fake_IR_masked
                            output_FG_RealVis += real_vis_masked
                            output_FG_Mask_ori += temp_connect_mask
                    else:
                        IoU_th = 0.1 * np.sum(temp_connect_mask)
                        if np.sum(road_mask_prod) > IoU_th:
                            output_FG_Mask += temp_connect_mask
                            output_FG_FakeIR += fake_IR_masked
                            output_FG_RealVis += real_vis_masked
                            output_FG_Mask_ori += temp_connect_mask
    # print(np.sum(output_FG_Mask))
    
    out_FG_mask = torch.tensor(output_FG_Mask).cuda(gpu_ids).expand(1, 3, 256, 256)
    out_FG_FakeIR = torch.tensor(output_FG_FakeIR).cuda(gpu_ids).expand(1, 3, 256, 256)
    out_FG_RealVis = torch.tensor(output_FG_RealVis).cuda(gpu_ids).expand(1, 3, 256, 256)
    out_FG_mask_ori = torch.tensor(output_FG_Mask_ori).cuda(gpu_ids).expand(1, 3, 256, 256)

    #########Flip fake mask
    vis_GT_flip = torch.flip(torch.squeeze(vis_segmask), dims=[1])
    vis_GT_flip_segmask = vis_GT_flip.float().detach().cpu().numpy()
    output_FG_Mask_Flip = np.zeros_like(real_IR_segmask)
    output_FG_FakeIR_Flip = np.zeros_like(fake_IR_numpy)
    output_FG_RealVis_Flip = np.zeros_like(real_vis_numpy)
    # output_HL_Mask_Flip = np.zeros_like(real_IR_segmask)
    fake_IR_numpy_Flip = torch.squeeze(torch.flip(fake_IR, dims=[3])).detach().cpu().numpy()
    real_vis_numpy_Flip = torch.squeeze(torch.flip(real_vis, dims=[3])).detach().cpu().numpy()
    IR_FG_mask_update = IR_FG_mask + output_FG_Mask
    IR_road_mask_update = IR_road_mask - IR_road_mask * output_FG_Mask
    for i in range(len(vis_FG_idx_list)):
        
        #######erode
        temp_mask_ori = np.zeros_like(vis_GT_flip_segmask)
        temp_mask_ori = np.where(vis_GT_flip_segmask == vis_FG_idx_list[i], 1.0, 0.0)
        max_pool_k3 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        temp_mask_erode = -max_pool_k3(-torch.Tensor(temp_mask_ori).unsqueeze(0).unsqueeze(0))
        temp_mask = torch.squeeze(temp_mask_erode).numpy()
        #########
        label_connect, num = measure.label(temp_mask, connectivity=2, background=0, return_num=True)
        for j in range(1, num+1):
            "Since background index is 0, the num is num+1."
            temp_connect_mask = np.zeros_like(label_connect)
            temp_connect_mask = np.where(label_connect == j, 1.0, 0.0)
            road_mask_prod = temp_connect_mask * IR_road_mask_update
            FG_mask_overlap = temp_connect_mask * IR_FG_mask_update

            fake_IR_masked_Flip = temp_connect_mask * fake_IR_numpy_Flip
            real_vis_masked_Flip = temp_connect_mask * real_vis_numpy_Flip
            # IoU_th = 0.5 * np.sum(temp_connect_mask)
            if np.sum(FG_mask_overlap) == 0:
                if np.sum(temp_connect_mask) > 50:
                    if vis_FG_idx_list[i] in traffic_sign_list:
                        if i == 0:
                            temp_FG_Mask2, temp_FG_FakeIR2, temp_FG_RealVis2, temp_highlight_mask2, temp_TopMask2, temp_BottomMask2 = ObtainTLightMixedMask(temp_connect_mask, fake_IR_masked_Flip, real_vis_masked_Flip, h)
                            output_FG_Mask_Flip += temp_FG_Mask2
                            output_FG_FakeIR_Flip += temp_FG_FakeIR2
                            output_FG_RealVis_Flip += temp_FG_RealVis2
                            output_HL_Mask += temp_highlight_mask2
                            output_Light_TopMask += temp_TopMask2
                            output_Light_BottomMask += temp_BottomMask2
                        else:
                            output_FG_Mask_Flip += temp_connect_mask
                            output_FG_FakeIR_Flip += fake_IR_masked_Flip
                            output_FG_RealVis_Flip += real_vis_masked_Flip
                    else:
                        IoU_th = 0.1 * np.sum(temp_connect_mask)
                        if np.sum(road_mask_prod) > IoU_th:
                            output_FG_Mask_Flip += temp_connect_mask
                            output_FG_FakeIR_Flip += fake_IR_masked_Flip
                            output_FG_RealVis_Flip += real_vis_masked_Flip
    
    out_FG_mask_flip = torch.tensor(output_FG_Mask_Flip).cuda(gpu_ids).expand(1, 3, 256, 256)
    out_FG_FakeIR_flip = torch.tensor(output_FG_FakeIR_Flip).cuda(gpu_ids).expand(1, 3, 256, 256)
    out_FG_RealVis_flip = torch.tensor(output_FG_RealVis_Flip).cuda(gpu_ids).expand(1, 3, 256, 256)
    out_HL_mask = torch.tensor(output_HL_Mask).cuda(gpu_ids).expand(1, 3, 256, 256)
    out_Light_TopMask = torch.tensor(output_Light_TopMask).cuda(gpu_ids).expand(1, 256, 256)
    out_Light_BottomMask = torch.tensor(output_Light_BottomMask).cuda(gpu_ids).expand(1, 256, 256)
    out_Light_mask = torch.cat([out_Light_TopMask, out_Light_BottomMask], dim=0)


    return out_FG_mask, out_FG_FakeIR, out_FG_RealVis, out_FG_mask_flip, out_FG_FakeIR_flip, \
            out_FG_RealVis_flip, out_FG_mask_ori, out_HL_mask, out_Light_mask

def ObtainTLightMixedMask(temp_connect_mask, fake_IR_masked, real_vis_masked, patch_height):
    "Extraction of traffic light related areas and masks."

    temp_connect_mask_row_sum = np.sum(temp_connect_mask, axis=1)
    temp_connect_mask_col_sum = np.sum(temp_connect_mask, axis=0)
    region_AspectRatio = np.max(temp_connect_mask_col_sum) / np.max(temp_connect_mask_row_sum)
    _, temp_r2g_area_ori, temp_red_mask_ori = Red2Green(real_vis_masked, temp_connect_mask)
    _, temp_g2r_area_ori, temp_green_mask_ori = Green2Red(real_vis_masked, temp_connect_mask)

    row_pos_array = np.matmul(np.arange(patch_height).reshape((patch_height,1)), np.ones((1,patch_height)))
    mask_pos = temp_connect_mask * row_pos_array
    mask_pos_padding_h = temp_connect_mask * row_pos_array + \
                        (np.ones_like(temp_connect_mask) - temp_connect_mask) * patch_height
    mask_pos_row_min = int(mask_pos_padding_h.min())
    mask_pos_row_max = int(mask_pos.max())
    mask_mid_row = (mask_pos_row_min + mask_pos_row_max) // 2
    top_mask = np.zeros_like(temp_connect_mask)
    top_mask[mask_pos_row_min:(mask_mid_row+1), :] = 1.0
    bottom_mask = np.ones_like(temp_connect_mask) - top_mask
    # IoU_th = 0.8
    IoU_th = 0.5
    if region_AspectRatio > 1.75:
        "When the aspect ratio of a given traffic light instance is greater than a given threshold, vertical flip and "
        "transition between red and green lights are performed."
        ver_flip_idx = torch.rand(1)
        temp_VerFlip_mask, temp_VerFlip_fakeIR, temp_VerFlip_realVis = LocalVerticalFlip(temp_connect_mask, fake_IR_masked, real_vis_masked, mask_pos_row_min, mask_pos_row_max)
        temp_r2g_vis, temp_r2g_area, temp_red_mask = Red2Green(temp_VerFlip_realVis, temp_VerFlip_mask)
        temp_g2r_vis, temp_g2r_area, temp_green_mask = Green2Red(temp_VerFlip_realVis, temp_VerFlip_mask)
        DLS_idx = np.random.random(1)
        Decay_factor = 1.0
        if ver_flip_idx > 0.5:
            
            #########Generating double light spot
            # DLS_idx = torch.rand(1)
            # Decay_factor = torch.rand(1)
            if temp_r2g_area > temp_g2r_area:
                vis_GT_masked = (temp_r2g_vis - 0.5) * 2.0
                output_FG_RealVis = vis_GT_masked
                output_highlight_mask = temp_red_mask
                
                if DLS_idx > 0.5:
                    "In the traffic light instance of NTIR image, a double light spot situation (i.e., the temperature of "
                    "both the illuminated main signal and the unilluminated auxiliary signal is higher) is usually presented "
                    "due to the high frequency of illumination of the red and green lights."

                    mask_vertical_IoU = ComIoUNumpy(temp_VerFlip_mask, temp_connect_mask)
                    if mask_vertical_IoU > IoU_th:
                        "When the traffic light instance mask is approximately rectangular, i.e., the IoU of the vertically "
                        "flipped mask with respect to the original mask is greater than a given threshold, a two-spot traffic "
                        "light is synthesized in the pseudo-NTIR image."

                        ###Synthesis of fake NTIR with higher temperatures for both red and green lights.
                        # print('Case1.')
                        top_fake_IR_masked = (0.25 + 0.5 * Decay_factor) * (top_mask * temp_connect_mask * fake_IR_masked)
                        bottom_fake_IR_masked = bottom_mask * temp_VerFlip_mask * temp_VerFlip_fakeIR
                        output_FG_FakeIR = top_fake_IR_masked + bottom_fake_IR_masked
                        fused_mask = top_mask * temp_connect_mask + bottom_mask * temp_VerFlip_mask
                        output_FG_Mask = temp_VerFlip_mask * fused_mask
                    else:
                        # print('Case2.')
                        output_FG_Mask = temp_VerFlip_mask
                        output_FG_FakeIR = temp_VerFlip_fakeIR
                else:
                    # print('Case3.')
                    output_FG_Mask = temp_VerFlip_mask
                    output_FG_FakeIR = temp_VerFlip_fakeIR

            else:
                vis_GT_masked = (temp_g2r_vis - 0.5) * 2.0
                output_FG_RealVis = vis_GT_masked
                output_highlight_mask = temp_green_mask

                if DLS_idx > 0.5:
                    mask_vertical_IoU = ComIoUNumpy(temp_VerFlip_mask, temp_connect_mask)
                    if mask_vertical_IoU > IoU_th:
                        ###Synthesis of fake NTIR with higher temperatures for both red and green lights.
                        # print('Case4.')
                        bottom_fake_IR_masked = (0.25 + 0.5 * Decay_factor) * (bottom_mask * temp_connect_mask * fake_IR_masked)
                        top_fake_IR_masked = top_mask * temp_VerFlip_mask * temp_VerFlip_fakeIR

                        output_FG_FakeIR = top_fake_IR_masked + bottom_fake_IR_masked
                        fused_mask = bottom_mask * temp_connect_mask + top_mask * temp_VerFlip_mask
                        output_FG_Mask = temp_VerFlip_mask * fused_mask
                    else:
                        # print('Case5.')
                        output_FG_Mask = temp_VerFlip_mask
                        output_FG_FakeIR = temp_VerFlip_fakeIR
                else:
                    # print('Case6.')
                    output_FG_Mask = temp_VerFlip_mask
                    output_FG_FakeIR = temp_VerFlip_fakeIR
            ####################
        else:
            output_FG_RealVis = real_vis_masked
            # print('Case2. \n')
            if temp_r2g_area_ori > temp_g2r_area_ori:
                output_highlight_mask = temp_red_mask_ori

                if DLS_idx > 0.5:
                    mask_vertical_IoU = ComIoUNumpy(temp_VerFlip_mask, temp_connect_mask)
                    if mask_vertical_IoU > IoU_th:
                        ###Synthesis of fake NTIR with higher temperatures for both red and green lights.
                        
                        top_fake_IR_masked = top_mask * temp_connect_mask * fake_IR_masked
                        bottom_fake_IR_masked = (0.25 + 0.5 * Decay_factor) * (bottom_mask * temp_VerFlip_mask * temp_VerFlip_fakeIR)
                        output_FG_FakeIR = top_fake_IR_masked + bottom_fake_IR_masked
                        fused_mask = top_mask * temp_connect_mask + bottom_mask * temp_VerFlip_mask
                        output_FG_Mask = temp_connect_mask * fused_mask
                    else:
                        output_FG_Mask = temp_connect_mask
                        output_FG_FakeIR = fake_IR_masked
                else:
                    output_FG_Mask = temp_connect_mask
                    output_FG_FakeIR = fake_IR_masked
            else:
                output_highlight_mask = temp_green_mask_ori

                if DLS_idx > 0.5:
                    mask_vertical_IoU = ComIoUNumpy(temp_VerFlip_mask, temp_connect_mask)
                    if mask_vertical_IoU > IoU_th:
                        ###Synthesis of fake NTIR with higher temperatures for both red and green lights.
                        
                        top_fake_IR_masked = (0.25 + 0.5 * Decay_factor) * (top_mask * temp_VerFlip_mask * temp_VerFlip_fakeIR)
                        bottom_fake_IR_masked = bottom_mask * temp_connect_mask * fake_IR_masked
                        output_FG_FakeIR = top_fake_IR_masked + bottom_fake_IR_masked
                        fused_mask = top_mask * temp_VerFlip_mask + bottom_mask * temp_connect_mask
                        output_FG_Mask = temp_connect_mask * fused_mask
                    else:
                        output_FG_Mask = temp_connect_mask
                        output_FG_FakeIR = fake_IR_masked
                else:
                    output_FG_Mask = temp_connect_mask
                    output_FG_FakeIR = fake_IR_masked
    else:
        output_FG_Mask = temp_connect_mask
        output_FG_FakeIR = fake_IR_masked
        output_FG_RealVis = real_vis_masked
        if temp_r2g_area_ori > temp_g2r_area_ori:
            output_highlight_mask = temp_red_mask_ori
        else:
            output_highlight_mask = temp_green_mask_ori

    output_FG_top_mask = output_FG_Mask * top_mask
    output_FG_bottom_mask = output_FG_Mask * bottom_mask
    # print(output_FG_FakeIR.shape)
    return output_FG_Mask, output_FG_FakeIR, output_FG_RealVis, output_highlight_mask, output_FG_top_mask, output_FG_bottom_mask

def ComIoUNumpy(input_mask1, input_mask2):
    "input_mask:h*w. Numpy array."
    mask_inter = input_mask1 * input_mask2
    mask_fused = input_mask1 + input_mask2
    mask_union = np.zeros_like(input_mask1)
    mask_union = np.where(mask_fused > 0.0, 1.0, 0.0)
    res_IoU = np.sum(mask_inter) / np.sum(mask_union)

    return res_IoU


def LocalVerticalFlip(input_mask, fake_IR, real_vis, region_min_row, region_max_row):
    "input_mask:h*w. fake_IR:c*h*w. Numpy array."

    h = input_mask.shape[0]
    # w = input_mask.shape[1]
    input_mask_flip = np.flip(input_mask, 0)
    fake_IR_flip = np.flip(fake_IR, 1)
    real_vis_flip = np.flip(real_vis, 1)
    flip_region_min_row = int(h - region_max_row - 1)
    flip_region_max_row = int(h - region_min_row - 1)
    output_mask = np.zeros_like(input_mask)
    output_fake_IR = np.zeros_like(fake_IR)
    output_real_vis = np.zeros_like(real_vis)

    output_mask[region_min_row:region_max_row+1, :] = input_mask_flip[flip_region_min_row:flip_region_max_row+1, :]
    output_fake_IR[:, region_min_row:region_max_row+1, :] = fake_IR_flip[:, flip_region_min_row:flip_region_max_row+1, :]
    output_real_vis[:, region_min_row:region_max_row+1, :] = real_vis_flip[:, flip_region_min_row:flip_region_max_row+1, :]

    return output_mask, output_fake_IR, output_real_vis

def Red2Green(input_rgb, input_mask):
    "input_rgb: c*h*w. input_mask: h*w. Numpy array."
    input_rgb_norm = (input_rgb + 1.0) * 0.5
    input_rgb_masked = input_rgb_norm * input_mask
    # input_hsv = rgb_to_hsv(torch.tensor(input_rgb_masked).unsqueeze(0))
    input_hsv = rgb_to_hsv(torch.Tensor(input_rgb_masked).unsqueeze(0))
    input_hsv_numpy = torch.squeeze(input_hsv).numpy()
    # print(input_hsv_numpy.shape)
    input_h = input_hsv_numpy[0, :, :] * 180.0
    input_s = input_hsv_numpy[1, :, :] * 255.0
    input_v = input_hsv_numpy[2, :, :] * 255.0
    # s_mask = np.zeros_like(input_s)
    s_mask = np.where(input_s > 42, 1.0, 0.0)
    v_mask = np.where(input_v > 45, 1.0, 0.0)
    h_mask1 = np.where(input_h < 25, 1.0, 0.0)
    h_mask2 = np.where(input_h > 155, 1.0, 0.0)
    #########
    red_mask1_ori = s_mask * v_mask * h_mask1
    red_mask2_ori = s_mask * v_mask * h_mask2
    max_pool_k5 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
    # HL_Mask_dilate = max_pool_k5(HL_Mask)
    red_mask1_dilate = -max_pool_k5(-max_pool_k5(torch.Tensor(red_mask1_ori).unsqueeze(0).unsqueeze(0)))
    red_mask2_dilate = -max_pool_k5(-max_pool_k5(torch.Tensor(red_mask2_ori).unsqueeze(0).unsqueeze(0)))
    red_mask1 = torch.squeeze(red_mask1_dilate).numpy()
    red_mask2 = torch.squeeze(red_mask2_dilate).numpy()
    #############
    red_mask_area1 = np.sum(red_mask1)
    red_mask_area2 = np.sum(red_mask2)
    red_mask_area = red_mask_area1 + red_mask_area2
    red_mask_fused = red_mask1 + red_mask2
    if red_mask_area1 > 0:
        h_r2g_out1 = red_mask1 * ((input_h * 0.12) + 87.0)
    else:
        h_r2g_out1 = np.zeros_like(input_h)

    if red_mask_area2 > 0:
        h_r2g_out2 = red_mask2 * ((input_h * 0.12) + 64.0)
    else:
        h_r2g_out2 = np.zeros_like(input_h)

    h_out = (np.ones_like(input_h) - red_mask1 - red_mask2) * input_h + h_r2g_out1 + h_r2g_out2
    hsv_out = np.zeros_like(input_hsv_numpy)
    hsv_out[0, :, :] = h_out / 180.0
    s_out = (np.ones_like(input_h) - red_mask1 - red_mask2) * input_hsv_numpy[1, :, :] + (red_mask1 + \
                red_mask2) * (input_hsv_numpy[1, :, :] * 0.5)
    hsv_out[1, :, :] = s_out
    hsv_out[2, :, :] = input_hsv_numpy[2, :, :]
    # out_r2g = hsv_to_rgb(torch.tensor(hsv_out).unsqueeze(0))
    out_r2g = hsv_to_rgb(torch.Tensor(hsv_out).unsqueeze(0))
    res_numpy = torch.squeeze(out_r2g).numpy()

    return res_numpy, red_mask_area, red_mask_fused

def Green2Red(input_rgb, input_mask):
    "input_rgb: c*h*w. input_mask: h*w. Numpy array."
    input_rgb_norm = (input_rgb + 1.0) * 0.5
    input_rgb_masked = input_rgb_norm * input_mask
    # input_hsv = rgb_to_hsv(torch.tensor(input_rgb_masked).unsqueeze(0))
    input_hsv = rgb_to_hsv(torch.Tensor(input_rgb_masked).unsqueeze(0))
    input_hsv_numpy = torch.squeeze(input_hsv).numpy()
    input_h = input_hsv_numpy[0, :, :] * 180.0
    input_s = input_hsv_numpy[1, :, :] * 255.0
    input_v = input_hsv_numpy[2, :, :] * 255.0
    # s_mask = np.zeros_like(input_s)
    s_mask = np.where(input_s > 25, 1.0, 0.0)
    v_mask = np.where(input_v > 45, 1.0, 0.0)
    h_mask1 = np.where(input_h < 90, 1.0, 0.0)
    h_mask2 = np.where(input_h > 67, 1.0, 0.0)
    h_mask3 = np.where(input_h > 90, 1.0, 0.0)
    h_mask4 = np.where(input_h < 110, 1.0, 0.0)
    ######Padding
    green_mask1_ori = s_mask * v_mask * h_mask1 * h_mask2
    green_mask2_ori = s_mask * v_mask * h_mask3 * h_mask4
    max_pool_k5 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
    # HL_Mask_dilate = max_pool_k5(HL_Mask)
    green_mask1_dilate = -max_pool_k5(-max_pool_k5(torch.Tensor(green_mask1_ori).unsqueeze(0).unsqueeze(0)))
    green_mask2_dilate = -max_pool_k5(-max_pool_k5(torch.Tensor(green_mask2_ori).unsqueeze(0).unsqueeze(0)))
    green_mask1 = torch.squeeze(green_mask1_dilate).numpy()
    green_mask2 = torch.squeeze(green_mask2_dilate).numpy()
    ##############
    green_mask_area1 = np.sum(green_mask1)
    green_mask_area2 = np.sum(green_mask2)
    green_mask_area = green_mask_area1 + green_mask_area2
    green_mask_fused = green_mask1 + green_mask2
    if green_mask_area1 > 0:
        h_g2r_out1 = green_mask1 * ((input_h * 0.5) - 33.5)
    else:
        h_g2r_out1 = np.zeros_like(input_h)

    if green_mask_area2 > 0:
        h_g2r_out2 = green_mask2 * ((input_h * (-0.5)) + 55.0)
    else:
        h_g2r_out2 = np.zeros_like(input_h)

    h_out = (np.ones_like(input_h) - green_mask1 - green_mask2) * input_h + h_g2r_out1 + h_g2r_out2
    hsv_out = np.zeros_like(input_hsv_numpy)
    hsv_out[0, :, :] = h_out / 180.0
    s_out = (np.ones_like(input_h) - green_mask1 - green_mask2) * input_hsv_numpy[1, :, :] + (green_mask1 + \
                green_mask2) * (input_hsv_numpy[1, :, :] * 4.0)
    hsv_out[1, :, :] = s_out
    hsv_out[2, :, :] = input_hsv_numpy[2, :, :]
    # out_g2r = hsv_to_rgb(torch.tensor(hsv_out).unsqueeze(0))
    out_g2r = hsv_to_rgb(torch.Tensor(hsv_out).unsqueeze(0))
    res_numpy = torch.squeeze(out_g2r).numpy()

    return res_numpy, green_mask_area, green_mask_fused

def FakeVisFGMergeMask(IR_seg_tensor, vis_segmask, gpu_ids):
    "Selecting a suitable foreground mask from the fake visible image and fusing it with the real visible image, "
    "and keeping the original foreground area unchanged."

    sm = torch.nn.Softmax(dim = 1)
    pred_sm1 = sm(IR_seg_tensor.detach())
    pred_max_tensor1 = torch.max(pred_sm1, dim=1)
    pred_max_category1 = pred_max_tensor1[1]

    IR_segmask = pred_max_category1.float()
    # vis_FG_idx_list = [6, 7, 11, 12, 14, 15, 16, 17]
    vis_FG_idx_list = [6, 7, 17]
    # vis_FG_idx_list = [6, 7]
    large_FG_list = [15, 16]
    traffic_sign_list = [6, 7]
    vis_GT_segmask = torch.squeeze(vis_segmask).float().detach().cpu().numpy()
    real_IR_segmask = torch.squeeze(IR_segmask).float().detach().cpu().numpy()
    vis_road_mask = np.zeros_like(vis_GT_segmask)
    vis_road_mask = np.where(vis_GT_segmask < 2.0, 1.0, 0.0)
    vis_FG1_mask = np.zeros_like(vis_GT_segmask)
    vis_FG1_mask = np.where(vis_GT_segmask > 10.0, 1.0, 0.0)
    vis_light_mask = np.zeros_like(vis_GT_segmask)
    vis_light_mask = np.where(vis_GT_segmask == 6.0, 1.0, 0.0)
    vis_sign_mask = np.zeros_like(vis_GT_segmask)
    vis_sign_mask = np.where(vis_GT_segmask == 7.0, 1.0, 0.0)
    vis_FG_mask = vis_FG1_mask + vis_light_mask + vis_sign_mask
    output_FG_Mask = np.zeros_like(real_IR_segmask)
    for i in range(len(vis_FG_idx_list)):
        temp_mask = np.zeros_like(vis_GT_segmask)
        temp_mask = np.where(real_IR_segmask == vis_FG_idx_list[i], 1.0, 0.0)
        label_connect, num = measure.label(temp_mask, connectivity=2, background=0, return_num=True)
        for j in range(1, num+1):
            "Since background index is 0, the num is num+1."
            temp_connect_mask = np.zeros_like(label_connect)
            temp_connect_mask = np.where(label_connect == j, 1.0, 0.0)
            road_mask_prod = temp_connect_mask * vis_road_mask
            FG_mask_overlap = temp_connect_mask * vis_FG_mask
            # IoU_th = 0.5 * np.sum(temp_connect_mask)
            if np.sum(FG_mask_overlap) == 0:
                if np.sum(temp_connect_mask) > 50:
                    if vis_FG_idx_list[i] in traffic_sign_list:
                        output_FG_Mask += temp_connect_mask
                    elif vis_FG_idx_list[i] in large_FG_list:
                        IoU_th = 0.1 * np.sum(temp_connect_mask)
                        if np.sum(road_mask_prod) > IoU_th:
                            output_FG_Mask += temp_connect_mask
                    else:
                        IoU_th = 0.1 * np.sum(temp_connect_mask)
                        if np.sum(road_mask_prod) > IoU_th:
                            output_FG_Mask += temp_connect_mask

    out_FG_mask = torch.tensor(output_FG_Mask).cuda(gpu_ids).expand(1, 3, 256, 256)

    #########Flip fake mask
    IR_mask_flip = torch.flip(torch.squeeze(IR_segmask), dims=[1])
    IR_flip_segmask = IR_mask_flip.float().detach().cpu().numpy()
    output_FG_Mask_Flip = np.zeros_like(real_IR_segmask)
    vis_FG_mask_update = vis_FG_mask + output_FG_Mask
    vis_road_mask_update = vis_road_mask - vis_road_mask * output_FG_Mask
    for i in range(len(vis_FG_idx_list)):
        temp_mask = np.zeros_like(vis_GT_segmask)
        temp_mask = np.where(IR_flip_segmask == vis_FG_idx_list[i], 1.0, 0.0)
        label_connect, num = measure.label(temp_mask, connectivity=2, background=0, return_num=True)
        for j in range(1, num+1):
            "Since background index is 0, the num is num+1."
            temp_connect_mask = np.zeros_like(label_connect)
            temp_connect_mask = np.where(label_connect == j, 1.0, 0.0)
            road_mask_prod = temp_connect_mask * vis_road_mask_update
            FG_mask_overlap = temp_connect_mask * vis_FG_mask_update
            # IoU_th = 0.5 * np.sum(temp_connect_mask)
            if np.sum(FG_mask_overlap) == 0:
                if np.sum(temp_connect_mask) > 50:
                    if vis_FG_idx_list[i] in traffic_sign_list:
                        output_FG_Mask_Flip += temp_connect_mask
                    elif vis_FG_idx_list[i] in large_FG_list:
                        IoU_th = 0.1 * np.sum(temp_connect_mask)
                        if np.sum(road_mask_prod) > IoU_th:
                            output_FG_Mask_Flip += temp_connect_mask
                    else:
                        IoU_th = 0.1 * np.sum(temp_connect_mask)
                        if np.sum(road_mask_prod) > IoU_th:
                            output_FG_Mask_Flip += temp_connect_mask

    out_FG_mask_flip = torch.tensor(output_FG_Mask_Flip).cuda(gpu_ids).expand(1, 3, 256, 256)

    return out_FG_mask, out_FG_mask_flip, IR_segmask

def UpdateFakeIRSegGT(fake_IR, Seg_mask, dis_th):
    "The GT corresponding to the high-brightness region in the vegetation area in the fake IR image is set as "
    "an uncertain region, which is to reduce the perception of the street light as vegetation in the "
    "real IR image."

    _, _, h, w = fake_IR.size()
    _, seg_h, seg_w = Seg_mask.size()
    GT_mask_resize = F.interpolate(Seg_mask.expand(1, 1, seg_h, seg_w).float(), size=[h, w], mode='nearest')
    # real_mask_resize = F.interpolate(real_mask.expand(1, 1, seg_h, seg_w).float(), size=[h, w], mode='nearest')
    GT_mask = torch.squeeze(GT_mask_resize)
    veg_mask = torch.zeros_like(GT_mask)
    veg_mask = torch.where(GT_mask == 8.0, torch.ones_like(GT_mask), torch.zeros_like(GT_mask))

    fake_img_norm = (fake_IR + 1.0) * 0.5
    fake_IR_gray = torch.squeeze(.299 * fake_img_norm[:,0:1,:,:] + .587 * fake_img_norm[:,1:2,:,:] + .114 * fake_img_norm[:,2:3,:,:])
    
    if torch.sum(veg_mask) > 0:
        region_veg = veg_mask.mul(fake_IR_gray)
        region_veg_mean = torch.sum(region_veg) / torch.sum(veg_mask)
        region_veg_max = torch.max(region_veg)
        veg_range_high_ratio = (region_veg_max - region_veg_mean) / (region_veg_mean + 1e-4)

        "If the difference between the maximum brightness value and the average brightness value of a vegetation region is "
        "greater than a given threshold, the semantic labeling of the corresponding bright region (i.e., the region with "
        "greater than average brightness) is set to uncertain."
        if veg_range_high_ratio > dis_th:
            veg_high_mask = torch.zeros_like(GT_mask)
            veg_high_mask = torch.where(region_veg > region_veg_mean, torch.ones_like(GT_mask), torch.zeros_like(GT_mask))
            mask_new_GT = veg_high_mask * 255.0 + (torch.ones_like(veg_high_mask) - veg_high_mask).mul(GT_mask)
            out_mask = mask_new_GT.expand(1, h, w)
        else:
            out_mask = Seg_mask
    else:
        out_mask = Seg_mask

    return out_mask

def IRComPreProcessv6(FG_mask, FG_mask_flip, Fake_IR_masked_ori, Fake_IR_masked_flip_ori, Real_IR, Real_IR_SegMask, HL_Mask_ori):
    "Gaussian blurring is applied to the fake NTIR images to enhance the plausibility of the appearance of FG region."

    #### HL_Mask padding
    max_pool_k5 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
    # HL_Mask_dilate = max_pool_k5(HL_Mask)
    HL_Mask = -max_pool_k5(-max_pool_k5(HL_Mask_ori))

    if torch.sum(HL_Mask) > 0:
        FG_mask_sub_HL = FG_mask - HL_Mask
        FG_LL_mask = torch.zeros_like(FG_mask)
        FG_LL_mask = torch.where(FG_mask_sub_HL > 0.0, torch.ones_like(FG_mask), torch.zeros_like(FG_mask))

        FG_mask_sub_HL_flip = FG_mask_flip - HL_Mask
        FG_LL_mask_flip = torch.zeros_like(FG_mask)
        FG_LL_mask_flip = torch.where(FG_mask_sub_HL_flip > 0.0, torch.ones_like(FG_mask), torch.zeros_like(FG_mask))
    else:
        FG_LL_mask = FG_mask
        FG_LL_mask_flip = FG_mask_flip

    ######Calculate the mean value of the road area
    IR_SegMask = torch.squeeze(Real_IR_SegMask.float())
    RealIR_RoadMask = torch.zeros_like(IR_SegMask)
    RealIR_RoadMask = torch.where(IR_SegMask == 0.0, torch.ones_like(IR_SegMask), torch.zeros_like(IR_SegMask))
    Real_IR_norm = (Real_IR + 1.0) * 0.5
    Fake_IR_norm = (Fake_IR_masked_ori + 1.0) * 0.5
    Fake_IR_flip_norm = (Fake_IR_masked_flip_ori + 1.0) * 0.5
    Real_IR_gray = torch.squeeze(.299 * Real_IR_norm[:,0:1,:,:] + .587 * Real_IR_norm[:,1:2,:,:] + .114 * Real_IR_norm[:,2:3,:,:])
    Fake_IR_gray = torch.squeeze(.299 * Fake_IR_norm[:,0:1,:,:] + .587 * Fake_IR_norm[:,1:2,:,:] + .114 * Fake_IR_norm[:,2:3,:,:])
    Fake_IR_flip_gray = torch.squeeze(.299 * Fake_IR_flip_norm[:,0:1,:,:] + .587 * Fake_IR_flip_norm[:,1:2,:,:] + .114 * Fake_IR_flip_norm[:,2:3,:,:])
    FG_area = torch.sum(FG_LL_mask + FG_LL_mask_flip)
    Fake_IR_FG_mean = (torch.sum(FG_LL_mask.mul(Fake_IR_gray)) + torch.sum(FG_LL_mask_flip.mul(Fake_IR_flip_gray))) / (FG_area + 1)
    Fake_IR_Fused = FG_LL_mask.mul(Fake_IR_gray) + FG_LL_mask_flip.mul(Fake_IR_flip_gray)
    Fake_IR_Fused_MaxValue = torch.max(Fake_IR_Fused)
    if torch.sum(RealIR_RoadMask) > 0:
        ######Adaptive luminance adjustment strategy: Adaptive scaling of luminance adjustment based on the mean value of the road area
        real_IR_Road_Mean = torch.sum(RealIR_RoadMask.mul(Real_IR_gray)) / torch.sum(RealIR_RoadMask)
        RB_Scale_Mean = real_IR_Road_Mean.detach() / (Fake_IR_FG_mean.detach() + 1e-6)
        real_IR_Road_MaxValue = torch.max(RealIR_RoadMask.mul(Real_IR_gray))
        #######Prevent the maximum value from crossing the boundary.
        RB_Scale_Max = real_IR_Road_MaxValue.detach() / (Fake_IR_Fused_MaxValue.detach() + 1e-6)
        RB_Scale = torch.min(RB_Scale_Mean, RB_Scale_Max)
        
    else:
        RB_Scale_Mean = torch.mean(Real_IR_gray) / (Fake_IR_FG_mean.detach() + 1e-6)
        real_IR_MaxValue = torch.max(Real_IR_gray)
        RB_Scale_Max = real_IR_MaxValue.detach() / (Fake_IR_Fused_MaxValue.detach() + 1e-6)
        RB_Scale = torch.min(RB_Scale_Mean, RB_Scale_Max)
    
    Fake_IR_masked_RB_norm = RB_Scale * (FG_LL_mask.mul(Fake_IR_gray)) + (FG_mask - FG_LL_mask).mul(Fake_IR_gray)
    Fake_IR_masked_flip_RB_norm = RB_Scale * (FG_LL_mask_flip.mul(Fake_IR_flip_gray)) + \
                                (FG_mask_flip - FG_LL_mask_flip).mul(Fake_IR_flip_gray)

    Fake_IR_masked_RB = ((Fake_IR_masked_RB_norm - 0.5) * 2.0).mul(FG_mask)
    Fake_IR_masked_flip_RB = ((Fake_IR_masked_flip_RB_norm - 0.5) * 2.0).mul(FG_mask_flip)

    IR_com = (torch.ones_like(FG_mask) - FG_mask - FG_mask_flip).mul(Real_IR) + \
                Fake_IR_masked_RB.expand_as(Real_IR) + Fake_IR_masked_flip_RB.expand_as(Real_IR)

    return IR_com

def TrafLighCorlLoss(real_IR, fake_vis, IR_mask, real_vis, vis_Light_mask, HL_Mask_ori, gpu_ids):
    "Traffic light color loss: The color distribution of the traffic lights in the fake visible image is encouraged to be consistent "
    "with the real image."

    max_pool_k5 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
    vis_HL_Mask = -max_pool_k5(-max_pool_k5(HL_Mask_ori))
    if torch.sum(vis_HL_Mask) > 0:
        IR_seg_mask_2D = torch.squeeze(IR_mask)
        # h, w = IR_seg_mask_2D.size()
        IR_Light_mask = torch.zeros_like(IR_seg_mask_2D)
        IR_Light_mask = torch.where(IR_seg_mask_2D == 6.0, torch.ones_like(IR_seg_mask_2D), torch.zeros_like(IR_seg_mask_2D))
        if torch.sum(IR_Light_mask) > 100:
            IR_Light_top_mask = get_ROI_top_part_mask(IR_Light_mask, gpu_ids)
            vis_Light_top_mask = vis_Light_mask[0, :, :]
            IR_Light_bottom_mask = IR_Light_mask - IR_Light_top_mask
            vis_Light_bottom_mask = vis_Light_mask[1, :, :]
            Real_IR_norm = (real_IR + 1.0) * 0.5
            Fake_vis_norm = torch.squeeze((fake_vis + 1.0) * 0.5)
            Real_vis_norm = torch.squeeze((real_vis + 1.0) * 0.5)
            Real_IR_gray = torch.squeeze(.299 * Real_IR_norm[:,0:1,:,:] + .587 * Real_IR_norm[:,1:2,:,:] + .114 * Real_IR_norm[:,2:3,:,:])
            IR_gray_light = Real_IR_gray * IR_Light_mask
            IR_gray_light_mean = torch.sum(IR_gray_light) / torch.sum(IR_Light_mask)
            IR_HL_mask = torch.zeros_like(IR_seg_mask_2D)
            IR_HL_mask = torch.where(IR_gray_light > IR_gray_light_mean, torch.ones_like(IR_seg_mask_2D), torch.zeros_like(IR_seg_mask_2D))
            IR_top_HL_mask = IR_HL_mask * IR_Light_top_mask
            IR_bottom_HL_mask = IR_HL_mask * IR_Light_bottom_mask
            vis_top_HL_mask = vis_HL_Mask[0, 0, :, :] * vis_Light_top_mask
            vis_bottom_HL_mask = vis_HL_Mask[0, 0, :, :] * vis_Light_bottom_mask
            HL_top_idx = torch.sum(IR_top_HL_mask) * torch.sum(vis_top_HL_mask)
            HL_bottom_idx = torch.sum(IR_bottom_HL_mask) * torch.sum(vis_bottom_HL_mask)
            if HL_top_idx > 0:
                fake_vis_top_masked = Fake_vis_norm.mul(IR_top_HL_mask.expand_as(Fake_vis_norm))
                fake_vis_top_HL_Light_mean = torch.sum(fake_vis_top_masked, dim=(1, 2), keepdim=True) / torch.sum(IR_top_HL_mask)
                real_vis_top_masked = Real_vis_norm.mul(vis_top_HL_mask.expand_as(Real_vis_norm))
                real_vis_top_HL_Light_mean = torch.sum(real_vis_top_masked, dim=(1, 2), keepdim=True) / torch.sum(vis_top_HL_mask)
                HL_top_loss = torch.sqrt(torch.sum((real_vis_top_HL_Light_mean.detach() - fake_vis_top_HL_Light_mean) ** 2))
            else:
                HL_top_loss = torch.zeros(1).cuda(gpu_ids)

            if HL_bottom_idx > 0:
                fake_vis_bottom_masked = Fake_vis_norm.mul(IR_bottom_HL_mask.expand_as(Fake_vis_norm))
                fake_vis_bottom_HL_Light_mean = torch.sum(fake_vis_bottom_masked, dim=(1, 2), keepdim=True) / torch.sum(IR_bottom_HL_mask)
                real_vis_bottom_masked = Real_vis_norm.mul(vis_bottom_HL_mask.expand_as(Real_vis_norm))
                real_vis_bottom_HL_Light_mean = torch.sum(real_vis_bottom_masked, dim=(1, 2), keepdim=True) / torch.sum(vis_bottom_HL_mask)
                # HL_bottom_loss = torch.sqrt(torch.sum((real_vis_bottom_HL_Light_mean.detach() - fake_vis_bottom_HL_Light_mean) ** 2))
                bottom_loss_sim = torch.sqrt(torch.sum((real_vis_bottom_HL_Light_mean.detach() - fake_vis_bottom_HL_Light_mean) ** 2))
                if HL_top_idx > 0:
                    bottom_loss_var = torch.sqrt(torch.sum((real_vis_top_HL_Light_mean.detach() - fake_vis_bottom_HL_Light_mean) ** 2))
                    Norm_factor = torch.min(bottom_loss_sim.detach(), bottom_loss_var)
                    HL_bottom_loss = bottom_loss_sim / (Norm_factor + 0.05)
                else:
                    HL_bottom_loss = bottom_loss_sim

            else:
                HL_bottom_loss = torch.zeros(1).cuda(gpu_ids)
            
            out_losses = HL_top_loss + HL_bottom_loss
        else:
            out_losses = torch.zeros(1).cuda(gpu_ids)
    else:
        out_losses = torch.zeros(1).cuda(gpu_ids)

    return out_losses

def get_ROI_top_part_mask(input_mask, gpu_ids):
    h, w = input_mask.size()
    row_pos_array = torch.mm(torch.as_tensor(torch.arange(h).reshape((h,1)), dtype=torch.float), torch.ones((1,h)))
    row_pos_array_masked = input_mask * (row_pos_array.cuda(gpu_ids))
    center_row_tensor = torch.sum(row_pos_array_masked, dim=0) / (torch.sum(input_mask, dim=0) + 1e-6)
    out_mask = torch.zeros_like(input_mask)
    
    for i in range(w):
        temp_row = int(center_row_tensor[i])
        if temp_row > 0:
            out_mask[:temp_row, i] = torch.ones(1).cuda(gpu_ids)
    
    mask_top_part = out_mask * input_mask

    return mask_top_part


######The conversion code between rgb and hsv images is derived from https://blog.csdn.net/Brikie/article/details/115086835.
def rgb_to_hsv(input_rgb):
    "input_rgb : 4D tensor."
    hue = torch.Tensor(input_rgb.shape[0], input_rgb.shape[2], input_rgb.shape[3]).to(input_rgb.device)

    hue[ input_rgb[:,2]==input_rgb.max(1)[0] ] = 4.0 + ( (input_rgb[:,0]-input_rgb[:,1]) / ( input_rgb.max(1)[0] - input_rgb.min(1)[0] + 1e-8) ) [ input_rgb[:,2]==input_rgb.max(1)[0] ]
    hue[ input_rgb[:,1]==input_rgb.max(1)[0] ] = 2.0 + ( (input_rgb[:,2]-input_rgb[:,0]) / ( input_rgb.max(1)[0] - input_rgb.min(1)[0] + 1e-8) ) [ input_rgb[:,1]==input_rgb.max(1)[0] ]
    hue[ input_rgb[:,0]==input_rgb.max(1)[0] ] = (0.0 + ( (input_rgb[:,1]-input_rgb[:,2]) / ( input_rgb.max(1)[0] - input_rgb.min(1)[0] + 1e-8) ) [ input_rgb[:,0]==input_rgb.max(1)[0] ]) % 6

    hue[input_rgb.min(1)[0]==input_rgb.max(1)[0]] = 0.0
    hue = hue/6

    saturation = ( input_rgb.max(1)[0] - input_rgb.min(1)[0] ) / ( input_rgb.max(1)[0] + 1e-8 )
    saturation[ input_rgb.max(1)[0]==0 ] = 0

    value = input_rgb.max(1)[0]
    
    hue = hue.unsqueeze(1)
    saturation = saturation.unsqueeze(1)
    value = value.unsqueeze(1)
    hsv = torch.cat([hue, saturation, value],dim=1)
    return hsv

def hsv_to_rgb(input_hsv):
    "input_hsv : 4D tensor."
    h, s, v = input_hsv[:,0,:,:], input_hsv[:,1,:,:], input_hsv[:,2,:,:]
    # ###Treatment of out-of-bounds values
    h = h%1
    s = torch.clamp(s,0,1)
    v = torch.clamp(v,0,1)

    r = torch.zeros_like(h)
    g = torch.zeros_like(h)
    b = torch.zeros_like(h)
    
    hi = torch.floor(h * 6)
    f = h * 6 - hi
    p = v * (1 - s)
    q = v * (1 - (f * s))
    t = v * (1 - ((1 - f) * s))
    
    hi0 = hi==0
    hi1 = hi==1
    hi2 = hi==2
    hi3 = hi==3
    hi4 = hi==4
    hi5 = hi==5
    
    r[hi0] = v[hi0]
    g[hi0] = t[hi0]
    b[hi0] = p[hi0]
    
    r[hi1] = q[hi1]
    g[hi1] = v[hi1]
    b[hi1] = p[hi1]
    
    r[hi2] = p[hi2]
    g[hi2] = v[hi2]
    b[hi2] = t[hi2]
    
    r[hi3] = p[hi3]
    g[hi3] = q[hi3]
    b[hi3] = v[hi3]
    
    r[hi4] = t[hi4]
    g[hi4] = p[hi4]
    b[hi4] = v[hi4]
    
    r[hi5] = v[hi5]
    g[hi5] = p[hi5]
    b[hi5] = q[hi5]
    
    r = r.unsqueeze(1)
    g = g.unsqueeze(1)
    b = b.unsqueeze(1)
    rgb = torch.cat([r, g, b], dim=1)
    return rgb

class SSIM_Loss(SSIM):
    def forward(self, img1, img2):
        return ( 1 - super(SSIM_Loss, self).forward(img1, img2) )

# Defines the total variation (TV) loss, which encourages spatial smoothness in the generated image.
class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]

# Defines the generator that consists of Resnet blocks between a few
# downsampling/upsampling operations.
# Code and idea originally from Justin Johnson's architecture.
# https://github.com/jcjohnson/fast-neural-style/
class ResnetGenEncoder(nn.Module):
    def __init__(self, input_nc, n_blocks=4, ngf=64, norm_layer=nn.BatchNorm2d,
                 use_dropout=False, gpu_ids=[], use_bias=False, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGenEncoder, self).__init__()
        self.gpu_ids = gpu_ids

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.PReLU()]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.PReLU()]

        mult = 2**n_downsampling
        for _ in range(n_blocks):
            model += [ResnetBlock(ngf * mult, norm_layer=norm_layer,
                                  use_dropout=use_dropout, use_bias=use_bias, padding_type=padding_type)]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        return self.model(input)

###Add 1 Pyramid Guided Attention Block v4 before ResBlock groups
class ResnetGenEncoderv2(nn.Module):
    def __init__(self, input_nc, n_blocks=4, ngf=64, norm_layer=nn.BatchNorm2d,
                 use_dropout=False, gpu_ids=[], use_bias=False, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGenEncoderv2, self).__init__()
        self.gpu_ids = gpu_ids

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.PReLU()]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.PReLU()]

        mult = 2**n_downsampling
        for _ in range(n_blocks):
            model += [ResnetBlock(ngf * mult, norm_layer=norm_layer,
                                  use_dropout=use_dropout, use_bias=use_bias, padding_type=padding_type)]

        self.model = nn.Sequential(*model)
        self.module_SGPA = SGPABlock(ngf * mult, norm_layer=norm_layer, use_bias=use_bias, gpu_ids=gpu_ids, padding_type='reflect')

    def forward(self, input):

        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            temp_fea = nn.parallel.data_parallel(self.model, input, self.gpu_ids)
            _, _, h, w = temp_fea.size()
            input_resize = F.interpolate(input, size=(h, w), mode='bilinear', align_corners=False)
            out, attmap1, attmap2, attmap3 = nn.parallel.data_parallel(self.module_SGPA, torch.cat((temp_fea, input_resize), 1), self.gpu_ids)

        else:
            temp_fea = self.model(input)
            _, _, h, w = temp_fea.size()
            input_resize = F.interpolate(input, size=(h, w), mode='bilinear', align_corners=False)
            out, attmap1, attmap2, attmap3 = self.module_SGPA(torch.cat((temp_fea, input_resize), 1))

        return out, attmap1, attmap2, attmap3

####Add 1 Pyramid Guided Attention Block v4 before ResBlock groups
class ResnetGenDecoderv1(nn.Module):
    def __init__(self, output_nc, n_blocks=5, ngf=64, norm_layer=nn.BatchNorm2d,
                 use_dropout=False, gpu_ids=[], use_bias=False, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGenDecoderv1, self).__init__()
        self.gpu_ids = gpu_ids
        

        model = []
        n_downsampling = 2
        mult = 2**n_downsampling

        for _ in range(n_blocks):
            model += [ResnetBlock(ngf * mult, norm_layer=norm_layer,
                                  use_dropout=use_dropout, use_bias=use_bias, padding_type=padding_type)]
            
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=4, stride=2,
                                         padding=1, output_padding=0,
                                         bias=use_bias),
                      nn.GroupNorm(32, int(ngf * mult / 2)),
                      nn.PReLU()]

        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        return self.model(input)

class ResnetGenShared(nn.Module):
    def __init__(self, n_domains, n_blocks=2, ngf=64, norm_layer=nn.BatchNorm2d,
                 use_dropout=False, gpu_ids=[], use_bias=False, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGenShared, self).__init__()
        self.gpu_ids = gpu_ids

        model = []
        n_downsampling = 2
        mult = 2**n_downsampling

        for _ in range(n_blocks):
            model += [ResnetBlock(ngf * mult, norm_layer=norm_layer, n_domains=n_domains,
                                  use_dropout=use_dropout, use_bias=use_bias, padding_type=padding_type)]

        self.model = SequentialContext(n_domains, *model)

    def forward(self, input, domain):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, (input, domain), self.gpu_ids)
        return self.model(input, domain)

class ResnetGenDecoder(nn.Module):
    def __init__(self, output_nc, n_blocks=5, ngf=64, norm_layer=nn.BatchNorm2d,
                 use_dropout=False, gpu_ids=[], use_bias=False, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGenDecoder, self).__init__()
        self.gpu_ids = gpu_ids

        model = []
        n_downsampling = 2
        mult = 2**n_downsampling

        for _ in range(n_blocks):
            model += [ResnetBlock(ngf * mult, norm_layer=norm_layer,
                                  use_dropout=use_dropout, use_bias=use_bias, padding_type=padding_type)]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=4, stride=2,
                                         padding=1, output_padding=0,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.PReLU()]

        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        return self.model(input)

# Define a SGPABlock, attention maps of three scale are fused adaptively. Spatial Gradient Pyramid Attention Module
class SGPABlock(nn.Module):
    def __init__(self, in_dim, norm_layer, use_bias, gpu_ids=[], padding_type='reflect'):
        super(SGPABlock, self).__init__()

        self.gpu_ids = gpu_ids
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        # self.dsamp_filter = self.Tensor([1]).view(1,1,1,1)
        self.grad_filter = self.Tensor([0,0,0,-1,0,1,0,0,0]).view(1,1,3,3)

        self.GradConv = nn.Sequential(nn.Conv2d(2, 32, kernel_size=7, padding=3, bias=use_bias, padding_mode='zeros'), norm_layer(32), nn.PReLU())
        self.GradAtt = nn.Sequential(nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=use_bias, padding_mode='zeros'), norm_layer(32), nn.PReLU(), nn.Conv2d(32, 1, kernel_size=3, padding=1, bias=use_bias, padding_mode='zeros'), nn.Sigmoid())
        
        self.ConvLK1 = nn.Sequential(nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1, bias=use_bias, padding_mode=padding_type), norm_layer(in_dim), nn.PReLU())
        self.ConvCF1 = nn.Sequential(nn.Conv2d(in_dim, 32, kernel_size=1, padding=0, bias=use_bias), norm_layer(32), nn.PReLU())
        self.ConvLK2 = nn.Sequential(nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1, bias=use_bias, padding_mode=padding_type), norm_layer(in_dim), nn.PReLU())
        self.ConvCF2 = nn.Sequential(nn.Conv2d(in_dim, 32, kernel_size=1, padding=0, bias=use_bias), norm_layer(32), nn.PReLU())
        self.ConvLK3 = nn.Sequential(nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1, bias=use_bias, padding_mode=padding_type), norm_layer(in_dim), nn.PReLU())
        self.ConvCF3 = nn.Sequential(nn.Conv2d(in_dim, 32, kernel_size=1, padding=0, bias=use_bias), norm_layer(32), nn.PReLU())

        self.ds1 = nn.Sequential(nn.AvgPool2d(kernel_size=2, stride=2), nn.AvgPool2d(kernel_size=2, stride=2))
        self.ConvCF_Up1 = nn.Sequential(nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1, bias=use_bias, padding_mode=padding_type), norm_layer(in_dim), nn.PReLU())
        self.ConvCF_Up2 = nn.Sequential(nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1, bias=use_bias, padding_mode=padding_type), norm_layer(in_dim), nn.PReLU())
        
    def getgradmaps(self, input_gray_img):

        dx = F.conv2d(input_gray_img, self.grad_filter, padding=1)
        dy = F.conv2d(input_gray_img, self.grad_filter.transpose(-2,-1), padding=1)
        gradient = torch.cat([dx, dy], 1)
        # x_gradmagn = torch.sqrt(torch.pow(dx, 2) + torch.pow(dy, 2) + 1e-6)
        
        return gradient

    def forward(self, x):
        
        _, _, h, w = x.size()
        # print(x.size())
        input_fea = x[:, :-3, :, :]
        ori_img = x[:, -3:, :, :]
        gray = (.299 * ori_img[:,0,:,:] + .587 * ori_img[:,1,:,:] + .114 * ori_img[:,2,:,:]).unsqueeze_(1)

        gray_dsamp1 = F.interpolate(gray, size=(h // 4, w // 4), mode='bilinear', align_corners=False)
        gray_dsamp2 = F.interpolate(gray, size=(h // 2, w // 2), mode='bilinear', align_corners=False)
        gray_dsamp3 = F.interpolate(gray, size=(h, w), mode='bilinear', align_corners=False)

        gradfea1 = self.GradConv(self.getgradmaps(gray_dsamp1))
        gradfea2 = self.GradConv(self.getgradmaps(gray_dsamp2))
        gradfea3 = self.GradConv(self.getgradmaps(gray_dsamp3))

        fea_ds1 = self.ds1(input_fea)
        fea_LKC1 = self.ConvLK1(fea_ds1)
        fea_CF1 = self.ConvCF1(fea_LKC1)
        gradattmap1 = self.GradAtt(torch.cat([fea_CF1, gradfea1], 1))
        fea_att1 = gradattmap1.expand_as(fea_LKC1).mul(fea_LKC1) + fea_ds1

        fea_ds2 = self.ConvCF_Up1(F.interpolate(fea_att1, size=(h // 2, w // 2), mode='bilinear', align_corners=False))
        fea_LKC2 = self.ConvLK2(fea_ds2)
        fea_CF2 = self.ConvCF2(fea_LKC2)
        gradattmap2 = self.GradAtt(torch.cat([fea_CF2, gradfea2], 1))
        fea_att2 = gradattmap2.expand_as(fea_LKC2).mul(fea_LKC2) + fea_ds2

        fea_ds3 = self.ConvCF_Up2(F.interpolate(fea_att2, size=(h, w), mode='bilinear', align_corners=False))
        fea_LKC3 = self.ConvLK3(fea_ds3)
        fea_CF3 = self.ConvCF3(fea_LKC3)
        gradattmap3 = self.GradAtt(torch.cat([fea_CF3, gradfea3], 1))
        out = gradattmap3.expand_as(fea_LKC3).mul(fea_LKC3) + input_fea

        AM1_us = F.interpolate(gradattmap1, size=(h, w), mode='bilinear', align_corners=False)
        AM2_us = F.interpolate(gradattmap2, size=(h, w), mode='bilinear', align_corners=False)

        return out, AM1_us, AM2_us, gradattmap3

class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, norm_layer):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = norm_layer(planes)
        self.relu = nn.PReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.zero_()

class ASPP(nn.Module):
    def __init__(self, in_dim, num_classes, norm_layer):
        super(ASPP, self).__init__()
        
        dilations = [1, 6, 12, 18]

        self.aspp1 = _ASPPModule(in_dim, 64, 1, padding=0, dilation=dilations[0], norm_layer=norm_layer)
        self.aspp2 = _ASPPModule(in_dim, 64, 3, padding=dilations[1], dilation=dilations[1], norm_layer=norm_layer)
        self.aspp3 = _ASPPModule(in_dim, 64, 3, padding=dilations[2], dilation=dilations[2], norm_layer=norm_layer)
        self.aspp4 = _ASPPModule(in_dim, 64, 3, padding=dilations[3], dilation=dilations[3], norm_layer=norm_layer)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(in_dim, 64, 1, stride=1, bias=False),
                                             nn.PReLU())
        self.conv1 = nn.Conv2d(320, 256, 1, bias=False)
        self.bn1 = norm_layer(256)
        self.relu = nn.PReLU()
        self.dropout = nn.Dropout(0.5)
        self.conv_1x1_4 = nn.Conv2d(256, num_classes, kernel_size=1)
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        out = self.conv_1x1_4(self.dropout(x))

        return out, x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.zero_()


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, norm_layer, use_dropout, use_bias, padding_type='reflect', n_domains=0):
        super(ResnetBlock, self).__init__()

        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim + n_domains, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.PReLU()]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim + n_domains, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        self.conv_block = SequentialContext(n_domains, *conv_block)

    def forward(self, input):
        if isinstance(input, tuple):
            return input[0] + self.conv_block(*input)
        return input + self.conv_block(input)

# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, tensor=torch.FloatTensor, norm_layer=nn.BatchNorm2d, gpu_ids=[]):
        super(NLayerDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids
        self.grad_filter = tensor([0,0,0,-1,0,1,0,0,0]).view(1,1,3,3)
        self.dsamp_filter = tensor([1]).view(1,1,1,1)
        self.blur_filter = tensor(gkern_2d())

        self.model_rgb = self.model(input_nc, ndf, n_layers, norm_layer)
        self.model_gray = self.model(1, ndf, n_layers, norm_layer)
        self.model_grad = self.model(2, ndf, n_layers-1, norm_layer)

    def model(self, input_nc, ndf, n_layers, norm_layer):
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = int(np.ceil((kw-1)/2))
        sequences = [[
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.PReLU()
        ]]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequences += [[
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult + 1,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult + 1),
                nn.PReLU()
            ]]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequences += [[
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.PReLU(),
            \
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        ]]

        return SequentialOutput(*sequences)

    def forward(self, input):
        blurred = torch.nn.functional.conv2d(input, self.blur_filter, groups=3, padding=2)
        gray = (.299*input[:,0,:,:] + .587*input[:,1,:,:] + .114*input[:,2,:,:]).unsqueeze_(1)

        gray_dsamp = nn.functional.conv2d(gray, self.dsamp_filter, stride=2)
        dx = nn.functional.conv2d(gray_dsamp, self.grad_filter)
        dy = nn.functional.conv2d(gray_dsamp, self.grad_filter.transpose(-2,-1))
        gradient = torch.cat([dx,dy], 1)

        if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor):
            outs1 = nn.parallel.data_parallel(self.model_rgb, blurred, self.gpu_ids)
            outs2 = nn.parallel.data_parallel(self.model_gray, gray, self.gpu_ids)
            outs3 = nn.parallel.data_parallel(self.model_grad, gradient, self.gpu_ids)
        else:
            outs1 = self.model_rgb(blurred)
            outs2 = self.model_gray(gray)
            outs3 = self.model_grad(gradient)
        return outs1, outs2, outs3

class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        # print(x.size())
        if x.size(0) == 1:
            # These two lines run much faster in pytorch 0.4 than the two lines listed below.
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)

        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x


class NLayerDiscriminatorSN(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, tensor=torch.FloatTensor, norm_layer=nn.BatchNorm2d, gpu_ids=[]):
        super(NLayerDiscriminatorSN, self).__init__()
        self.gpu_ids = gpu_ids
        self.grad_filter = tensor([0,0,0,-1,0,1,0,0,0]).view(1,1,3,3)
        self.dsamp_filter = tensor([1]).view(1,1,1,1)
        self.blur_filter = tensor(gkern_2d())

        self.model_rgb = self.model(input_nc, ndf, n_layers, norm_layer)
        self.model_gray = self.model(1, ndf, n_layers, norm_layer)
        self.model_grad = self.model(2, ndf, n_layers-1, norm_layer)

    def model(self, input_nc, ndf, n_layers, norm_layer):
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = int(np.ceil((kw-1)/2))
        sequences = [[
            SNConv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.PReLU()
        ]]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequences += [[
                SNConv2d(ndf * nf_mult_prev, ndf * nf_mult + 1,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                nn.PReLU()
            ]]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequences += [[
            SNConv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            nn.PReLU(),
            \
            SNConv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        ]]

        return SequentialOutput(*sequences)

    def forward(self, input):
        blurred = torch.nn.functional.conv2d(input, self.blur_filter, groups=3, padding=2)
        gray = (.299*input[:,0,:,:] + .587*input[:,1,:,:] + .114*input[:,2,:,:]).unsqueeze_(1)

        gray_dsamp = nn.functional.conv2d(gray, self.dsamp_filter, stride=2)
        dx = nn.functional.conv2d(gray_dsamp, self.grad_filter)
        dy = nn.functional.conv2d(gray_dsamp, self.grad_filter.transpose(-2,-1))
        gradient = torch.cat([dx,dy], 1)

        if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor):
            outs1 = nn.parallel.data_parallel(self.model_rgb, blurred, self.gpu_ids)
            outs2 = nn.parallel.data_parallel(self.model_gray, gray, self.gpu_ids)
            outs3 = nn.parallel.data_parallel(self.model_grad, gradient, self.gpu_ids)
        else:
            outs1 = self.model_rgb(blurred)
            outs2 = self.model_gray(gray)
            outs3 = self.model_grad(gradient)
        return outs1, outs2, outs3

# Defines the SegmentorHeadv2. Zero Padding and CSG for positional encoding.
class SegmentorHeadv2(nn.Module):
    def __init__(self, input_nc, n_blocks=4, ngf=64, num_classes=19, norm_layer=nn.BatchNorm2d,
                 use_dropout=False, gpu_ids=[], use_bias=False, padding_type='zero'):
        super(SegmentorHeadv2, self).__init__()
        assert(n_blocks >= 0)
        self.gpu_ids = gpu_ids

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(5, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.PReLU()]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.PReLU()]

        mult = 2**n_downsampling
        for _ in range(n_blocks):
            model += [ResnetBlock(ngf * mult, norm_layer=norm_layer,
                                  use_dropout=use_dropout, use_bias=use_bias, padding_type=padding_type)]

        mult = 2**(n_downsampling)

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=4, stride=2,
                                         padding=1, output_padding=0,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.PReLU()]

        model += [ASPP(int(ngf), num_classes, norm_layer)]

        self.model = nn.Sequential(*model)
        self.csg = CatersianGrid()

    def forward(self, input):

        if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor):
            outs, seg_fea = nn.parallel.data_parallel(self.model, torch.cat((input, self.csg(input)), 1), self.gpu_ids)
        else:
            outs, seg_fea = self.model(torch.cat((input, self.csg(input)), 1))
        return outs, seg_fea

class Plexer(nn.Module):
    def __init__(self):
        super(Plexer, self).__init__()

    def apply(self, func):
        for net in self.networks:
            net.apply(func)

    def cuda(self, device_id):
        for net in self.networks:
            net.cuda(device_id)

    def init_optimizers(self, opt, lr, betas):
        self.optimizers = [opt(net.parameters(), lr=lr, betas=betas) \
                           for net in self.networks]

    def zero_grads(self, dom_a, dom_b):
        self.optimizers[dom_a].zero_grad()
        self.optimizers[dom_b].zero_grad()

    def step_grads(self, dom_a, dom_b):
        self.optimizers[dom_a].step()
        self.optimizers[dom_b].step()

    def update_lr(self, new_lr):
        for opt in self.optimizers:
            for param_group in opt.param_groups:
                param_group['lr'] = new_lr

    def update_lr_2domain(self, new_lr, dom_a, dom_b):
        "Add by lfy."
        # print(len(self.optimizers))
        # print(self.optimizers[dom_a])
        for param_group in self.optimizers[dom_a].param_groups:
            param_group['lr'] = new_lr

        for param_group in self.optimizers[dom_b].param_groups:
            param_group['lr'] = new_lr

    def save(self, save_path):
        for i, net in enumerate(self.networks):
            filename = save_path + ('%d.pth' % i)
            torch.save(net.cpu().state_dict(), filename)

    def load(self, save_path):
        for i, net in enumerate(self.networks):
            filename = save_path + ('%d.pth' % i)
            net.load_state_dict(torch.load(filename))

class G_Plexer(Plexer):
    def __init__(self, n_domains, encoder, enc_args, decoder, dec_args,
                 block=None, shenc_args=None, shdec_args=None):
        super(G_Plexer, self).__init__()
        self.encoders = [encoder(*enc_args) for _ in range(n_domains)]
        self.decoders = [decoder(*dec_args) for _ in range(n_domains)]

        self.sharing = block is not None
        if self.sharing:
            self.shared_encoder = block(*shenc_args)
            self.shared_decoder = block(*shdec_args)
            self.encoders.append( self.shared_encoder )
            self.decoders.append( self.shared_decoder )
        self.networks = self.encoders + self.decoders

    def init_optimizers(self, opt, lr, betas):
        self.optimizers = []
        for enc, dec in zip(self.encoders, self.decoders):
            params = itertools.chain(enc.parameters(), dec.parameters())
            self.optimizers.append( opt(params, lr=lr, betas=betas) )

    def forward(self, input, in_domain, out_domain):
        encoded = self.encode(input, in_domain)
        return self.decode(encoded, out_domain)

    def encode(self, input, domain):
        output = self.encoders[domain].forward(input)
        if self.sharing:
            return self.shared_encoder.forward(output, domain)
        return output

    def decode(self, input, domain):
        if self.sharing:
            input = self.shared_decoder.forward(input, domain)
        return self.decoders[domain].forward(input)

    def zero_grads(self, dom_a, dom_b):
        self.optimizers[dom_a].zero_grad()
        if self.sharing:
            self.optimizers[-1].zero_grad()
        self.optimizers[dom_b].zero_grad()

    def step_grads(self, dom_a, dom_b):
        self.optimizers[dom_a].step()
        if self.sharing:
            self.optimizers[-1].step()
        self.optimizers[dom_b].step()

    def __repr__(self):
        e, d = self.encoders[0], self.decoders[0]
        e_params = sum([p.numel() for p in e.parameters()])
        d_params = sum([p.numel() for p in d.parameters()])
        return repr(e) +'\n'+ repr(d) +'\n'+ \
            'Created %d Encoder-Decoder pairs' % len(self.encoders) +'\n'+ \
            'Number of parameters per Encoder: %d' % e_params +'\n'+ \
            'Number of parameters per Deocder: %d' % d_params

class D_Plexer(Plexer):
    def __init__(self, n_domains, model, model_args):
        super(D_Plexer, self).__init__()
        self.networks = [model(*model_args) for _ in range(n_domains)]

    def forward(self, input, domain):
        discriminator = self.networks[domain]
        return discriminator.forward(input)

    def __repr__(self):
        t = self.networks[0]
        t_params = sum([p.numel() for p in t.parameters()])
        return repr(t) +'\n'+ \
            'Created %d Discriminators' % len(self.networks) +'\n'+ \
            'Number of parameters per Discriminator: %d' % t_params

class S_Plexer(Plexer):
    def __init__(self, n_domains, model, model_args):
        super(S_Plexer, self).__init__()
        self.networks = [model(*model_args) for _ in range(n_domains)]

    def init_optimizers(self, opt, lr, betas):
         self.optimizers = [opt(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, betas=betas) \
                           for net in self.networks]

    def forward(self, input, domain):
        segmentor = self.networks[domain]
        return segmentor.forward(input)

    def update_lr_2domain(self, new_lr, dom_a, dom_b):
        "Add by lfy."
        # print(len(self.optimizers))
        # print(self.optimizers[dom_a])
        for param_group_a in self.optimizers[dom_a].param_groups:
            param_group_a['lr'] = new_lr
            print('Learning rate of SegA is: %.4f.' % param_group_a['lr'])

        for param_group_b in self.optimizers[dom_b].param_groups:
            # print(param_group_b['lr'])
            param_group_b['lr'] = new_lr
            print('Learning rate of SegB is: %.4f.' % param_group_b['lr'])

    def __repr__(self):
        t = self.networks[0]
        t_params = sum([p.numel() for p in t.parameters()])
        return repr(t) +'\n'+ \
            'Created %d Segmentors' % len(self.networks) +'\n'+ \
            'Number of parameters per Segmentor: %d' % t_params

class SequentialContext(nn.Sequential):
    def __init__(self, n_classes, *args):
        super(SequentialContext, self).__init__(*args)
        self.n_classes = n_classes
        self.context_var = None

    def prepare_context(self, input, domain):
        if self.context_var is None or self.context_var.size()[-2:] != input.size()[-2:]:
            tensor = torch.cuda.FloatTensor if isinstance(input.data, torch.cuda.FloatTensor) \
                     else torch.FloatTensor
            self.context_var = tensor(*((1, self.n_classes) + input.size()[-2:]))

        self.context_var.data.fill_(-1.0)
        self.context_var.data[:,domain,:,:] = 1.0
        return self.context_var

    def forward(self, *input):
        if self.n_classes < 2 or len(input) < 2:
            return super(SequentialContext, self).forward(input[0])
        x, domain = input

        for module in self._modules.values():
            if 'Conv' in module.__class__.__name__:
                context_var = self.prepare_context(x, domain)
                x = torch.cat([x, context_var], dim=1)
            elif 'Block' in module.__class__.__name__:
                x = (x,) + input[1:]
            x = module(x)
        return x

class SequentialOutput(nn.Sequential):
    def __init__(self, *args):
        args = [nn.Sequential(*arg) for arg in args]
        super(SequentialOutput, self).__init__(*args)

    def forward(self, input):
        predictions = []
        layers = self._modules.values()
        for i, module in enumerate(layers):
            output = module(input)
            if i == 0:
                input = output;  continue
            predictions.append( output[:,-1,:,:] )
            if i != len(layers) - 1:
                input = output[:,:-1,:,:]
        return predictions
