import torch.nn.functional as F
from torch import Tensor
from torch.autograd import Variable
from sklearn import metrics
from utils.denoising_utils import *
import math
import torch.nn as nn
import numpy as np
import cv2  # Faster Fourier transforms than NumPy and Scikit-Image


_euler_num = 2.718281828				# 	euler number
_pi = 3.14159265						# 	pi
_ln_2_pi = 1.837877						# 	ln(2 * pi)
_CLIP_MIN = 1e-6            			# 	min clip value after softmax or sigmoid operations
_CLIP_MAX = 1.0    						# 	max clip value after softmax or sigmoid operations
_POS_ALPHA = 5e-4						# 	add this factor to ensure the AA^T is positive definite
_IS_SUM = 1

# class LinLoss(nn.Module):
#     def __init__(self):
#         super(LinLoss, self).__init__()
#     def forward(self, x1, x2, x3):
#
#         out_np = x1.detach().numpy()[0]
#         out_list = []
#         for g in range(3):
#             out_temp = out_np[g]
#             [H, W] = out_temp.shape
#             out_pil_one_line = out_temp.reshape(H * W)
#             for h in out_pil_one_line:
#                 out_list.append(h)
#
#         x2_np = x2.detach().numpy()[0]
#         img_np1_list = []
#         for i in range(3):
#             img_np1_emp = x2_np[i]
#             [h, w] = img_np1_emp.shape
#             A_one_line = img_np1_emp.reshape(h * w)
#             for j in A_one_line:
#                 img_np1_list.append(j)
#
#         x3_np = x3.detach().numpy()[0]
#         img_noisy_np_list = []
#         for e in range(3):
#             img_noisy_np_temp = x3_np[e]
#             [h, w] = img_noisy_np_temp.shape
#             B_one_line = img_noisy_np_temp.reshape(h * w)
#             for f in B_one_line:
#                 img_noisy_np_list.append(f)
#
#         result_NMI = metrics.normalized_mutual_info_score(img_np1_list,
#                                                           out_list) + metrics.normalized_mutual_info_score(
#             img_noisy_np_list, out_list)  # 互信息最大化
#
#         return Variable(torch.from_numpy(np.asarray(100-result_NMI*100).astype('float32')), requires_grad=True)

class LinLoss(nn.Module):
    def __init__(self):
        super(LinLoss, self).__init__()
    def forward(self, x1, x2, x3):

        out_np = x1.detach().numpy()[0]
        out_list = []

        out_temp = out_np[0,:,:]
        [H, W] = out_temp.shape
        out_pil_one_line = out_temp.reshape(H * W)
        for h in out_pil_one_line:
            out_list.append(h)

        x2_np = x2.detach().numpy()[0]
        img_np1_list = []
        x2_np_temp = x2_np[0, :, :]
        [h, w] = x2_np_temp.shape
        A_one_line = x2_np.reshape(h * w)
        for j in A_one_line:
             img_np1_list.append(j)


        x3_np = x3.detach().numpy()[0]
        img_noisy_np_list = []
        x3_np_temp = x3_np[0,:,:]
        [h, w] = x3_np_temp.shape
        A_one_line = x2_np.reshape(h * w)
        for f in A_one_line:
            img_noisy_np_list.append(f)

        result_NMI = metrics.normalized_mutual_info_score(img_np1_list,
                                                          out_list) + metrics.normalized_mutual_info_score(
            img_noisy_np_list, out_list)  # 互信息最大化

        return Variable(torch.from_numpy(np.asarray(2-result_NMI).astype('float32')), requires_grad=True)

    def backward (self, y):
        return y

class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
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

def simplex(t: Tensor, axis=1) -> bool:
    """
    check if the matrix is the probability distribution
    :param t:
    :param axis:
    :return:
    """
    _sum = t.sum(axis).type(torch.float32)
    _ones = torch.ones_like(_sum, dtype=torch.float32)
    return torch.allclose(_sum, _ones, rtol=1e-4, atol=1e-4)

class Entropy(nn.Module):
    def __init__(self, reduce=True, eps=1e-16):
        super().__init__()
        r"""
        the definition of Entropy is - \sum p(xi) log (p(xi))
        """
        self.eps    = eps
        self.reduce = reduce

    def forward(self, input: torch.Tensor):
        # assert input.shape.__len__() >= 2
        # b, _, *s = input.shape
        input = F.softmax(input[0,0,:,:], dim=1)
        # assert simplex(input)
        e = input * (input + self.eps).log()
        e = -1.0 * e.sum(1)
        # assert e.shape == torch.Size([b, *s]), "Size is not right  chen's assert"
        if self.reduce:
            return torch.mean(e)
        return e

class get_entropy(nn.Module):
    def __init__(self):
        super(get_entropy, self).__init__()

    def forward(self, data_df, columns=None):
        if (columns is None) and (data_df.shape[1] > 1):
            pe_value_array = data_df[columns].unique()
            ent = 0.0
        for x_value in pe_value_array:
            p = float(data_df[data_df[columns] == x_value].shape[0]) / data_df.shape[0]
            logp = np.log2(p)
            ent -= p * logp
        return ent

class PCloss(nn.Module):
    def __init__(self):
        super(PCloss, self).__init__()
    def forward(self, InputImage):
        InputImage = np.array(np_to_pil(torch_to_np(InputImage)))
        minWaveLength = 3
        mult = 2.1
        NumberScales = 5
        NumberAngles = 6
        sigmaOnf = 0.55
        k = 2.0
        cutOff = 0.5
        g = 10
        epsilon = .0001 # Used to prevent division by zero.

        f_cv = cv2.dft(np.float32(InputImage), flags=cv2.DFT_COMPLEX_OUTPUT)

        #------------------------------
        nrows, ncols = InputImage.shape

        sum= nrows * ncols

        epsilon = [0.0001] * sum
        epsilon = np.array(epsilon).reshape(nrows, ncols)
        EO = np.zeros((nrows,ncols,NumberScales,NumberAngles),dtype=complex)
        PC = np.zeros((nrows,ncols,NumberAngles))
        covx2 = np.zeros((nrows,ncols))
        covy2 = np.zeros((nrows,ncols))
        covxy = np.zeros((nrows,ncols))
        EnergyV = np.zeros((nrows,ncols,3))
        pcSum = np.zeros((nrows,ncols))

        # Matrix of radii
        cy = math.floor(nrows/2)
        cx = math.floor(ncols/2)
        y, x = np.mgrid[0:nrows, 0:ncols]
        y = (y-cy)/nrows
        x = (x-cx)/ncols

        radius = np.sqrt(x**2 + y**2)
        radius[cy, cx] = 1

        theta = np.arctan2(-y, x)
        sintheta = np.sin(theta)
        costheta = np.cos(theta)
        annularBandpassFilters = np.empty((nrows, ncols, NumberScales))

        filterorder = 15  # filter 'sharpness'
        cutoff = .45
        normradius = radius / (abs(x).max()*2)
        lowpassbutterworth = 1.0 / (1.0 + (normradius / cutoff)**(2*filterorder))
        for s in np.arange(NumberScales):
            wavelength = minWaveLength*mult**s
            fo = 1.0/wavelength                  # Centre frequency of filter.
            logGabor = np.exp((-(np.log(radius/fo))**2) / (2 * math.log(sigmaOnf)**2))
            annularBandpassFilters[:,:,s] = logGabor*lowpassbutterworth  # Apply low-pass filter
            annularBandpassFilters[cy,cx,s] = 0          # Set the value at the 0 frequency point of the filter
                                                         # back to zero (undo the radius fudge).

        for o in np.arange(NumberAngles):
            # Construct the angular filter spread function
            angl = o*math.pi/NumberAngles # Filter angle.

            ds = sintheta * math.cos(angl) - costheta * math.sin(angl)      # Difference in sine.
            dc = costheta * math.cos(angl) + sintheta * math.sin(angl)      # Difference in cosine.
            dtheta = np.abs(np.arctan2(ds,dc))                              # Absolute angular distance.

            # Scale theta so that cosine spread function has the right wavelength
            #   and clamp to pi
            dtheta = np.minimum(dtheta*NumberAngles/2, math.pi)
            spread = (np.cos(dtheta)+1)/2

            sumE_ThisOrient   = np.zeros((nrows, ncols))  # Initialize accumulator matrices.
            sumO_ThisOrient   = np.zeros((nrows, ncols))
            sumAn_ThisOrient  = np.zeros((nrows, ncols))
            Energy            = np.zeros((nrows, ncols))

            maxAn = []
            for s in np.arange(NumberScales):
                filter = annularBandpassFilters[:,:,s] * spread # Multiply radial and angular
                                                                # components to get the filter.

                criticalfiltershift = np.fft.ifftshift( filter )
                criticalfiltershift_cv = np.empty((nrows, ncols, 2))
                for ip in range(2):
                    criticalfiltershift_cv[:,:,ip] = criticalfiltershift

                # Convolve image with even and odd filters returning the result in EO
                MatrixEO = cv2.idft( criticalfiltershift_cv * f_cv )
                EO[:,:,s,o] = MatrixEO[:,:,1] + 1j*MatrixEO[:,:,0]

                An = cv2.magnitude(MatrixEO[:,:,0], MatrixEO[:,:,1])    # Amplitude of even & odd filter response.

                sumAn_ThisOrient = sumAn_ThisOrient + An             # Sum of amplitude responses.
                sumE_ThisOrient = sumE_ThisOrient + MatrixEO[:,:,1] # Sum of even filter convolution results.
                sumO_ThisOrient = sumO_ThisOrient + MatrixEO[:,:,0] # Sum of odd filter convolution results.

                if s == 0:
                    tau = np.median(sumAn_ThisOrient) / math.sqrt(math.log(4))
                    maxAn = An
                else:
                    maxAn = np.maximum(maxAn,An)

            EnergyV[:,:,0] = EnergyV[:,:,0] + sumE_ThisOrient
            EnergyV[:,:,1] = EnergyV[:,:,1] + math.cos(angl)*sumO_ThisOrient
            EnergyV[:,:,2] = EnergyV[:,:,2] + math.sin(angl)*sumO_ThisOrient

            # Get weighted mean filter response vector, this gives the weighted mean
            # phase angle.
            XEnergy = np.sqrt(sumE_ThisOrient**2 + sumO_ThisOrient**2) + epsilon
            MeanE = sumE_ThisOrient / XEnergy
            MeanO = sumO_ThisOrient / XEnergy

            for s in np.arange(NumberScales):
                # Extract even and odd convolution results.
                E = EO[:,:,s,o].real
                O = EO[:,:,s,o].imag

                Energy = Energy + E*MeanE + O*MeanO - np.abs(E*MeanO - O*MeanE)

            totalTau = tau * (1 - (1/mult)**NumberScales)/(1-(1/mult))

            EstNoiseEnergyMean = totalTau*math.sqrt(math.pi/2)        # Expected mean and std
            EstNoiseEnergySigma = totalTau*math.sqrt((4-math.pi)/2)   # values of noise energy

            T =  EstNoiseEnergyMean + k*EstNoiseEnergySigma # Noise threshold

            Energy = np.maximum(Energy - T, 0)

            width = (sumAn_ThisOrient/(maxAn + epsilon) - 1) / (NumberScales-1)

            weight = 1.0 / (1 + np.exp( (cutOff - width)*g))

            # Apply weighting to energy and then calculate phase congruency
            PC[:,:,o] = weight*Energy/sumAn_ThisOrient   # Phase congruency for this orientatio

            pcSum = pcSum + PC[:,:,o]

            # Build up covariance data for every point
            covx = PC[:,:,o]*math.cos(angl)
            covy = PC[:,:,o]*math.sin(angl)
            covx2 = covx2 + covx**2
            covy2 = covy2 + covy**2
            covxy = covxy + covx*covy

        covx2 = covx2/(NumberAngles/2)
        covy2 = covy2/(NumberAngles/2)
        covxy = 4*covxy/NumberAngles   # This gives us 2*covxy/(norient/2)
        denom = np.sqrt(covxy**2 + (covx2-covy2)**2)+epsilon
        M = (covy2+covx2 + denom)/2          # Maximum moment
        m = (covy2+covx2 - denom)/2          # ... and minimum moment

        # Orientation and feature phase/type computation
        ORM = np.arctan2(EnergyV[:,:,2], EnergyV[:,:,1])
        ORM[ORM<0] = ORM[ORM<0]+math.pi       # Wrap angles -pi..0 to 0..pi
        return M, m, nrows, ncols

"""
Well-exposedness Quality Measure
"""
class exposednessloss(nn.Module):
    def __init__(self):
        super(exposednessloss, self).__init__()
    def forward(self, image, sigma=0.2):
        """
       FUNCTION: exposedness
            Call to compute third quality measure - exposure using a gaussian curve
         INPUTS:
            image = input image (colored)
            sigma = gaussian curve parameter
        OUTPUTS:
            exposedness measure
        """
    #'-----------------------------------------------------------------------------#
        image = cv2.normalize(image, None, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)
        gauss_curve = lambda i : np.exp(-((i-0.5)**2) / (2*sigma*sigma))
        E = gauss_curve(image[:,:,2])
        return E.astype('float64')

