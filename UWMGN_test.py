from __future__ import print_function
import numpy as np
import time
from models import *
from lin_loss import *
import time
from skimage.measure import compare_psnr
from utils.denoising_utils import *

os.environ['CUDA VISIBLE_DEVICES'] = '0'

torch.backends.cudnn.enable = True

torch.backends.cudnn.benchmark = True

dtype = torch.cuda.FloatTensor
# dtype = torch.cuda.FloatTensor
imsize = -1
PLOT = True

def np_plot(np_matrix, title):
    plt.clf()
    fig = plt.imshow(np_matrix.transpose(1, 2, 0), interpolation = 'nearest')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.title(title)
    plt.axis('off')
    plt.pause(0.05)

def add_noise(model):
    for n in [x for x in model.parameters() if len(x.size()) == 4]:
        noise = torch.randn(n.size()) * param_noise_sigma * learning_rate
        noise = noise.type(dtype)
        n.data = n.data + noise
start = time.time()
def closure_sgld():
    global i, net_input, sgld_mean, sample_count, psrn_noisy_last, last_net, sgld_mean_each
    if reg_noise_std > 0:
        net_input = net_input_saved + (noise.normal_() * reg_noise_std)

    out = net2(net_input)
    out = (img_np_torch * out) + img_np1_torch * (1 - out)  # [1,3,384,576]

    loss1 = IE(out)
    loss2 = TV(out)*10
    loss3 = mse(out, img_noisy_torch) * 1000
    # loss4 = mse(out, img_np1_torch) * 1000
    # loss5 = mse(out, img_np1_torch) * 1000

    total_loss = loss1 + loss2 + loss3
    # total_loss.requires_grad = True
    total_loss.backward()


    out_np = out.detach().cpu().numpy()[0]

    # if i == 100:
    #     img = np_to_pil(out.detach().cpu().numpy()[0])
    #     # img.save('G:/pycharmwork/deep-image-prior/data/low_light/ycbcr_image/y/avg{}.png'.format(i))  # 保存
        # img.save(
        #     os.path.join(r'E:\linzhenyu\deep-image-prior\data\denoising\denoising_result\temp', str(sigma), str(num)+'.jpg'))  # 保存

    psrn_noisy = compare_psnr(img_np, out.detach().cpu().numpy()[0])
    psrn_gt = compare_psnr(label_np, out_np)

    sgld_psnr_list.append(psrn_gt)

    # Backtracking
    if roll_back and i % show_every:
        if psrn_noisy - psrn_noisy_last < -10:
            print('Falling back to previous checkpoint.')
            for new_param, net_param in zip(last_net, net2.parameters()):
                net_param.detach().copy_(new_param.cuda())
            return total_loss * 0
        else:
            last_net = [x.detach().cpu() for x in net2.parameters()]
            psrn_noisy_last = psrn_noisy

    print('Iteration %05d    Loss %f  psrn_noisy:%f  psrn_gt: %f ' % (i, total_loss.item(), psrn_noisy, psrn_gt), '\r', end='\n')

    if PLOT and i % show_every == 0:
        out_np = torch_to_np(out)
        plot_image_grid([np.clip(out_np, 0, 1),
                         np.clip(torch_to_np(out), 0, 1)], factor=figsize, nrow=1)  # 网络两次输出图像
        time.sleep(5)
        plt.close()


    if i > burnin_iter and np.mod(i, MCMC_iter) == 0:  # 当 i 大于burnin_iter = 7000 时  且 i 能被 MCMC_iter 整除
        sgld_mean += out_np
        sample_count += 1.

    if i > burnin_iter:   # 记录 i 大于burnin_iter 之后的每一步输出
        sgld_mean_each += out_np
        sgld_mean_tmp = sgld_mean_each / (i - burnin_iter)
        sgld_mean_psnr_each = compare_psnr(label_np, sgld_mean_tmp)
        sgld_psnr_mean_list.append(sgld_mean_psnr_each)  # record the PSNR of avg after burn-in

    i += 1
    return total_loss

if __name__ == "__main__":
#加载数据
    P_sgld=[]
    for num in range(6,7):
        num = num + 1
        sigma = 10
        sigma_ = sigma / 255.

        label = os.path.join(r'E:\linzhenyu\images\common10', str(num)+'.jpg')#标签图像
        label_pil = crop_image(get_image(label, imsize)[0], d=32)  # 保证w h 都能被32整除
        label_np = pil_to_np(label_pil)
        img_label_torch = np_to_torch(label_np)

        img_noisy_pil, img_noisy_np = get_noisy_image(label_np, sigma_)  # 噪声图像

        # load image
        fname = os.path.join(r'E:\linzhenyu\images\first_result\common10\bm3d_result\common10', str(sigma), str(num)+'.jpg')  # bm3d 图像 512*512
        # add noise
        img_pil = crop_image(get_image(fname, imsize)[0], d=32)  # 保证w h 都能被32整除
        img_np = pil_to_np(img_pil)  # 待处理图  ndarray  [1,512,512]
        # img_noisy_pil, img_noisy_np = get_noisy_image(img_np, sigma_) #噪声图像

        # 添加辅助通道
        fname1 =os.path.join( r'E:\linzhenyu\images\first_result\common10\ffdnetresult', str(sigma), str(num)+'.jpg')  # ffdnet 辅助图
        # fname1 = r'E:\linzhenyu\images\ffdnetresult\40\7.jpg'
        img_pil1 = crop_image(get_image(fname1, imsize)[0], d=32)
        img_np1 = pil_to_np(img_pil1)  # 辅助图像


        img_noisy_torch = np_to_torch(img_noisy_np).type(dtype)  # .type(dtype)  # 噪声图像  [1,1,512,512]
        img_np_torch = np_to_torch(img_np).type(dtype)  # .type(dtype)  # 初步降噪图像  [1,1,512,512]
        img_np1_torch = np_to_torch(img_np1).type(dtype)  # .type(dtype)  # 初步降噪图像  [1,1,512,512]
        # label='data/low_light/3.bmp' # 标签图
        # img_pil = crop_image(get_image(label, imsize)[0], d=32) # 保证w h 都能被32整除
        # label_np = pil_to_np(img_pil)  # 原图  ndarray  [1,512,512]

        if PLOT:
            # plot_image_grid([img_np, img_np1,img_noisy_np], 3, 10)  # show 原图、初步降噪图像和噪声图像
            plot_image_grid([img_noisy_np, img_np, img_np1], 3, 10)  # show 原图、初步降噪图像和噪声图像

        INPUT = 'noise'  # 初始化网络输入方法 noise or meshgrid(for large-hole inpainting)
        pad = 'reflection'  # 卷积操作填充方法
        OPT_OVER = 'net'

        reg_noise_std = 1. / 30.  # 每次迭代后加normal noise 的标准差
        LR = 0.01  # learning Rate
        OPTIMIZER = 'adam'  # 优化器
        show_every = 200  # 每100次迭代显示结果


        weight_decay = 5e-8

        num_iter = 1000  # 迭代次数
        input_depth = 2  # 输入的维度  bm3d+dncnn为输入，维度为2
        # input_depth = 32
        figsize = 4  # 显示图像的大小
        learning_rate = 0.01
        roll_back = True  # to prevent numerical issues
        burnin_iter = 400  # burn-in iteration for SGLD    迭代总次数为4000，burnin_iter为2500


        ## SGLD

        sgld_psnr_list = []  # psnr between sgld out and gt
        sgld_mean = 0
        roll_back = True  # To solve the oscillation of model training
        last_net = None
        psrn_noisy_last = 0
        MCMC_iter = 50
        param_noise_sigma = 2

        sgld_mean_each = 0
        sgld_psnr_mean_list = []  # record the PSNR of avg after burn-in

        net2 = get_net(input_depth, 'skip', pad,
                       skip_n33d=128,
                       skip_n33u=128,
                       skip_n11=4,
                       num_scales=5,
                       upsample_mode='bilinear').type(dtype)

        ## Input random noise
        net_input = get_noise(input_depth, INPUT, (img_pil.size[1], img_pil.size[0])).type(dtype).detach()  #噪声作为输入

        # 将初步降噪图像作为输入
        # net_input = torch.cat([img_np_torch, img_np1_torch.type(dtype).detach()], dim=1)


        net_input_saved = net_input.detach().clone()
        noise = net_input.detach().clone()

        i = 0
        sample_count = 0

        s = sum([np.prod(list(p.size())) for p in net2.parameters()])
        print('Number of params: %d' % s)

        # loss
        TV = TVLoss().type(dtype)  # 总变分
        IE = Entropy().type(dtype)  # 信息熵

        mse = torch.nn.MSELoss().type(dtype)  # .type(dtype)
        lin_loss = LinLoss().type(dtype)
        pcloss = PCloss().type(dtype)

        nmi_last = 0
        out_avg = None

        ## Optimizing
        print('Starting optimization with SGLD')
        optimizer = torch.optim.Adam(net2.parameters(), lr=LR, weight_decay=weight_decay)
        for j in range(num_iter):
            optimizer.zero_grad()
            closure_sgld()
            optimizer.step()
            add_noise(net2)
        sgld_mean = sgld_mean / sample_count  # 输出图像的均值图像
        sgld_mean_each = sgld_mean_each/(i - burnin_iter)

        img = np_to_pil(sgld_mean)
        img_each = np_to_pil(sgld_mean_each)
        # img.save('img.jpg')
        img.save(
            os.path.join(r'E:\linzhenyu\deep-image-prior\data\denoising\denoising_result\sgld', str(sigma), str(num)+'.jpg'))  # 保存
        torch.save(sgld_mean, os.path.join(r'E:\linzhenyu\deep-image-prior\data\denoising\denoising_result\sgld', str(sigma), str(num) + '.pt'))

        sgld_mean_psnr = compare_psnr(label_np, sgld_mean)  # 均值图像与原始图像计算psnr
        bm3d_psnr=compare_psnr(label_np, img_np)
        dncnn_psnr = compare_psnr(label_np, img_np1)
        print(sgld_mean_psnr, bm3d_psnr, dncnn_psnr)
        P_sgld.append(sgld_mean_psnr)
        print(P_sgld)
        print(np.mean(P_sgld))

        end = time.time()
        print(end - start)
