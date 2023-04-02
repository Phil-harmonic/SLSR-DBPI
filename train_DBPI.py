import torch
import torch.nn as nn
from networks_DBPI import Downsampler, Upsampler, weights_init_G, weights_init_U
from dataset import DatasetFromH5
from tqdm import tqdm
import argparse
from torch.utils.data import DataLoader
import xlsxwriter


parser = argparse.ArgumentParser(description="SRDenseNet")
parser.add_argument("--root-train", type=str, default="dataset/QSand_hydrate_train.h5",
                    help="Train dataset root")
parser.add_argument("--root-test", type=str, default="dataset/QSand_hydrate_eval.h5", help="Test dataset root")
parser.add_argument("--batchSize", type=int, default=1)
parser.add_argument("--testBatchSize", type=int, default=1)
parser.add_argument("--nEpochs", type=int, default=1)
parser.add_argument("--lr", type=int, default=1e-4)
parser.add_argument("--step", type=int, default=30, help="Set the learning rate to the initial LR decayed by momentum "
                                                         "every n epochs")
parser.add_argument("--cuda", type=bool, default=True, help="Using cuda?")
parser.add_argument("--threads", type=int, default=0)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--logPath", type=str, default="dataset/log")
parser.add_argument("--beta1", type=float, default=0.5)
opt = parser.parse_args()

train_set = DatasetFromH5(opt.root_train)
train_loader = DataLoader(dataset=train_set, batch_size=opt.batchSize, drop_last=True,
                          num_workers=opt.threads, shuffle=True, pin_memory=True)

down_sampler = Downsampler().cuda()
up_sampler = Upsampler().cuda()
down_sampler.apply(weights_init_G)
up_sampler.apply(weights_init_U)

L1 = nn.L1Loss()
optimizer_D = torch.optim.Adam(down_sampler.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizer_U = torch.optim.Adam(up_sampler.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

# start train
log = True
if log:
    workbook = xlsxwriter.Workbook(opt.logPath + ".xlsx")
    sheet = workbook.add_worksheet("loss")

for ep in range(opt.nEpochs):
    avg_loss_du = 0
    avg_loss_ud = 0
    avg_loss = 0
    down_sampler.train()
    up_sampler.train()

    with tqdm(total=len(train_set) - len(train_set) % opt.batchSize, ncols=120, colour="white") as t:
        t.set_description("Epoch: {}/{}".format(ep+1, opt.nEpochs))
        for iteration, data in enumerate(train_loader):
            hr = data[1].cuda()

            if (iteration + 1) % 3000 == 0:
                for param in optimizer_D.param_groups:
                    param["lr"] /= 10.
                for param in optimizer_U.param_groups:
                    param["lr"] /= 10.

            optimizer_D.zero_grad()
            optimizer_U.zero_grad()

            # down_sample -> up_sample
            g_down = down_sampler(hr)
            g_down_up = up_sampler(torch.clamp(g_down.detach(), -1, 1))

            # up_sample -> down_sample
            g_up = up_sampler(hr)
            g_up_down = down_sampler(torch.clamp(g_up.detach() + 1e-1 * torch.randn_like(g_up), -1, 1))

            rand_input = torch.rand(3).cuda().view(1, 3, 1, 1) * 2 - 1
            rand_l = rand_input * torch.ones_like(g_down)
            rand_h = rand_input * torch.ones_like(hr)
            d_rand = down_sampler(rand_h)
            u_rand = up_sampler(rand_l)

            loss_du = L1(g_down_up, hr)
            loss_ud = L1(g_up_down, hr)
            avg_loss_du += loss_du
            avg_loss_ud += loss_ud

            total_loss = loss_ud + loss_du + L1(d_rand, rand_l) + .1 * L1(u_rand, rand_h)
            avg_loss += total_loss

            # sheet.write(iteration, 0, loss_du.item())
            # sheet.write(iteration, 1, loss_ud.item())
            # sheet.write(iteration, 2, total_loss.item())

            total_loss.backward()
            optimizer_D.step()
            optimizer_U.step()

            # t.set_postfix(loss_ud="{:.6f}".format(loss_ud.item()), loss_du="{.6f}".format(loss_du.item()))
            t.set_postfix(loss="{:.6f}".format(total_loss.item()))
            t.update(len(hr))

        avg_loss_du /= len(train_loader)
        avg_loss_ud /= len(train_loader)
        avg_loss /= len(train_loader)
        print('#' * 10 + "Avg_Loss_DU: {:.6f}     Avg_Loss_UD: {:.6f}     Avg_Loss: {:.6f}".format(
            avg_loss_du, avg_loss_ud, avg_loss) + '#' * 10 + "\n")
        sheet.write(ep, 0, avg_loss_du)
        sheet.write(ep, 1, avg_loss_ud)
        sheet.write(ep, 2, avg_loss)

        torch.save(down_sampler.state_dict(), "weights/down_sampler.pth")
        torch.save(up_sampler.state_dict(), "weights/up_sampler.pth")

workbook.close()
