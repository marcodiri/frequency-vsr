from typing import Any, Dict, Literal

import lightning as L
import torch
import torch.nn.functional as F
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler

from models.networks.base_nets import BaseDiscriminator, BaseGenerator
from optim import define_criterion


class SRGAN(L.LightningModule):
    def __init__(
        self,
        generator: BaseGenerator,
        discriminator: BaseDiscriminator,
        *,
        losses: Dict,
        gen_lr: float = 5e-5,
        dis_lr: float = 5e-5,
    ):
        super(SRGAN, self).__init__()
        self.save_hyperparameters(ignore=["generator", "discriminator"])
        self.G = generator
        self.D = discriminator

        # pixel criterion
        self.pix_crit, self.pix_w = define_criterion(losses.get("pixel_crit"))

        # feature criterion
        self.feat_crit, self.feat_w = define_criterion(losses.get("feature_crit"))
        if losses.get("feature_crit")["type"] == "CosineSimilarity":
            self.feat_net = VGGFeatureExtractor(
                losses["feature_crit"].get("feature_layers", [8, 17, 26, 35])
            )
        else:
            self.feat_net = None

        # frequency criterion
        self.freq_crit, (self.freq_high_w, self.freq_low_w) = define_criterion(
            losses.get("freq_crit")
        )
        if self.freq_crit is not None:

            class FreqNet(L.LightningModule):
                def __init__(self, G):
                    super().__init__()
                    self.init_conv = G.init_conv
                    self.feb_block = G.feb_block

                def forward(self, x):
                    self.freeze()

                    out = self.init_conv(x)

                    out_blocks, low_f, high_f = self.feb_block(out)

                    self.unfreeze()

                    return low_f, high_f

            self.freq_net = FreqNet(self.G)

        # gan criterion
        self.gan_crit, self.gan_w = define_criterion(losses.get("gan_crit"))

        self.automatic_optimization = False

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optim_G = torch.optim.Adam(params=self.G.parameters(), lr=self.hparams.gen_lr)
        optim_D = torch.optim.SGD(params=self.D.parameters(), lr=self.hparams.dis_lr)

        return optim_G, optim_D

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        optim_G, optim_D = self.optimizers()

        # ------------ prepare data ------------ #
        hr_true, lr_data = batch

        # ------------ clear optimizers ------------ #
        optim_G.zero_grad()
        optim_D.zero_grad()

        # ------------ forward G ------------ #
        hr_fake, low_f_fake, high_f_fake = self.G(lr_data)

        # ------------ forward D ------------ #
        self.D.unfreeze()

        # forward real sequence (gt)
        real_pred_D = self.D(hr_true)

        # forward fake sequence (hr)
        fake_pred_D = self.D(hr_fake.detach())

        # ------------ optimize D ------------ #
        to_log, to_log_prog = {}, {}

        loss_real_D = self.gan_crit(real_pred_D, True)
        loss_fake_D = self.gan_crit(fake_pred_D, False)
        loss_D = loss_real_D + loss_fake_D

        # update D
        self.manual_backward(loss_D)
        optim_D.step()
        to_log["D_real_loss"] = loss_real_D
        to_log["D_fake_loss"] = loss_fake_D

        to_log_prog["D_loss"] = loss_D

        # ------------ optimize G ------------ #
        self.D.freeze()

        # calculate losses
        loss_G = 0

        # pixel (pix) loss
        if self.pix_crit is not None:
            loss_pix_G = self.pix_crit(hr_fake, hr_true)
            loss_G += self.pix_w * loss_pix_G
            to_log["G_pixel_loss"] = loss_pix_G

        # frequency loss
        if self.freq_crit is not None:
            high_f_true, low_f_true = self.freq_net(hr_true)
            _, _, c, h_l, w_l = high_f_fake.shape
            _, _, c, h_h, w_h = high_f_true.shape

            high_f_true = high_f_true.view(-1, c, h_h, w_h)
            low_f_true = low_f_true.view(-1, c, h_h, w_h)

            # # small true vs small fake
            # high_f_fake = high_f_fake.view(-1, c, h_l, w_l)
            # low_f_fake = low_f_fake.view(-1, c, h_l, w_l)
            # high_f_true_small = F.interpolate(
            #     high_f_true, size=(h_l, w_l), mode="nearest"
            # )
            # low_f_true_small = F.interpolate(
            #     low_f_true, size=(h_l, w_l), mode="nearest"
            # )
            # loss_high_f1 = self.freq_crit(high_f_true_small, high_f_fake)
            # loss_low_f1 = self.freq_crit(low_f_true_small, low_f_fake)

            # big true vs big fake
            high_f_fake, low_f_fake = self.freq_net(hr_fake)
            high_f_fake = high_f_fake.view(-1, c, h_h, w_h)
            low_f_fake = low_f_fake.view(-1, c, h_h, w_h)
            loss_high_f2 = self.freq_crit(high_f_true, high_f_fake)
            loss_low_f2 = self.freq_crit(low_f_true, low_f_fake)

            # loss_G += self.freq_high_w * loss_high_f1 + self.freq_low_w * loss_low_f1
            loss_G += self.freq_high_w * loss_high_f2 + self.freq_low_w * loss_low_f2
            to_log["G_high_freq_loss"] = loss_high_f2
            to_log["G_low_freq_loss"] = loss_low_f2

        # feature (feat) loss
        if self.feat_crit is not None:
            if self.feat_net is None:
                loss_feat_G = self.feat_crit(hr_fake, hr_true.detach()).mean()
            else:
                hr_feat_lst = self.feat_net(hr_fake)
                gt_feat_lst = self.feat_net(hr_true)
                loss_feat_G = 0
                for hr_feat, gt_feat in zip(hr_feat_lst, gt_feat_lst):
                    loss_feat_G += self.feat_crit(hr_feat, gt_feat.detach())

            loss_G += self.feat_w * loss_feat_G
            to_log_prog["G_lpip_loss"] = loss_feat_G

        # gan loss
        fake_pred_G = self.D(hr_fake)

        loss_gan_G = self.gan_crit(fake_pred_G, True)
        loss_G += self.gan_w * loss_gan_G
        to_log["G_gan_loss"] = loss_gan_G
        to_log_prog["G_loss"] = loss_G

        # update G
        self.manual_backward(loss_G)
        optim_G.step()

        self.log_dict(to_log_prog, prog_bar=True)
        self.log_dict(to_log, prog_bar=False)

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        hr_true, lr_data = batch

        hr_fake, _, _ = self.G(lr_data)

        # ssim_val = self.ssim(y_fake, y_true).mean()
        # lpips_val = self.lpips_alex(y_fake, y_true).mean()
        # self.ssim_validation.append(float(ssim_val))
        # self.lpips_validation.append(float(lpips_val))

        return lr_data, hr_true, hr_fake
