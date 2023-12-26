from typing import Dict, Literal

import lightning as L
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler

from models.networks.base_nets import BaseDiscriminator, BaseGenerator
from optim import define_criterion
from utils import net_utils


class SRGAN(L.LightningModule):
    def __init__(
        self,
        generator: BaseGenerator,
        discriminator: BaseDiscriminator,
        *,
        crop_border_ratio: float = 0.75,
        losses: Dict,
        gen_lr: float = 5e-5,
        dis_lr: float = 5e-5,
        dis_update_policy: Literal["always", "adaptive"] = "always",
        dis_update_threshold: float = 0.4,
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
        hr_fake = self.G(lr_data)

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

        hr_fake = self.G(lr_data)

        # ssim_val = self.ssim(y_fake, y_true).mean()
        # lpips_val = self.lpips_alex(y_fake, y_true).mean()
        # self.ssim_validation.append(float(ssim_val))
        # self.lpips_validation.append(float(lpips_val))

        return lr_data, hr_true, hr_fake
