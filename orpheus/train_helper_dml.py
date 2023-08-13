import torch
import torch.nn as nn
import torch.nn.functional as F

import core.loss as closs
import core.process as utils

from model.mol_translate import MoLTranslate

class TrainerAE(nn.Module):
    def __init__(self, backbone, prior, slicer, sample_rate=44100):
        super().__init__()

        scales = [2048, 1024, 512, 256, 128, 32]
        num_mels = [320, 160, 80, 40, 20, 5]
        maxes = [33, 8, 2.5, 0.6, 0.18, 0.015]

        self.backbone = backbone
        self.prior = prior
        self.slicer = slicer
        
        self.translator = MoLTranslate(scales, num_mels)

        log_epsilon = 1e-7
        self.num_skipped_features = 1

        stft = closs.MultiScaleSTFT(scales, sample_rate, num_mels=num_mels)
        self.entropy_distance = closs.AudioDistanceCE(stft, self.translator, 1024, 4, {scale: max for scale, max in zip(scales, maxes)})
        self.distance = closs.AudioDistanceV1(stft, log_epsilon)

    def stage1(self):
        self.translator.cuda()
        self.backbone.cuda()
        self.prior.cuda()
        self.slicer.cuda()
        self.distance.cuda()
    
    def stage2(self):
        self.backbone.freeze_encoder()
        self.backbone.cuda()
        self.distance.cuda()

    def split_features(self, features):
        feature_real = []
        feature_fake = []
        for scale in features:
            true, fake = zip(*map(
                lambda x: torch.split(x, x.shape[0] // 2, 0),
                scale,
            ))
            feature_real.append(true)
            feature_fake.append(fake)
        return feature_real, feature_fake

    def forward_nm(self, x):
        x_subbands = self.backbone.decompose(x)
        y_subbands, _ = self.backbone.forward_nm(x_subbands)

        y = self.backbone.recompose(y_subbands)

        mb_dist = self.distance(y_subbands, x_subbands)
        fb_dist = self.distance(y, x)

        with torch.no_grad():
            f_loss = F.mse_loss(y, x)

        return (mb_dist["spectral_distance"], fb_dist["spectral_distance"], torch.tensor(0.), f_loss)

    def forward_wd(self, x, discriminator):
        x_subbands = self.backbone.decompose(x)
        y_subbands, _ = self.backbone.forward_nm(x_subbands)

        y = self.backbone.recompose(y_subbands)

        mb_dist = self.distance(y_subbands, x_subbands)
        fb_dist = self.distance(y, x)

        r_loss = mb_dist["spectral_distance"] + fb_dist["spectral_distance"]

        xy = torch.cat([x, y], 0)
        features = discriminator(xy)

        feature_real, feature_fake = self.split_features(features)

        feature_matching_distance = 0.
        loss_dis = 0
        loss_adv = 0

        for scale_real, scale_fake in zip(feature_real, feature_fake):
            current_feature_distance = sum(
                map(
                    lambda a, b : closs.mean_difference(a, b, relative=True),
                    scale_real[self.num_skipped_features:],
                    scale_fake[self.num_skipped_features:],
                )) / len(scale_real[self.num_skipped_features:])

            feature_matching_distance = feature_matching_distance + current_feature_distance

            _dis, _adv = closs.hinge_gan(scale_real[-1], scale_fake[-1])

            loss_dis = loss_dis + _dis
            loss_adv = loss_adv + _adv

        feature_matching_distance = feature_matching_distance / len(feature_real)

        with torch.no_grad():
            f_loss = F.mse_loss(y, x)

        return (r_loss, loss_adv, loss_dis, feature_matching_distance, f_loss)

    def forward(self, x):
        # y_subbands, _, x_subbands, z, mask = self.backbone(x)

        x_subbands = self.backbone.decompose(x)
        B, C, L = x_subbands.size()
        y_subbands_mix, z = self.backbone.forward_nm(x_subbands)

        # ce_mask = mask.repeat(1, 2048 // 16)

        y_weights, y_means, y_scales, y_subbands = self.backbone.sum_mix(y_subbands_mix)

        continuity_loss = self.entropy_distance(x_subbands, y_means.transpose(1, 2), y_weights.transpose(1, 2), y_scales.transpose(1, 2), mask=None)["entropy_distance"]

        print(continuity_loss.item())

        y = self.backbone.recompose(y_subbands)
        fb_dist = self.distance(y, x)

        z_samples = self.prior.sample(z.shape[0] * z.shape[2])

        d_loss = self.slicer.fgw_dist(z.transpose(1, 2).reshape(-1, z.shape[1]), z_samples)

        with torch.no_grad():
            f_loss = F.mse_loss(y, x)

        return (continuity_loss, fb_dist["spectral_distance"], d_loss, f_loss)