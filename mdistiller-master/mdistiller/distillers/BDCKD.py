import torch
import torch.nn as nn
import torch.nn.functional as F
from ._base import Distiller
import logging
import sys

logging.basicConfig(level=logging.WARNING, stream=sys.stdout)
logger = logging.getLogger(__name__)
pil_logger = logging.getLogger("PIL")
pil_logger.setLevel(logging.WARNING)


class BDC(nn.Module):
    def __init__(self, is_vec=True, input_dim=640):
        super(BDC, self).__init__()
        self.is_vec = is_vec
        self.input_dim = input_dim[0]
        output_dim = self.input_dim
        self.output_dim = int(output_dim * output_dim)
        self.temperature = nn.Parameter(
            torch.log((1.0 / (2 * input_dim[1] * input_dim[2])) * torch.ones(1, 1)),
            requires_grad=True,
        )

        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, a=0, mode="fan_out", nonlinearity="leaky_relu"
                )
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def BDCovpool(self, x, t):
        logger.debug(f"BDCovpool: x size before processing {x.size()}")
        batchSize, dim, M = x.data.shape
        I = (
            torch.eye(dim, dim, device=x.device)
            .view(1, dim, dim)
            .repeat(batchSize, 1, 1)
            .type(x.dtype)
        )
        logger.debug(f"BDCovpool: Identity matrix I size {I.size()}")

        I_M = torch.ones(batchSize, dim, dim, device=x.device).type(x.dtype)
        logger.debug(f"BDCovpool: Ones matrix I_M size {I_M.size()}")

        x_pow2 = x.bmm(x.transpose(1, 2))
        logger.debug(f"BDCovpool: x_pow2 size after bmm {x_pow2.size()}")

        dcov = I_M.bmm(x_pow2 * I) + (x_pow2 * I).bmm(I_M) - 2 * x_pow2
        logger.debug(f"BDCovpool: dcov size after computation {dcov.size()}")

        dcov = torch.clamp(dcov, min=0.0001)
        logger.debug(f"BDCovpool: dcov size after clamp {dcov.size()}")

        dcov = torch.exp(t) * dcov
        logger.debug(f"BDCovpool: dcov size after exp and multiplication {dcov.size()}")

        dcov = torch.sqrt(dcov + 1e-5)
        logger.debug(f"BDCovpool: dcov size after sqrt {dcov.size()}")

        t = (
            dcov
            - 1.0 / dim * dcov.bmm(I_M)
            - 1.0 / dim * I_M.bmm(dcov)
            + 1.0 / (dim * dim) * I_M.bmm(dcov).bmm(I_M)
        )
        logger.debug(f"BDCovpool: t size after final computation {t.size()}")

        return t

    def forward(self, x):
        logger.debug(f"BDC.forward: Input size {x.size()}")
        x = self.BDCovpool(x, self.temperature)
        logger.debug(f"BDC.forward: After BDCovpool size {x.size()}")
        x = x.reshape(x.shape[0], -1)
        logger.debug(f"BDC.forward: After reshape size {x.size()}")
        return x


class reduction(nn.Module):
    def __init__(self, input_dim=1000, output_dim=500, activate="relu"):
        super(reduction, self).__init__()
        if activate == "relu":
            self.act = nn.ReLU(inplace=True)
        elif activate == "leaky_relu":
            self.act = nn.LeakyReLU(0.1)
        else:
            self.act = nn.ReLU(inplace=True)

        self.conv_dr_block = nn.Sequential(
            nn.Conv1d(input_dim, output_dim, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm1d(output_dim),
            self.act,
        )

    def forward(self, x):
        logger.debug(f"reduction.forward: Input size {x.size()}")
        x = self.conv_dr_block(x)
        logger.debug(f"reduction.forward: After conv_dr_block size {x.size()}")
        return x


def kd_loss(logits_student, logits_teacher, temperature):
    logger.debug(f"kd_loss: logits_student size {logits_student.size()}")
    logger.debug(f"kd_loss: logits_teacher size {logits_teacher.size()}")
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    loss_kd *= temperature**2
    return loss_kd


def z_score(input_tensor, eps=1e-12, dim=0):
    logger.debug(f"z_score: input_tensor size {input_tensor.size()}")

    mean = torch.mean(input_tensor, dim=dim)
    std = torch.std(input_tensor, dim=dim)
    output_tensor = (input_tensor - mean) / (std + eps)
    return output_tensor


def calc_gram_matrix(features):
    logger.debug(f"calc_gram_matrix: features size {features.size()}")

    return torch.mm(features.transpose(0, 1), features)


class BDCKD(Distiller):
    """Distilling the Knowledge in a Neural Network with BDC"""

    def __init__(self, student, teacher, cfg):
        super(BDCKD, self).__init__(student, teacher)
        self.temperature = cfg.BDCKD.T
        self.ce_loss_weight = cfg.BDCKD.CE_WEIGHT
        self.kd_loss_weight = cfg.BDCKD.KD_WEIGHT
        self.bdc_loss_weight = cfg.BDCKD.BDC_WEIGHT
        self.bdc_inner_weight = cfg.BDCKD.BDC_INNER_WEIGHT
        self.is_imagenet = False
        if cfg.DATASET.TYPE == "imagenet":
            self.reduction = reduction(activate="leaky_relu").cuda()
            self.is_imagenet = True
        self.bdc = BDC(
            is_vec=False,
            input_dim=[cfg.SOLVER.BATCH_SIZE, 100, 1],
        ).cuda()
        self.bdc_inner = BDC(
            is_vec=False,
            input_dim=[100, cfg.SOLVER.BATCH_SIZE, 1],
        ).cuda()

    def forward_train(self, image, target, **kwargs):
        logger.debug(f"BDCKD.forward_train: Image size {image.size()}")

        logits_student, features_student = self.student(image)
        with torch.no_grad():
            logits_teacher, features_teacher = self.teacher(image)

        if self.ce_loss_weight != 0:
            loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        if self.kd_loss_weight != 0:
            loss_kd = self.kd_loss_weight * kd_loss(
                logits_student, logits_teacher, self.temperature
            )

        logits_student = z_score(logits_student)
        logits_teacher = z_score(logits_teacher)

        if self.is_imagenet:
            logits_student_rd = logits_student.unsqueeze(2)
            logits_teacher_rd = logits_teacher.unsqueeze(2)
            logits_student_rd = self.reduction(logits_student_rd)
            logits_teacher_rd = self.reduction(logits_teacher_rd)
            logits_student_rd = logits_student_rd.squeeze()
            logits_teacher_rd = logits_teacher_rd.squeeze()
        else:
            logits_student_rd = logits_student
            logits_teacher_rd = logits_teacher

        y_s_t = logits_student_rd.transpose(0, 1)
        y_t_t = logits_teacher_rd.transpose(0, 1)

        y_s = logits_student_rd.unsqueeze(2)
        y_t = logits_teacher_rd.unsqueeze(2)

        y_s_t = y_s_t.unsqueeze(2)
        y_t_t = y_t_t.unsqueeze(2)
        if self.bdc_loss_weight != 0:
            bdc = self.bdc

            bdc_y_s = bdc(y_s)
            bdc_y_t = bdc(y_t)

            bdc_y_s = z_score(bdc_y_s)
            bdc_y_t = z_score(bdc_y_t)

            bdc_y_s = F.normalize(bdc_y_s, p=2, dim=1)
            bdc_y_t = F.normalize(bdc_y_t, p=2, dim=1)
            loss_bdc = (
                -1 * torch.mul(bdc_y_s, bdc_y_t).sum().mean() * self.bdc_loss_weight
            )

        if self.bdc_inner_weight != 0:
            bdc_inner = self.bdc_inner

            bdc_y_s_inner = bdc_inner(y_s_t)
            bdc_y_t_inner = bdc_inner(y_t_t)

            bdc_y_s_inner = z_score(bdc_y_s_inner, dim=0)
            bdc_y_t_inner = z_score(bdc_y_t_inner, dim=0)

            bdc_y_s_inner = F.normalize(bdc_y_s_inner, p=2, dim=1)
            bdc_y_t_inner = F.normalize(bdc_y_t_inner, p=2, dim=1)

            loss_bdc_inner = (
                -1
                * torch.mul(bdc_y_s_inner, bdc_y_t_inner).sum().mean()
                * self.bdc_inner_weight
            )

        losses_dict = {}
        if self.kd_loss_weight != 0:
            losses_dict["loss_kd"] = loss_kd
        if self.ce_loss_weight != 0:
            losses_dict["loss_ce"] = loss_ce
        if self.bdc_loss_weight != 0:
            losses_dict["loss_bdc"] = loss_bdc
        if self.bdc_inner_weight != 0:
            losses_dict["loss_bdc_inner"] = loss_bdc_inner
        return logits_student, losses_dict

    def get_learnable_parameters(self):
        return [v for k, v in self.student.named_parameters()] + (
            [v for k, v in self.reduction.named_parameters()]
            if hasattr(self, "reduction")
            else []
        )
