import torch
import torch.nn as nn
import torch.nn.functional as F
class KDLoss(nn.Module):
    def __init__(self, temp_factor):
        super(KDLoss, self).__init__()
        self.temp_factor = temp_factor
        self.kl_div = nn.KLDivLoss(reduction="sum")

    def forward(self, input, target):
        log_p = torch.log_softmax(input/self.temp_factor, dim=1)
        q = torch.softmax(target/self.temp_factor, dim=1)
        loss = self.kl_div(log_p, q)*(self.temp_factor**2)/input.size(0)
        return loss
class DKDLoss(nn.Module):
    def __init__(self, temperature, alpha, beta):
        super(DKDLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_div = nn.KLDivLoss(reduction='batchmean')

    def forward(self, student_outputs, teacher_outputs, labels):
        t_logits = teacher_outputs / self.temperature
        s_logits = student_outputs / self.temperature

        # Separate target and non-target logits
        batch_size = s_logits.size(0)
        target_mask = F.one_hot(labels, num_classes=s_logits.size(1)).float()
        non_target_mask = 1 - target_mask

        t_logits_target = (t_logits * target_mask).sum(dim=1, keepdim=True)
        s_logits_target = (s_logits * target_mask).sum(dim=1, keepdim=True)
        t_logits_non_target = t_logits * non_target_mask
        s_logits_non_target = s_logits * non_target_mask

        ce_loss = self.ce_loss(student_outputs, labels)
        kd_target = self.kl_div(F.log_softmax(s_logits_target, dim=1), F.softmax(t_logits_target, dim=1))
        kd_non_target = self.kl_div(F.log_softmax(s_logits_non_target, dim=1), F.softmax(t_logits_non_target, dim=1))

        return self.alpha * ce_loss + self.beta * kd_target + (1 - self.alpha - self.beta) * kd_non_target

