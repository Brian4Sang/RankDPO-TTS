import torch
import torch.nn.functional as F
from typing import Tuple


def tpr_loss(disc_real_outputs, disc_generated_outputs, tau):
    loss = 0
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        m_DG = torch.median((dr - dg))
        L_rel = torch.mean((((dr - dg) - m_DG) ** 2)[dr < dg + m_DG])
        loss += tau - F.relu(tau - L_rel)
    return loss


def mel_loss(real_speech, generated_speech, mel_transforms):
    loss = 0
    for transform in mel_transforms:
        mel_r = transform(real_speech)
        mel_g = transform(generated_speech)
        loss += F.l1_loss(mel_g, mel_r)
    return loss


class DPOLoss(torch.nn.Module):
    """
    DPO Loss
    """

    def __init__(self, beta: float, label_smoothing: float = 0.0, ipo: bool = False) -> None:
        super().__init__()
        self.beta = beta
        self.label_smoothing = label_smoothing
        self.ipo = ipo

    def forward(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps
        logits = pi_logratios - ref_logratios
        if self.ipo:
            losses = (logits - 1 / (2 * self.beta)) ** 2  # Eq. 17 of https://arxiv.org/pdf/2310.12036v2.pdf
        else:
            # Eq. 3 https://ericmitchell.ai/cdpo.pdf; label_smoothing=0 gives original DPO (Eq. 7 of https://arxiv.org/pdf/2305.18290.pdf)
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )
        loss = losses.mean()
        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()

        return loss, chosen_rewards, rejected_rewards
    
class RankDPOLoss(torch.nn.Module):
    """
    RankDpo for listwise preference optimization
    
    """
    
    def __init__(self,beta:float = 1.0, rf_free: bool=True):
        super().__init__()
        self.beta = beta
        self.rf_free = rf_free # is rf_model used
        
    def forward(
        self,
        policy_logps: torch.Tensor,
        reward_scores: torch.Tensor,
        reference_logp = None
    )-> torch.Tensor:
        assert policy_logps.shape == reward_scores.shape
        B, k = policy_logps.shape
        
        # compute logits
        if self.rf_free:
            scores = policy_logps
        else:
            assert reference_logp is not None, "rf_logps must be proviced"
            scores = policy_logps - reference_logp
            
            
        # compute loss
        loss = 0.0
        count = 0
        for b in range(B):
            s = scores[b]
            r = reward_scores[b]

                    # compute reward
            ranks = r.argsort(descending=True).argsort() + 1
            discount = torch.log1p(ranks.float())
            gain = 2 * r - 1
            
            for i in range(k):
                for j in range(i):
                    delta_ij = torch.abs(gain[i] - gain[j]) * torch.abs(1 / discount[i] - 1 / discount[j])
                    diff = s[i] - s[j]
                    loss += delta_ij * F.logsigmoid(-self.beta * diff)
                    count += 1
          
        return -loss / count

