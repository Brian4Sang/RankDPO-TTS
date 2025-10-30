# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Zhihao Du)
#               2025 Alibaba Inc (authors: Xiang Lyu, Bofan Zhou)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
import math
import torch.nn.functional as F
from matcha.models.components.flow_matching import BASECFM
from cosyvoice.utils.common import set_all_random_seed


class ConditionalCFM(BASECFM):
    def __init__(self, in_channels, cfm_params, n_spks=1, spk_emb_dim=64, estimator: torch.nn.Module = None):
        super().__init__(
            n_feats=in_channels,
            cfm_params=cfm_params,
            n_spks=n_spks,
            spk_emb_dim=spk_emb_dim,
        )
        self.t_scheduler = cfm_params.t_scheduler
        self.training_cfg_rate = cfm_params.training_cfg_rate
        self.inference_cfg_rate = cfm_params.inference_cfg_rate
        in_channels = in_channels + (spk_emb_dim if n_spks > 0 else 0)
        # Just change the architecture of the estimator here
        self.estimator = estimator

    @torch.inference_mode()
    def forward(self, mu, mask, n_timesteps, temperature=1.0, spks=None, cond=None, prompt_len=0, cache=torch.zeros(1, 80, 0, 2)):
        """Forward diffusion

        Args:
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            n_timesteps (int): number of diffusion steps
            temperature (float, optional): temperature for scaling noise. Defaults to 1.0.
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)
            cond: Not used but kept for future purposes

        Returns:
            sample: generated mel-spectrogram
                shape: (batch_size, n_feats, mel_timesteps)
        """

        z = torch.randn_like(mu).to(mu.device).to(mu.dtype) * temperature
        cache_size = cache.shape[2]
        # fix prompt and overlap part mu and z
        if cache_size != 0:
            z[:, :, :cache_size] = cache[:, :, :, 0]
            mu[:, :, :cache_size] = cache[:, :, :, 1]
        z_cache = torch.concat([z[:, :, :prompt_len], z[:, :, -34:]], dim=2)
        mu_cache = torch.concat([mu[:, :, :prompt_len], mu[:, :, -34:]], dim=2)
        cache = torch.stack([z_cache, mu_cache], dim=-1)

        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device, dtype=mu.dtype)
        if self.t_scheduler == 'cosine':
            t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
        return self.solve_euler(z, t_span=t_span, mu=mu, mask=mask, spks=spks, cond=cond), cache

    def solve_euler(self, x, t_span, mu, mask, spks, cond, streaming=False,
                stochastic=False, return_logprob=False):
        B = x.size(0)
        LOG2PI = math.log(2.0 * math.pi)
        logprob_total = None
        if return_logprob:
            logprob_total = x.new_zeros(B)

        # 当前步长
        t = t_span[0]
        dt = t_span[1] - t_span[0]

        sol = []
        for step in range(1, len(t_span)):
            # 1) 拼 2B 批次：cond / uncond
            x_in   = torch.cat([x,   x],   dim=0)          # [2B, 80, T]
            mask_in= torch.cat([mask,mask],dim=0)          # [2B, 1,  T]
            mu_in  = torch.cat([mu,  torch.zeros_like(mu)], dim=0)
            spks_in= torch.cat([spks,torch.zeros_like(spks)], dim=0) if spks is not None else None
            cond_in= torch.cat([cond,torch.zeros_like(cond)], dim=0) if cond is not None else None
            t_in   = torch.cat([t.expand(B), t.expand(B)], dim=0)  # [2B]

            # 2) 过 estimator，拿到 (mu_pred, log_sigma)，再 softplus 得 sigma
            mu_pred, log_sigma = self.forward_estimator(
                x_in, mask_in, mu_in, t_in, spks_in, cond_in, streaming=streaming,
                return_mu_logsigma=True
            )
            sigma = F.softplus(log_sigma) + 1e-6

            # 3) 拆回 cond / uncond
            mu_c, mu_u = mu_pred.split(B, dim=0)   # [B, 80, T] each
            sg_c, sg_u = sigma.split(B, dim=0)

            # 4) 速度：确定性（μ）或随机（μ + σ·ε）
            if stochastic:
                eps_c = torch.randn_like(mu_c)
                eps_u = torch.randn_like(mu_u)
                u_c = mu_c + sg_c * eps_c
                u_u = mu_u + sg_u * eps_u
                if return_logprob:
                    step_logprob = -0.5 * ( ((u_c - mu_c)**2)/(sg_c**2) + 2.0*torch.log(sg_c) + LOG2PI )
                    # mask 加权，再按句归一
                    step_logprob = (step_logprob * mask).sum(dim=[1,2]) / (mask.sum(dim=[1,2]) + 1e-6)
                    logprob_total = logprob_total + step_logprob
            else:
                u_c, u_u = mu_c, mu_u

            # 5) CFG 组合（与你原公式一致）
            dphi_dt = ((1.0 + self.inference_cfg_rate) * u_c - self.inference_cfg_rate * u_u)

            # 6) 欧拉步进
            x = x + dt * dphi_dt
            t = t + dt

            sol.append(x)
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t

        return sol[-1].float(), (logprob_total if return_logprob else None)

    def forward_estimator(self, x, mask, mu, t, spks, cond, streaming=False, return_mu_logsigma=False):
        if isinstance(self.estimator, torch.nn.Module):
            # PyTorch 分支：直接透传
            return self.estimator(x, mask, mu, t, spks, cond, streaming=streaming) \
                if not return_mu_logsigma else \
                self.estimator(x, mask, mu, t, spks, cond, streaming=streaming)
        else:
            # TRT 分支：目前只支持确定性 μ（不返回 σ）
            if return_mu_logsigma:
                raise NotImplementedError("TRT path does not support (mu, sigma) yet.")
            [estimator, stream], trt_engine = self.estimator.acquire_estimator()
            with stream:
                estimator.set_input_shape('x',    (2* x.size(0), 80, x.size(2)))
                estimator.set_input_shape('mask', (2* x.size(0), 1,  x.size(2)))
                estimator.set_input_shape('mu',   (2* x.size(0), 80, x.size(2)))
                estimator.set_input_shape('t',    (2* x.size(0),))
                estimator.set_input_shape('spks', (2* x.size(0), 80))
                estimator.set_input_shape('cond', (2* x.size(0), 80, x.size(2)))
                # 绑定地址、执行...（保持你原来的实现）
            self.estimator.release_estimator(estimator, stream)
            return x  # 占位：若要 TRT 真支持 μ，需要另外导出双头 engine

    def compute_loss(self, x1, mask, mu, spks=None, cond=None, streaming=False):
        """Computes diffusion loss

        Args:
            x1 (torch.Tensor): Target
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): target mask
                shape: (batch_size, 1, mel_timesteps)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            spks (torch.Tensor, optional): speaker embedding. Defaults to None.
                shape: (batch_size, spk_emb_dim)

        Returns:
            loss: conditional flow matching loss
            y: conditional flow
                shape: (batch_size, n_feats, mel_timesteps)
        """
        b, _, t = mu.shape

        # random timestep
        t = torch.rand([b, 1, 1], device=mu.device, dtype=mu.dtype)
        if self.t_scheduler == 'cosine':
            t = 1 - torch.cos(t * 0.5 * torch.pi)
        # sample noise p(x_0)
        z = torch.randn_like(x1)

        y = (1 - (1 - self.sigma_min) * t) * z + t * x1
        u = x1 - (1 - self.sigma_min) * z

        # during training, we randomly drop condition to trade off mode coverage and sample fidelity
        if self.training_cfg_rate > 0:
            cfg_mask = torch.rand(b, device=x1.device) > self.training_cfg_rate
            mu = mu * cfg_mask.view(-1, 1, 1)
            spks = spks * cfg_mask.view(-1, 1)
            cond = cond * cfg_mask.view(-1, 1, 1)

        # pred = self.estimator(y, mask, mu, t.squeeze(), spks, cond, streaming=streaming)
        # loss = F.mse_loss(pred * mask, u * mask, reduction="sum") / (torch.sum(mask) * u.shape[1])
        
        mu_pred, log_sigma = self.estimator(
            y, mask, mu, t.squeeze(), spks, cond, streaming=streaming
        )
        sigma = F.softplus(log_sigma) + 1e-6
        
        LOG2PI = math.log(2.0 * math.pi)
        nll = 0.5 * ( ((u - mu_pred)**2) / (sigma**2) + 2.0 * torch.log(sigma) + LOG2PI )
        loss = (nll * mask).sum() / (torch.sum(mask) * u.shape[1])
        return loss, y


class CausalConditionalCFM(ConditionalCFM):
    def __init__(self, in_channels, cfm_params, n_spks=1, spk_emb_dim=64, estimator: torch.nn.Module = None):
        super().__init__(in_channels, cfm_params, n_spks, spk_emb_dim, estimator)
        set_all_random_seed(0)
        self.rand_noise = torch.randn([1, 80, 50 * 300])

    @torch.inference_mode()
    def forward(self, mu, mask, n_timesteps, temperature=1.0,
                spks=None, cond=None, prompt_len=0, cache=None,
                streaming=False, stochastic=False, return_logprob=False):
        """
        返回: (mel, logprob或None), cache
        """
        if cache is None:
            cache = torch.zeros(1, 80, 0, 2, device=mu.device, dtype=mu.dtype)

        # 初始噪声
        z = self.rand_noise[:, :, :mu.size(2)].to(mu.device).to(mu.dtype) * temperature

        # 时间网格
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device, dtype=mu.dtype)
        if self.t_scheduler == 'cosine':
            t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)

        x, logprob = self.solve_euler(
            x=z, t_span=t_span, mu=mu, mask=mask,
            spks=spks, cond=cond, streaming=streaming,
            stochastic=stochastic, return_logprob=return_logprob
        )
        return (x, logprob), cache
