import torch
import wandb
import einops
import torch.nn as nn
from tqdm import tqdm
from omegaconf.dictconfig import DictConfig
from torch.distributions.kl import kl_divergence
from .memory import ReplayBuffer
from torch.nn.utils import clip_grad_norm_
from .models import (
    Encoder,
    Decoder,
    RSSM,
    CostModel,
)


def train_backbone(
    config: DictConfig,
    train_buffer: ReplayBuffer,
    test_buffer: ReplayBuffer,
):

    # define models and optimizer
    device = "cuda" if (torch.cuda.is_available() and not config.disable_gpu) else "cpu"

    encoder = Encoder(
        y_dim=train_buffer.y_dim,
        hidden_dim=config.hidden_dim,
        a_dim=config.a_dim
    ).to(device)

    decoder = Decoder(
        rnn_hidden_dim=config.rnn_hidden_dim,
        x_dim=config.x_dim,
        hidden_dim=config.hidden_dim,
        y_dim=train_buffer.y_dim
    )

    rssm = RSSM(
        x_dim=config.x_dim,
        u_dim=train_buffer.u_dim,
        a_dim=config.a_dim,
        rnn_hidden_dim=config.rnn_hidden_dim,
        rnn_input_dim=config.rnn_input_dim,
        min_var=config.min_var,
    )

    wandb.watch([encoder, rssm, decoder], log="all", log_freq=10)

    all_params = (
        list(encoder.parameters()) +
        list(decoder.parameters()) + 
        list(rssm.parameters())
    )

    optimizer = torch.optim.Adam(all_params, lr=config.lr, eps=config.eps, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=config.num_updates
    )

    # train and test loop
    for update in tqdm(range(config.num_updates)):
        
        # train
        encoder.train()
        decoder.train()
        rssm.train()

        y, u, _, _ = train_buffer.sample(
            batch_size=config.batch_size,
            chunk_length=config.chunk_length,
        )

        # convert to tensor, transform to device, reshape to time-first
        y = torch.as_tensor(y, device=device)
        a = encoder(einops.rearrange(y, "b l y -> (b l) y"))
        a = einops.rearrange(a, "(b l) a -> l b a", b=config.batch_size)
        u = torch.as_tensor(u, device=device)
        u = einops.rearrange(u, "b l u -> l b u")

        priors, posteriors, rnn_hiddens = rssm(a=a, u=u)
        # x0:T-1
        posterior_samples = torch.stack([p.rsample() for p in posteriors], dim=0)
        # reconstruction loss
        y_recon = decoder(
            x=einops.rearrange(posterior_samples, "l b x -> (l b) x"),
            h=torch.cat(rnn_hiddens, dim=0),
        )
        y_true = einops.rearrange(y, "l b y -> (l b) y")
        reconstruction_loss = nn.MSELoss()(y_recon, y_true)
        # KL loss
        kl_loss = 0.0
        for t in range(config.chunk_length):
            kl_loss += kl_divergence(posteriors[t], priors[t]).clamp(min=config.free_nats).mean()
        kl_loss = kl_loss / config.chunk_length

        total_loss = reconstruction_loss + config.kl_beta * kl_loss

        optimizer.zero_grad()
        total_loss.backward()

        clip_grad_norm_(all_params, config.clip_grad_norm)
        optimizer.step()
        scheduler.step()

        wandb.log({
            "train/y reconstruction loss": reconstruction_loss.item(),
            "train/kl loss": kl_loss.item(),
            "global_step": update,
        })
            
        if update % config.test_interval == 0:
            # test
            with torch.no_grad():              
                # train
                encoder.eval()
                decoder.eval()
                rssm.eval()

                y, u, _, _ = test_buffer.sample(
                    batch_size=config.batch_size,
                    chunk_length=config.chunk_length,
                )

                # convert to tensor, transform to device, reshape to time-first
                y = torch.as_tensor(y, device=device)
                a = encoder(einops.rearrange(y, "b l y -> (b l) y"))
                a = einops.rearrange(a, "(b l) a -> l b a", b=config.batch_size)
                u = torch.as_tensor(u, device=device)
                u = einops.rearrange(u, "b l u -> l b u")

                priors, posteriors, rnn_hiddens = rssm(a=a, u=u)
                # x0:T-1
                posterior_samples = torch.stack([p.rsample() for p in posteriors], dim=0)
                # reconstruction loss
                y_recon = decoder(
                    x=einops.rearrange(posterior_samples, "l b x -> (l b) x"),
                    h=torch.cat(rnn_hiddens, dim=0)
                )
                y_true = einops.rearrange(y, "l b y -> (l b) y")
                reconstruction_loss = nn.MSELoss()(y_recon, y_true)
                # KL loss
                kl_loss = 0.0
                for t in range(config.chunk_length):
                    kl_loss += kl_divergence(posteriors[t], priors[t]).clamp(min=config.free_nats).mean()
                kl_loss = kl_loss / config.chunk_length

                total_loss = reconstruction_loss + config.kl_beta * kl_loss

                wandb.log({
                    "test/y reconstruction loss": reconstruction_loss.item(),
                    "test/kl loss": kl_loss.item(),
                    "global_step": update,
                })
                
    return encoder, decoder, rssm


def train_cost(
    config: DictConfig,
    encoder: Encoder,
    rssm: RSSM,
    train_buffer: ReplayBuffer,
    test_buffer: ReplayBuffer,
):
    device = "cuda" if (torch.cuda.is_available() and not config.disable_gpu) else "cpu"

    cost_model = CostModel(
        x_dim=rssm.x_dim,
        rnn_hidden_dim=rssm.rnn_hidden_dim,
    ).to(device)

    # freeze the encoder and rssm
    for p in encoder.parameters():
        p.requires_grad = False
    
    for p in rssm.parameters():
        p.requires_grad = False

    encoder.eval()
    rssm.eval()

    wandb.watch([cost_model], log="all", log_freq=10)
    all_params = list(cost_model.parameters())
    optimizer = torch.optim.Adam(all_params, lr=config.lr, eps=config.eps, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=config.num_updates
    )

    # train and test loop
    for update in tqdm(range(config.num_updates)):    
        # train
        cost_model.train()

        y, u, c, _ = train_buffer.sample(
            batch_size=config.batch_size,
            chunk_length=config.chunk_length,
        )

        # convert to tensor, transform to device, reshape to time-first
        y = torch.as_tensor(y, device=device)
        a = encoder(einops.rearrange(y, "b l y -> (b l) y"))
        a = einops.rearrange(a, "(b l) a -> l b a", b=config.batch_size)
        u = torch.as_tensor(u, device=device)
        u = einops.rearrange(u, "b l u -> l b u")
        c = torch.as_tensor(c, device=device)
        
        _, posteriors, rnn_hiddens = rssm(a=a, u=u)
        # x0:T-1
        posterior_samples = torch.stack([p.rsample() for p in posteriors], dim=0)
        # compute cost loss
        cost_loss = 0.0
        for t in range(config.chunk_length):
            cost_loss += nn.MSELoss()(cost_model(x=posterior_samples[t], h=rnn_hiddens[t]), c[t])
        cost_loss = cost_loss / config.chunk_length

        optimizer.zero_grad()
        cost_loss.backward()

        clip_grad_norm_(all_params, config.clip_grad_norm)
        optimizer.step()
        scheduler.step()

        wandb.log({
            "train/cost loss": cost_loss.item(),
            "global_step": update,
        })
            
        if update % config.test_interval == 0:
            # test
            with torch.no_grad():
                cost_model.eval()

                y, u, c, _ = test_buffer.sample(
                    batch_size=config.batch_size,
                    chunk_length=config.chunk_length,
                )

                # convert to tensor, transform to device, reshape to time-first
                y = torch.as_tensor(y, device=device)
                a = encoder(einops.rearrange(y, "b l y -> (b l) y"))
                a = einops.rearrange(a, "(b l) a -> l b a", b=config.batch_size)
                u = torch.as_tensor(u, device=device)
                u = einops.rearrange(u, "b l u -> l b u")
                c = torch.as_tensor(c, device=device)
                
                _, posteriors, rnn_hiddens = rssm(a=a, u=u)
                # x0:T-1
                posterior_samples = torch.stack([p.rsample() for p in posteriors], dim=0)
                # compute cost loss
                cost_loss = 0.0
                for t in range(config.chunk_length):
                    cost_loss += nn.MSELoss()(cost_model(x=posterior_samples[t], h=rnn_hiddens[t]), c[t])
                cost_loss = cost_loss / config.chunk_length

                wandb.log({
                    "test/cost loss": cost_loss.item(),
                    "global_step": update,
                })
                
    return cost_model