import torch
import wandb
import einops
import torch.nn as nn
from tqdm import tqdm
from omegaconf.dictconfig import DictConfig
from torch.distributions.kl import kl_divergence
from .memory import ReplayBuffer
from .utils import detach_mvn
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
        a = encoder(y)
        a = torch.cat((
            torch.zeros((1, config.batch_size, config.a_dim), device=device),
            einops.rearrange(a, "b l a -> l b a")
        ), dim=0)
        y = torch.cat((
            torch.zeros((1, config.batch_size, train_buffer.y_dim), device=device),
            einops.rearrange(y, "b l y -> l b y")
        ), dim=0)
        u = torch.as_tensor(u, device=device)
        u = einops.rearrange(u, "b l u -> l b u")

        # Initial RNN hidden
        rnn_hidden = torch.zeros((config.batch_size, config.rnn_hidden_dim), device=device)
        priors, posteriors, rnn_hiddens = rssm(h=rnn_hidden, a=a, u=u)
        # x0:T
        posterior_samples = torch.stack([p.rsample() for p in posteriors], dim=0)
        # reconstruction loss
        y_recon = decoder(
            x=einops.rearrange(posterior_samples[1:], "l b x -> (l b) x"),
            h=torch.cat(rnn_hiddens[1:], dim=0),
        )
        y_true = einops.rearrange(y[1:], "l b y -> (l b) y")
        reconstruction_loss = nn.MSELoss()(y_recon, y_true)
        # KL loss
        kl_loss = 0.0
        for t in range(1, config.chunk_length+1):
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
                a = encoder(y)
                a = torch.cat((
                    torch.zeros((1, config.batch_size, config.a_dim), device=device),
                    einops.rearrange(a, "b l a -> l b a")
                ), dim=0)
                y = torch.cat((
                    torch.zeros((1, config.batch_size, test_buffer.y_dim), device=device),
                    einops.rearrange(y, "b l y -> l b y")
                ), dim=0)
                u = torch.as_tensor(u, device=device)
                u = einops.rearrange(u, "b l u -> l b u")

                # Initial RNN hidden
                rnn_hidden = torch.zeros((config.batch_size, config.rnn_hidden_dim), device=device)
                priors, posteriors, rnn_hiddens = rssm(h=rnn_hidden, a=a, u=u)
                # x0:T
                posterior_samples = torch.stack([p.rsample() for p in posteriors], dim=0)
                # reconstruction loss
                y_recon = decoder(
                    x=einops.rearrange(posterior_samples[1:], "l b x -> (l b) x"),
                    h=torch.cat(rnn_hiddens[1:], dim=0)
                )
                y_true = einops.rearrange(y[1:], "l b y -> (l b) y")
                reconstruction_loss = nn.MSELoss()(y_recon, y_true)
                # KL loss
                kl_loss = 0.0
                for t in range(1, config.chunk_length+1):
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
        u_dim=rssm.u_dim,
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
        a = encoder(y)
        a = torch.cat((
            torch.zeros((1, config.batch_size, rssm.a_dim), device=device),
            einops.rearrange(a, "b l a -> l b a")
        ), dim=0)
        y = torch.cat((
            torch.zeros((1, config.batch_size, train_buffer.y_dim), device=device),
            einops.rearrange(y, "b l y -> l b y")
        ), dim=0)
        u = torch.as_tensor(u, device=device)
        u = einops.rearrange(u, "b l u -> l b u")
        c = torch.as_tensor(c, device=device)
        c = torch.cat((
            torch.zeros((1, config.batch_size, 1), device=device),
            einops.rearrange(c, "b l c -> l b c")
        ), dim=0)

        # Initial RNN hidden
        rnn_hidden = torch.zeros((config.batch_size, config.rnn_hidden_dim), device=device)
        _, posteriors, rnn_hiddens = rssm(h=rnn_hidden, a=a, u=u)
        # x0:T
        posterior_samples = torch.stack([p.rsample() for p in posteriors], dim=0)
        # compute cost loss
        cost_loss = 0.0
        for t in range(1, config.chunk_length+1):
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
                a = encoder(y)
                a = torch.cat((
                    torch.zeros((1, config.batch_size, rssm.a_dim), device=device),
                    einops.rearrange(a, "b l a -> l b a")
                ), dim=0)
                y = torch.cat((
                    torch.zeros((1, config.batch_size, test_buffer.y_dim), device=device),
                    einops.rearrange(y, "b l y -> l b y")
                ), dim=0)
                u = torch.as_tensor(u, device=device)
                u = einops.rearrange(u, "b l u -> l b u")
                c = torch.as_tensor(c, device=device)
                c = torch.cat((
                    torch.zeros((1, config.batch_size, 1), device=device),
                    einops.rearrange(c, "b l c -> l b c")
                ), dim=0)

                # Initial RNN hidden
                rnn_hidden = torch.zeros((config.batch_size, config.rnn_hidden_dim), device=device)
                _, posteriors, rnn_hiddens = rssm(h=rnn_hidden, a=a, u=u)
                # x0:T
                posterior_samples = torch.stack([p.rsample() for p in posteriors], dim=0)
                # compute cost loss
                cost_loss = 0.0
                for t in range(1, config.chunk_length+1):
                    cost_loss += nn.MSELoss()(cost_model(x=posterior_samples[t], h=rnn_hiddens[t]), c[t])
                cost_loss = cost_loss / config.chunk_length

                wandb.log({
                    "test/cost loss": cost_loss.item(),
                    "global_step": update,
                })
                
    return cost_model