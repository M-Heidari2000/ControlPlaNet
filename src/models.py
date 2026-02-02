import torch
import torch.nn as nn
import torch.nn.init as init
from typing import Optional
from torch.distributions import MultivariateNormal


class Encoder(nn.Module):
    """
        y_t -> a_t
        embeds observations
    """

    def __init__(self, y_dim: int, hidden_dim: int, a_dim: int):
        super().__init__()

        self.mlp_layers = nn.Sequential(
            nn.Linear(y_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, a_dim),
        )

    def forward(self, y: torch.Tensor):
        return self.mlp_layers(y)


class Decoder(nn.Module):
    """
        [h_t, x_t] -> y_t
    """

    def __init__(self, rnn_hidden_dim: int, x_dim: int, hidden_dim: int, y_dim: int):
        super().__init__()

        self.mlp_layers = nn.Sequential(
            nn.Linear(rnn_hidden_dim + x_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, y_dim),
        )

    def forward(self, h: torch.Tensor, x: torch.Tensor):
        return self.mlp_layers(torch.cat([h, x], dim=1))
    

class RSSM(nn.Module):
    """
        prior: h_t = f(h_{t-1}, x_{t-1}, u_{t-1}), p(x_t) ~ N(mean, cov) mean and cov parameterized by h_t
        posterior: q(x_t) ~ N(mean, cov) mean and cov parameterized by [h_t, a_t]
    """

    def __init__(
        self,
        x_dim: int,
        u_dim: int,
        a_dim: int,
        rnn_hidden_dim: int,
        rnn_input_dim: int,
        min_var: float=1e-2,
    ):
        
        super().__init__()
        
        self.x_dim = x_dim
        self.u_dim = u_dim
        self.a_dim = a_dim
        self.rnn_hidden_dim = rnn_hidden_dim
        self.rnn_input_dim = rnn_input_dim
        self.min_var = min_var

        # RNN
        self.rnn = nn.GRUCell(input_size=self.rnn_input_dim, hidden_size=self.rnn_hidden_dim)

        # prior
        self.fc_xu = nn.Sequential(
            nn.Linear(self.x_dim + self.u_dim, self.rnn_input_dim),
            nn.ELU(),
        )
        self.prior_proj = nn.Sequential(
            nn.Linear(self.rnn_hidden_dim, self.rnn_input_dim),
            nn.ELU(),
        )
        self.prior_mean = nn.Linear(self.rnn_input_dim, self.x_dim)
        self.prior_var = nn.Sequential(
            nn.Linear(self.rnn_input_dim, self.x_dim),
            nn.Softplus(),
        )

        # posterior
        self.posterior_proj = nn.Sequential(
            nn.Linear(self.rnn_hidden_dim + self.a_dim, self.rnn_input_dim),
            nn.ELU(),
        )
        self.posterior_mean = nn.Linear(self.rnn_input_dim, self.x_dim)
        self.posterior_var = nn.Sequential(
            nn.Linear(self.rnn_input_dim, self.x_dim),
            nn.Softplus(),
        )

    def prior(self, h: torch.Tensor, x: torch.Tensor, u: torch.Tensor):
        """
            h_t = f(h_{t-1}, x_{t-1}, u_{t-1})
            p(x_t) = Normal parameterized by h_t

            inputs:
                - h: h_{t-1}
                - x: x_{t-1}
                - u: u_{t-1}
            outputs:
                - h: h_t
                - prior: p(x_t)
        """

        rnn_input = self.fc_xu(torch.cat([x, u], dim=1))
        h = self.rnn(rnn_input, h)
        prior_input = self.prior_proj(h)
        mean = self.prior_mean(prior_input)
        cov = torch.diag_embed(self.prior_var(prior_input) + self.min_var)
        prior = MultivariateNormal(loc=mean, covariance_matrix=cov)

        return h, prior

    def posterior(self, h: torch.Tensor, a: torch.Tensor):
        """
            q(x_t) parameterized by [h_t, a_t]
            inputs:
                - h: h_t
                - a: a_t    (encoded observation at time t)
            outputs:
                - posterior: q(x_t)
        """
        posterior_input = self.posterior_proj(torch.cat([h, a], dim=1))
        mean = self.posterior_mean(posterior_input)
        cov = torch.diag_embed(self.posterior_var(posterior_input) + self.min_var)
        posterior = MultivariateNormal(loc=mean, covariance_matrix=cov)

        return posterior
    
    def generate(self, h: torch.Tensor, x: torch.Tensor, u: torch.Tensor):
        """
            generates trajectory given initial state, initial rnn hidden, and a list of actions
            uses mean decoding
        """
        with torch.no_grad():
            samples = []
            rnn_hiddens = []

            if u.dim() == 2:
                u = u.unsqueeze(0)

            d, _, _ = u.shape
            state = x

            for l in range(d):
                h, prior = self.prior(h=h, x=state, u=u[l])
                state = prior.loc
                rnn_hiddens.append(h)
                samples.append(state)

            samples = torch.stack(state, dim=0)
            rnn_hiddens = torch.stack(rnn_hiddens, dim=0)

        return samples, rnn_hiddens
    
    def forward(
        self,
        h: torch.Tensor,
        u: torch.Tensor,
        a: torch.Tensor,
    ):
        """
            multi step inference of priors and posteriors

            inputs:
                - h: intial rnn hidden
                - a: a0:T
                - u: u0:T-1
            outputs:
                priors: one step priors over the states
                posteriors: posterior over the states
        """

        T, B, _ = u.shape
        device = u.device

        posterior = MultivariateNormal(
            loc=torch.zeros((B, self.x_dim), device=device),
            covariance_matrix=torch.eye(self.x_dim, device=device).repeat([B, 1, 1]),
        )
        prior = MultivariateNormal(
            loc=torch.zeros((B, self.x_dim), device=device),
            covariance_matrix=torch.eye(self.x_dim, device=device).repeat([B, 1, 1]),
        )
        rnn_hiddens = [h]

        posteriors = [posterior]
        priors = [prior]

        for t in range(T):
            h, prior = self.prior(h=h, x=posterior.rsample(), u=u[t])
            posterior = self.posterior(h=h, a=a[t])
            rnn_hiddens.append(h)
            priors.append(prior)
            posteriors.append(posterior)

        return priors, posteriors, rnn_hiddens


class CostModel(nn.Module):

    def __init__(
        self,
        x_dim: int,
        u_dim: int,
        rnn_hidden_dim: int,
        hidden_dim: Optional[int]=64,
    ):
        super().__init__()

        self.x_dim = x_dim
        self.u_dim = u_dim
        self.rnn_hidden_dim = rnn_hidden_dim

        self.mlp_layers = nn.Sequential(
            nn.Linear(self.rnn_hidden_dim + self.x_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),
        )

    def forward(self, h: torch.Tensor, x:torch.Tensor):
        cost = self.mlp_layers(torch.cat([x, h], dim=1))
        return cost