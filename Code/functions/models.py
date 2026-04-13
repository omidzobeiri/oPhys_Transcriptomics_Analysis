"""Neural network models (PyTorch) for RNN modeling."""

import torch
import torch.nn as nn


class DalesRNN(nn.Module):
    """Continuous-time RNN with Dale's law constraint.

    Excitatory units have non-negative outgoing weights.
    Inhibitory units have non-positive outgoing weights.

    Architecture:
      - Input: (orientation, contrast, TF, running_speed) -> 4 dims
      - Recurrent: N_E excitatory + N_I inhibitory units
      - Output: 8-way orientation classification
    """

    def __init__(self, n_exc=200, n_pvalb=20, n_sst=10, n_vip=10, n_lamp5=5,
                 n_input=4, n_output=8, dt=0.1, tau=1.0, noise_std=0.05):
        super().__init__()

        self.n_exc = n_exc
        self.n_inh = n_pvalb + n_sst + n_vip + n_lamp5
        self.n_total = n_exc + self.n_inh
        self.n_input = n_input
        self.n_output = n_output
        self.dt = dt
        self.tau = tau
        self.noise_std = noise_std

        self.unit_types = (
            ['L2/3 IT'] * (n_exc // 2) +
            ['L4/5 IT'] * (n_exc - n_exc // 2) +
            ['Pvalb'] * n_pvalb +
            ['Sst'] * n_sst +
            ['Vip'] * n_vip +
            ['Lamp5'] * n_lamp5
        )

        dale_mask = torch.ones(self.n_total)
        dale_mask[n_exc:] = -1
        self.register_buffer('dale_mask', dale_mask)

        self.W_rec_raw = nn.Parameter(torch.abs(torch.randn(self.n_total, self.n_total) * 0.1))
        self.W_in = nn.Parameter(torch.randn(self.n_total, n_input) * 0.3)
        self.W_out = nn.Parameter(torch.randn(n_output, self.n_total) * 0.3)

        self.bias = nn.Parameter(torch.zeros(self.n_total))
        self.b_out = nn.Parameter(torch.zeros(n_output))

    def get_effective_W(self):
        """Apply Dale's law: W_eff[i, j] = |W_raw[i, j]| * dale_mask[j]"""
        return torch.abs(self.W_rec_raw) * self.dale_mask.unsqueeze(0)

    def forward(self, inputs, n_steps=20):
        """
        inputs: (batch, n_input) -- stimulus parameters for one trial
        Returns: outputs (batch, n_steps, n_output), rates (batch, n_steps, n_total)
        """
        batch_size = inputs.shape[0]
        h = torch.zeros(batch_size, self.n_total, device=inputs.device)
        W_eff = self.get_effective_W()

        all_rates = []
        all_outputs = []

        for t in range(n_steps):
            noise = self.noise_std * torch.randn_like(h) if self.training else 0

            r = torch.relu(h)
            dh = (-h + r @ W_eff.T + inputs @ self.W_in.T + self.bias + noise) * (self.dt / self.tau)
            h = h + dh

            all_rates.append(r)
            output = h @ self.W_out.T + self.b_out
            all_outputs.append(output)

        rates = torch.stack(all_rates, dim=1)
        outputs = torch.stack(all_outputs, dim=1)
        return outputs, rates


class PredictiveRNN(nn.Module):
    """RNN that predicts population DF/F from stimulus + running input."""

    def __init__(self, n_input=5, n_hidden=256, n_output=100, n_layers=1):
        super().__init__()
        self.n_hidden = n_hidden
        self.rnn = nn.GRU(n_input, n_hidden, n_layers, batch_first=True)
        self.readout = nn.Linear(n_hidden, n_output)

    def forward(self, x, h0=None):
        """x: (batch, seq_len, n_input) -> pred: (batch, seq_len, n_output)"""
        out, hn = self.rnn(x, h0)
        pred = self.readout(out)
        return pred, out


class TemporalRNN(nn.Module):
    """RNN that predicts population DF/F at every 100ms within a trial."""

    def __init__(self, n_input=6, n_hidden=256, n_output=80, n_layers=2, dropout=0.1):
        super().__init__()
        self.rnn = nn.GRU(n_input, n_hidden, n_layers, batch_first=True, dropout=dropout)
        self.readout = nn.Sequential(
            nn.Linear(n_hidden, n_hidden // 2),
            nn.ReLU(),
            nn.Linear(n_hidden // 2, n_output),
        )

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.readout(out), out
