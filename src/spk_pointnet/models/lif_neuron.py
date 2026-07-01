"""
LIF Neuron implementation wrapping SpikingJelly's LIFNode.

Membrane potential update:
    u[t] = tau * u[t-1] + x[t]
Spike:
    s[t] = Heaviside(u[t] - V_th)
Reset (hard):
    u[t] = u[t] * (1 - s[t])

Surrogate gradient: 1 / (1 + |u - V_th| * k)^2  (sigmoid-like, k=4)
We use SpikingJelly's ATan surrogate which closely approximates this.

Usage:
    neuron = make_lif_neuron(tau=0.25, V_th=0.5)
    # For multi-step mode, call functional.set_step_mode(model, 'm') on the
    # parent model after construction.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, surrogate, functional


@dataclass
class LIFNeuronConfig:
    """Configuration for a LIF neuron."""
    tau: float = 4.0          # SpikingJelly tau is 1/(1-decay), so tau=4 => decay=0.75
    # NOTE: In SpikingJelly, tau is the membrane time constant such that
    # decay = 1 - 1/tau.  The paper uses u[t] = tau*u[t-1] + x[t] with tau=0.25
    # which is a *decay factor* (not time constant).  We keep the paper convention
    # and pass tau directly as the decay coefficient by using a custom approach:
    # SpikingJelly LIFNode: v[t+1] = v[t] - v[t]/tau + x[t]
    #                              = v[t]*(1 - 1/tau) + x[t]
    # Paper:               u[t]   = tau_paper * u[t-1] + x[t]   (tau_paper=0.25)
    # So decay = tau_paper = 0.25 => 1 - 1/tau_sj = 0.25 => tau_sj = 4/3
    # We handle this mapping inside make_lif_neuron.
    V_th: float = 0.5
    step_mode: str = 's'      # 's' single-step; use functional.set_step_mode for 'm'


def make_lif_neuron(
    tau: float = 0.25,
    V_th: float = 0.5,
    step_mode: str = 's',
    detach_reset: bool = True,
) -> neuron.LIFNode:
    """
    Create a SpikingJelly LIFNode that matches the paper's formulation:
        u[t] = tau * u[t-1] + x[t]    (tau = membrane decay factor, 0 < tau < 1)

    SpikingJelly's LIFNode implements:
        v[t] = v[t-1] - v[t-1] / tau_sj + x[t]
             = v[t-1] * (1 - 1/tau_sj) + x[t]

    Matching: decay_factor = tau  =>  1 - 1/tau_sj = tau
              tau_sj = 1 / (1 - tau)

    Args:
        tau:        Membrane decay factor in [0, 1) from the paper (default 0.25).
        V_th:       Spike threshold (default 0.5).
        step_mode:  'single' or 'multi'; set at model level via functional.set_step_mode.
        detach_reset: Detach reset signal from computation graph (recommended True).

    Returns:
        Configured LIFNode instance.
    """
    # Convert paper tau (decay factor) to SpikingJelly tau (time constant)
    if tau <= 0 or tau >= 1:
        raise ValueError(f"tau must be in (0, 1), got {tau}")
    tau_sj = 1.0 / (1.0 - tau)   # tau_paper=0.25 => tau_sj = 4/3 ≈ 1.333

    lif = neuron.LIFNode(
        tau=tau_sj,
        v_threshold=V_th,
        v_reset=0.0,           # hard reset to 0
        surrogate_function=surrogate.ATan(),   # differentiable surrogate gradient
        detach_reset=detach_reset,
        step_mode=step_mode,
    )
    return lif


class SigmoidSurrogate(surrogate.SurrogateFunctionBase):
    """
    Custom sigmoid-like surrogate gradient:
        forward:  Heaviside(u - V_th)
        backward: 1 / (1 + |u - V_th| * k)^2

    This matches the surrogate described in the paper.
    """

    def __init__(self, k: float = 4.0, spiking: bool = True):
        super().__init__(alpha=k, spiking=spiking)
        self.k = k

    @staticmethod
    def spiking_function(x, alpha):
        return (x >= 0).float()

    @staticmethod
    def primitive_function(x, alpha):
        # Not used directly but required by base class
        return torch.sigmoid(alpha * x)

    @staticmethod
    def backward(grad_output, x, alpha):
        # Surrogate gradient: 1/(1 + |x|*alpha)^2
        return grad_output / (1.0 + (x.abs() * alpha)) ** 2, None

    def forward(self, x):
        if self.spiking:
            return self.spiking_function(x, self.alpha)
        else:
            return self.primitive_function(x, self.alpha)


def make_lif_neuron_custom_surrogate(
    tau: float = 0.25,
    V_th: float = 0.5,
    k: float = 4.0,
    step_mode: str = 's',
    detach_reset: bool = True,
) -> neuron.LIFNode:
    """
    Create a LIFNode with the custom sigmoid surrogate gradient from the paper.

    Args:
        tau:   Membrane decay factor in [0, 1).
        V_th:  Spike threshold.
        k:     Surrogate sharpness parameter.
        step_mode: Step mode for SpikingJelly.
        detach_reset: Whether to detach reset signal.

    Returns:
        Configured LIFNode with custom surrogate.
    """
    if tau <= 0 or tau >= 1:
        raise ValueError(f"tau must be in (0, 1), got {tau}")
    tau_sj = 1.0 / (1.0 - tau)

    lif = neuron.LIFNode(
        tau=tau_sj,
        v_threshold=V_th,
        v_reset=0.0,
        surrogate_function=SigmoidSurrogate(k=k),
        detach_reset=detach_reset,
        step_mode=step_mode,
    )
    return lif
