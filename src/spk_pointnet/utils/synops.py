"""
Synaptic Operations (SynOps) counter for energy efficiency estimation.

Uses PyTorch forward hooks to monitor LIF neuron outputs during a forward pass.
Computes the theoretical energy consumption as:

    SNN energy (AC):  SynOps * 0.9 pJ
    ANN energy (MAC): equivalent_FLOPs / 2 * 4.6 pJ

Where:
    SynOps = sum over layers of (spike_rate * fan_in * num_output_neurons)
    spike_rate = mean fraction of spikes fired per neuron per timestep

The energy ratio ANN/SNN quantifies the efficiency advantage.

Usage:
    with SynOpsCounter(model) as counter:
        _ = model(encoded_input)
    report = counter.get_report()
    print(report)
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple

from spikingjelly.activation_based import neuron


# Energy constants
AC_ENERGY_PJ = 0.9    # pJ per accumulate operation (SNN)
MAC_ENERGY_PJ = 4.6   # pJ per multiply-accumulate operation (ANN)


class SynOpsCounter:
    """
    Context manager for counting synaptic operations in spiking networks.

    Attaches forward hooks to all LIF neurons in the model.
    Accumulates spike tensors during forward pass.
    Computes SynOps and energy estimates on exit.

    Args:
        model:    SpikingPointNet (or any nn.Module with LIFNode layers).
        verbose:  Print hook attachment info.

    Example:
        >>> with SynOpsCounter(model) as counter:
        ...     logits, _, _ = model(encoded)
        >>> report = counter.get_report()
        >>> print(f"SynOps: {report['total_synops']:.2e}")
        >>> print(f"AC energy: {report['ac_energy_pj']:.2f} pJ")
        >>> print(f"Energy ratio (ANN/SNN): {report['energy_ratio']:.1f}x")
    """

    def __init__(self, model: nn.Module, verbose: bool = False):
        self.model = model
        self.verbose = verbose
        self._hooks: List = []
        self._layer_data: List[Dict] = []
        self._layer_names: List[str] = []

    def __enter__(self):
        self._layer_data = []
        self._layer_names = []
        self._hooks = []
        self._attach_hooks()
        return self

    def __exit__(self, *args):
        self._remove_hooks()

    def _attach_hooks(self):
        """Attach forward hooks to all LIF neuron layers."""
        for name, module in self.model.named_modules():
            if isinstance(module, neuron.LIFNode):
                # Capture layer name and pre-computed fan_in via closure
                layer_entry = {
                    'name': name,
                    'spikes': [],        # accumulated spike tensors
                    'input_shape': None, # shape of last input
                    'output_shape': None,
                }
                self._layer_data.append(layer_entry)

                def make_hook(entry):
                    def hook(module, inp, output):
                        # output is the spike tensor (0/1 valued)
                        # Shape varies: [B, C], [B, N, C], etc.
                        spk = output.detach().float()
                        entry['spikes'].append(spk)
                        entry['input_shape'] = inp[0].shape if inp else None
                        entry['output_shape'] = output.shape
                    return hook

                h = module.register_forward_hook(make_hook(layer_entry))
                self._hooks.append(h)

                if self.verbose:
                    print(f"  [SynOps] Hooked: {name}")

    def _remove_hooks(self):
        """Remove all registered hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def get_report(self) -> Dict:
        """
        Compute SynOps and energy estimates from accumulated spike data.

        SynOps for layer L = spike_rate_L * fan_in_L * num_outputs_L

        where:
          spike_rate = mean(spikes) over batch and spatial dims
          fan_in = input feature dimension (C_in)
          num_outputs = output feature dimension (C_out)

        Returns:
            Dict with keys:
              'layer_synops': list of per-layer SynOps
              'total_synops': sum of all layer SynOps
              'total_spikes': total spike count across all layers
              'mean_spike_rate': mean firing rate across all LIF neurons
              'ac_energy_pj': total AC energy in picojoules (SNN)
              'mac_energy_pj': estimated MAC energy for equivalent ANN
              'ac_energy_j': total AC energy in joules
              'mac_energy_j': estimated MAC energy in joules
              'energy_ratio': mac_energy / ac_energy (how much more efficient SNN is)
              'layer_details': list of per-layer detail dicts
        """
        layer_synops = []
        layer_details = []
        total_spikes = 0
        total_spike_rate_sum = 0.0
        total_synops = 0.0

        # Estimate MAC count for ANN equivalent (for ratio computation)
        # We approximate: each LIF layer corresponds to a linear transform.
        # MACs per sample = fan_in * fan_out
        total_mac = 0.0

        for entry in self._layer_data:
            if not entry['spikes']:
                continue

            # Stack all spike tensors collected during this forward pass
            # Each element might have shape [B, C] or [B, N, C]
            all_spikes = torch.cat(
                [s.reshape(-1) for s in entry['spikes']]
            )

            spike_count = all_spikes.sum().item()
            spike_rate = all_spikes.mean().item()
            total_spikes += spike_count
            total_spike_rate_sum += spike_rate

            # Estimate fan_in and fan_out from output shape
            # For LIF after Linear [B, C_out]: fan_in estimated from input shape
            out_shape = entry['output_shape']
            in_shape = entry['input_shape']

            if out_shape is not None and len(out_shape) >= 2:
                fan_out = out_shape[-1]
            else:
                fan_out = 1

            if in_shape is not None and len(in_shape) >= 2:
                fan_in = in_shape[-1]
            else:
                fan_in = fan_out  # fallback

            # SynOps = spike_rate * fan_in * fan_out
            synops = spike_rate * fan_in * fan_out
            layer_synops.append(synops)
            total_synops += synops

            # ANN MAC estimate for this layer
            layer_mac = fan_in * fan_out
            total_mac += layer_mac

            layer_details.append({
                'name': entry['name'],
                'spike_rate': spike_rate,
                'total_spikes': spike_count,
                'fan_in': fan_in,
                'fan_out': fan_out,
                'synops': synops,
            })

        # Energy estimates
        ac_energy_pj = total_synops * AC_ENERGY_PJ
        mac_energy_pj = total_mac * MAC_ENERGY_PJ  # ANN equivalent

        ac_energy_j = ac_energy_pj * 1e-12
        mac_energy_j = mac_energy_pj * 1e-12

        energy_ratio = mac_energy_pj / ac_energy_pj if ac_energy_pj > 0 else float('inf')

        num_layers = len(layer_details)
        mean_spike_rate = total_spike_rate_sum / num_layers if num_layers > 0 else 0.0

        return {
            'layer_synops': layer_synops,
            'total_synops': total_synops,
            'total_spikes': total_spikes,
            'mean_spike_rate': mean_spike_rate,
            'ac_energy_pj': ac_energy_pj,
            'mac_energy_pj': mac_energy_pj,
            'ac_energy_j': ac_energy_j,
            'mac_energy_j': mac_energy_j,
            'energy_ratio': energy_ratio,
            'layer_details': layer_details,
        }

    def print_report(self):
        """Print a formatted SynOps report."""
        report = self.get_report()
        print("\n" + "="*60)
        print("  SynOps Energy Report")
        print("="*60)
        print(f"  Total SynOps:          {report['total_synops']:.3e}")
        print(f"  Mean spike rate:       {report['mean_spike_rate']:.4f}")
        print(f"  Total spikes:          {report['total_spikes']:.3e}")
        print(f"  SNN AC energy:         {report['ac_energy_pj']:.2f} pJ")
        print(f"  ANN MAC energy (est.): {report['mac_energy_pj']:.2f} pJ")
        print(f"  Energy ratio (ANN/SNN): {report['energy_ratio']:.1f}x")
        print("-"*60)
        print(f"  {'Layer':<40}  {'SpikeRate':>10}  {'SynOps':>12}")
        print("-"*60)
        for d in report['layer_details']:
            print(
                f"  {d['name'][:40]:<40}  "
                f"{d['spike_rate']:>10.4f}  "
                f"{d['synops']:>12.3e}"
            )
        print("="*60 + "\n")
        return report
