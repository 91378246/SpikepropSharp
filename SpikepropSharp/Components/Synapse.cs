﻿namespace SpikepropSharp.Components;

public sealed class Synapse
{
    public Neuron NeuronPre { get; }
    public double Weight { get; set; }
    public double Delay { get; set; }
    public double WeightDelta { get; set; }

    public Synapse(Neuron neuronPre, double delay)
    {
        NeuronPre = neuronPre;
        Delay = delay;
    }
}
