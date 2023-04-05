namespace SpikepropSharp.Components
{
    internal sealed class Synapse
    {
        public Neuron NeuronPre { get; }
        public double Weight { get; set; }
        public double Delay { get; set; }
        public double WeightDelta { get; set; }

        public Synapse(Neuron neuronPre, double weight, double delay)
        {
            NeuronPre = neuronPre;
            Weight = weight;
            Delay = delay;
        }
    }
}
