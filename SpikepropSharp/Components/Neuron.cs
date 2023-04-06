namespace SpikepropSharp.Components
{
    public sealed class Neuron
    {
        private const double TAU_M = 4;
        private const double TAU_S = 2;
        private const double TAU_R = 4;
        private const double THRESHOLD = 1;

        public string Name = "";
        public double Clamped { get; set; }
        public List<Synapse> SynapsesIn = new();
        public List<Neuron> NeuronsPost = new();
        public List<double> Spikes = new(); // Eq (1)

        public Neuron(string name = "neuron")
        {
            Name = name;
        }

        public static double Epsilon(double s) => // Eq (4)
            s < 0 ? 0 : Math.Exp(-s / TAU_M) - Math.Exp(-s / TAU_S);

        public static double Eta(double s) => // Eq (5)
            s < 0 ? 0 : -Math.Exp(-s / TAU_R);

        public double Epsilond(double s) =>
            s < 0 ? 0 : -Math.Exp(-s / TAU_M) / TAU_M + Math.Exp(-s / TAU_S) / TAU_S;

        public double Etad(double s) => 
            s < 0 ? 0 : Math.Exp(-s / TAU_R) / TAU_R;

        public void Fire(double time) =>
            Spikes.Add(time);

        public void Forward(double time) // Eq (2)
        {
            if (ComputeU(time) > THRESHOLD)
            {
                Fire(time);
            }
        }

        public double ComputeU(double time) // Eq (3)
        {
            double u = 0;
            foreach (Synapse synapseIn in SynapsesIn)
            {
                foreach (double spikePre in synapseIn.NeuronPre.Spikes)
                {
                    u += synapseIn.Weight * Epsilon(time - spikePre - synapseIn.Delay);
                }
            }

            foreach (double spike in Spikes)
            {
                u += Eta(time - spike);
            }

            return u;
        }

        public void ComputeDeltaWeights(double learningRate) // Eq (9)
        {
            foreach (Synapse synapse in SynapsesIn)
            {
                foreach (double spikeThis in Spikes)
                {
                    synapse.WeightDelta -= learningRate * ComputeDeDt(spikeThis) * ComputeDtDw(synapse, spikeThis);
                }
            }
        }

        public double ComputeDtDw(Synapse synapse, double spikeThis) // Eq (10)
        {
            return -ComputeDuDw(synapse, spikeThis) / ComputeDuDt(spikeThis);
        }

        public double ComputeDuDw(Synapse synapse, double spikeThis) // Eq (11)
        {
            double duDw = 0.0;
            foreach (double spikePre in synapse.NeuronPre.Spikes)
            {
                duDw += Epsilon(spikeThis - spikePre - synapse.Delay);
            }

            foreach (double spikeRef in Spikes)
            {
                if (spikeRef < spikeThis)
                {
                    duDw += -Etad(spikeThis - spikeRef) * ComputeDtDw(synapse, spikeRef);
                }
            }

            return duDw;
        }
        public double ComputeDuDt(double spikeThis) // Eq (12)
        {
            double duDt = 0.0;
            foreach (Synapse synapse in SynapsesIn)
            {
                foreach (double spikePre in synapse.NeuronPre.Spikes)
                {
                    duDt += synapse.Weight * Epsilond(spikeThis - spikePre - synapse.Delay);
                }
            }

            foreach (double spikeRef in Spikes)
            {
                if (spikeRef < spikeThis)
                {
                    duDt += Etad(spikeThis - spikeRef);
                }
            }

            if (duDt < 0.1) // handling discontinuity circumstance 1 Sec 3.2
            {
                duDt = 0.1;
            }
            return duDt;
        }

        public double ComputeDeDt(double spikeThis) // Eq (13)
        {
            if (Clamped > 0.0)
            {
                if (spikeThis == Spikes[0])
                {
                    return spikeThis - Clamped;
                }
            }

            double deDt = 0.0;
            foreach (Neuron neuronPost in NeuronsPost)
            {
                foreach (double spikePost in neuronPost.Spikes)
                {
                    if (spikePost > spikeThis)
                    {
                        deDt += neuronPost.ComputeDeDt(spikePost) * ComputeDposttDt(spikeThis, neuronPost, spikePost);
                    }
                }
            }
            return deDt;
        }

        public double ComputeDposttDt(double spikeThis, Neuron neuronPost, double spikePost) // Eq (14)
        {
            return -ComputeDpostuDt(spikeThis, neuronPost, spikePost) / neuronPost.ComputeDuDt(spikePost);
        }

        public double ComputeDpostuDt(double spikeThis, Neuron neuronPost, double spikePost) // Eq (15)
        {
            double dpostuDt = 0.0;
            foreach (Synapse synapse in neuronPost.SynapsesIn)
            {
                if (synapse.NeuronPre == this)
                {
                    dpostuDt -= synapse.Weight * Epsilond(spikePost - spikeThis - synapse.Delay);
                }
            }

            foreach (double spikeRef in neuronPost.Spikes)
            {
                if (spikeRef < spikePost)
                {
                    dpostuDt -= Etad(spikePost - spikeRef) * ComputeDposttDt(spikeThis, neuronPost, spikeRef);
                }
            }
            return dpostuDt;
        }

    }
}
