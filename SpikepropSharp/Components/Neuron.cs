namespace SpikepropSharp.Components
{
    internal sealed class Neuron
    {
        private const double TAU_M = 4.0;
        private const double TAU_S = 2.0;
        private const double TAU_R = 4.0;

        public string Name = "";
        public double Clamped { get; set; }
        public List<Synapse> SynapsesIn = new();
        public List<Neuron> NeuronsPost = new();
        public List<double> Spikes = new(); // Eq (1)

        public Neuron(string name = "neuron")
        {
            Name = name;
        }

        public double Epsilon(double s) // Eq (4)
        {
            if (s < 0.0)
            {
                return 0.0;
            }
            else
            {
                return Math.Exp(-s / TAU_M) - Math.Exp(-s / TAU_S);
            }
        }

        public double Eta(in double s) // Eq (5)
        {
            if (s < 0.0)
            {
                return 0.0;
            }
            else
            {
                return -Math.Exp(-s / TAU_R);
            }
        }

        public double Epsilond(in double s)
        {
            if (s < 0.0)
            {
                return 0.0;
            }
            else
            {
                return -Math.Exp(-s / TAU_M) / TAU_M + Math.Exp(-s / TAU_S) / TAU_S;
            }
        }

        public double Etad(in double s)
        {
            if (s < 0.0)
            {
                return 0.0;
            }
            else
            {
                return Math.Exp(-s / TAU_R) / TAU_R;
            }
        }

        public void Fire(double time)
        {
            Spikes.Add(time);
        }

        public void Forward(double time) // Eq (2)
        {
            const double threshold = 1.0;
            if (ComputeU(time) > threshold)
            {
                Fire(time);
            }
        }

        public double ComputeU(double time) // Eq (3)
        {
            double u = 0.0;
            foreach (var incoming_synapse in SynapsesIn)
            {
                foreach (var pre_spike in incoming_synapse.NeuronPre.Spikes)
                {
                    u += incoming_synapse.Weight * Epsilon(time - pre_spike - incoming_synapse.Delay);
                }
            }
            foreach (var ref_spike in Spikes)
            {
                u += Eta(time - ref_spike);
            }
            return u;
        }

        public void ComputeDeltaWeights(double learningRate) // Eq (9)
        {
            foreach (Synapse synapse in SynapsesIn)
            {
                foreach (double spike in Spikes)
                {
                    synapse.WeightDelta -= learningRate * ComputeDeDt(spike) * ComputeDtDw(synapse, spike);
                }
            }
        }

        public double ComputeDtDw(Synapse synapse, double spike) // Eq (10)
        {
            return -ComputeDuDw(synapse, spike) / ComputeDuDt(spike);
        }

        public double ComputeDuDw(Synapse synapse, double spike) // Eq (11)
        {
            double duDw = 0.0;
            foreach (double pre_spike in synapse.NeuronPre.Spikes)
            {
                duDw += Epsilon(spike - pre_spike - synapse.Delay);
            }

            foreach (double ref_spike in Spikes)
            {
                if (ref_spike < spike)
                {
                    duDw += -Etad(spike - ref_spike) * ComputeDtDw(synapse, ref_spike);
                }
            }

            return duDw;
        }
        public double ComputeDuDt(in double spike) // Eq (12)
        {
            double duDt = 0.0;
            foreach (Synapse synapse in SynapsesIn)
            {
                foreach (double pre_spike in synapse.NeuronPre.Spikes)
                {
                    duDt += synapse.Weight * Epsilond(spike - pre_spike - synapse.Delay);
                }
            }

            foreach (double ref_spike in Spikes)
            {
                if (ref_spike < spike)
                {
                    duDt += Etad(spike - ref_spike);
                }
            }

            if (duDt < 0.1) // handling discontinuity circumstance 1 Sec 3.2
            {
                duDt = 0.1;
            }
            return duDt;
        }

        public double ComputeDeDt(double spike) // Eq (13)
        {
            if (Clamped > 0.0)
            {
                if (spike == Spikes[0])
                {
                    return spike - Clamped;
                }
            }

            double deDt = 0.0;
            foreach (Neuron post_neuron_ptr in NeuronsPost)
            {
                foreach (double post_spike in post_neuron_ptr.Spikes)
                {
                    if (post_spike > spike)
                    {
                        deDt += post_neuron_ptr.ComputeDeDt(post_spike) * ComputeDposttDt(spike, post_neuron_ptr, post_spike);
                    }
                }
            }
            return deDt;
        }

        public double ComputeDposttDt(in double spike, Neuron post_neuron, in double post_spike) // Eq (14)
        {
            return -ComputeDpostuDt(spike, post_neuron, post_spike) / post_neuron.ComputeDuDt(post_spike);
        }

        public double ComputeDpostuDt(in double spike, Neuron post_neuron, in double post_spike) // Eq (15)
        {
            double dpostuDt = 0.0;
            foreach (Synapse synapse in post_neuron.SynapsesIn)
            {
                if (synapse.NeuronPre == this)
                {
                    dpostuDt -= synapse.Weight * Epsilond(post_spike - spike - synapse.Delay);
                }
            }

            foreach (double ref_post_spike in post_neuron.Spikes)
            {
                if (ref_post_spike < post_spike)
                {
                    dpostuDt -= Etad(post_spike - ref_post_spike) * ComputeDposttDt(spike, post_neuron, ref_post_spike);
                }
            }
            return dpostuDt;
        }

    }
}
