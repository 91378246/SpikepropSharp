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
        public Synapse[] SynapsesIn { get; set; } = Array.Empty<Synapse>();
        public Neuron[] NeuronsPost { get; set; } = Array.Empty<Neuron>();
        public List<double> Spikes { get; set; } = new(); // Eq (1)

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
            // Foreach syn in
            for (int synI = 0; synI < SynapsesIn.Length; synI++)
            {
                // Foreach spike of the pre neuron of the syn in
                for (int spikeI = 0; spikeI < SynapsesIn[synI].NeuronPre.Spikes.Count; spikeI++)
                {
                    u += SynapsesIn[synI].Weight * Epsilon(time - SynapsesIn[synI].NeuronPre.Spikes[spikeI] - SynapsesIn[synI].Delay);
                }
            }

            for (int spikeI = 0; spikeI < Spikes.Count; spikeI++)
            {
                u += Eta(time - Spikes[spikeI]);
            }

            return u;
        }

        public void ComputeDeltaWeights(double learningRate) // Eq (9)
        {
            Parallel.For(0, SynapsesIn.Length, new ParallelOptions() { MaxDegreeOfParallelism = Environment.ProcessorCount }, synI =>
            {
                for (int spikeI = 0; spikeI < Spikes.Count; spikeI++)
                {
                    SynapsesIn[synI].WeightDelta -= learningRate * ComputeDeDt(Spikes[spikeI]) * ComputeDtDw(SynapsesIn[synI], Spikes[spikeI]);
                }
            });
        }

        public double ComputeDtDw(Synapse synapse, double spikeThis) // Eq (10)
        {
            return -ComputeDuDw(synapse, spikeThis) / ComputeDuDt(spikeThis);
        }

        public double ComputeDuDw(Synapse synapse, double spikeThis) // Eq (11)
        {
            double duDw = 0;

            // Foreach spike from the pre neuron of the syn
            for (int spikeI = 0; spikeI < synapse.NeuronPre.Spikes.Count; spikeI++)
            {
                duDw += Epsilon(spikeThis - synapse.NeuronPre.Spikes[spikeI] - synapse.Delay);
            }

            // Foreach spike from this neuron
            for (int spikeI = 0; spikeI < Spikes.Count; spikeI++)
            {
                if (Spikes[spikeI] < spikeThis)
                {
                    duDw += -Etad(spikeThis - Spikes[spikeI]) * ComputeDtDw(synapse, Spikes[spikeI]);
                }
            }

            return duDw;
        }

        public double ComputeDuDt(double spikeThis) // Eq (12)
        {
            double duDt = 0;

            // Foreach syn in
            for (int synI = 0; synI < SynapsesIn.Length; synI++)
            {
                // Foreach spike of the pre neuron of the syn in
                for (int spikeI = 0; spikeI < SynapsesIn[synI].NeuronPre.Spikes.Count; spikeI++)
                {
                    duDt += SynapsesIn[synI].Weight * Epsilond(spikeThis - SynapsesIn[synI].NeuronPre.Spikes[spikeI] - SynapsesIn[synI].Delay);
                }
            }

            // Foreach spike from this neuron
            for (int spikeI = 0; spikeI < Spikes.Count; spikeI++)
            {
                if (Spikes[spikeI] < spikeThis)
                {
                    duDt += Etad(spikeThis - Spikes[spikeI]);
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
            if (Clamped > 0)
            {
                if (spikeThis == Spikes[0])
                {
                    return spikeThis - Clamped;
                }
            }

            double deDt = 0.0;

            // Foreach neuron post
            for (int neuronI = 0; neuronI < NeuronsPost.Length; neuronI++)
            {
                // Forach spike of thaht neuron post
                for (int spikeI = 0; spikeI < NeuronsPost[neuronI].Spikes.Count; spikeI++)
                {
                    if (NeuronsPost[neuronI].Spikes[spikeI] > spikeThis)
                    {
                        deDt += NeuronsPost[neuronI].ComputeDeDt(NeuronsPost[neuronI].Spikes[spikeI]) * ComputeDposttDt(spikeThis, NeuronsPost[neuronI], NeuronsPost[neuronI].Spikes[spikeI]);
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
            double dpostuDt = 0;

            // Foreach syn in of the given neuron post
            for (int synI = 0; synI < neuronPost.SynapsesIn.Length; synI++)
            {
                if (neuronPost.SynapsesIn[synI].NeuronPre == this)
                {
                    dpostuDt -= neuronPost.SynapsesIn[synI].Weight * Epsilond(spikePost - spikeThis - neuronPost.SynapsesIn[synI].Delay);
                }
            }

            // Foreach spike from the given neuron post
            for (int spikeI = 0; spikeI < neuronPost.Spikes.Count; spikeI++)
            {
                if (neuronPost.Spikes[spikeI] < spikePost)
                {
                    dpostuDt -= Etad(spikePost - neuronPost.Spikes[spikeI]) * ComputeDposttDt(spikeThis, neuronPost, neuronPost.Spikes[spikeI]);
                }
            }
            return dpostuDt;
        }

    }
}
