using SpikepropSharp.Utility;

namespace SpikepropSharp.Components
{
    public enum Layer
    {
        Input = 0,
        Hidden = 1,
        Output = 2
    }

    internal class Network
    {
        public List<Neuron>[] Layers { get; }

        private Random Rnd { get; }

        public Network(Random rnd)
        {
            Rnd = rnd;
            Layers = new List<Neuron>[3];
        }

        public void Create(string[] namesInput, string[] namesHidden, string[] namesOutput)
        {
            Layers[(int)Layer.Input] = CreateLayer(namesInput);
            Layers[(int)Layer.Hidden] = CreateLayer(namesHidden);
            Layers[(int)Layer.Output] = CreateLayer(namesOutput);

            ConnectLayers(Layers[(int)Layer.Input], Layers[(int)Layer.Hidden]);
            ConnectLayers(Layers[(int)Layer.Hidden], Layers[(int)Layer.Output]);

            InitializeWeights();

            static List<Neuron> CreateLayer(string[] keys)
            {
                List<Neuron> layer = new();
                foreach (string key in keys)
                {
                    layer.Add(new Neuron(key));
                }
                return new List<Neuron>(layer);
            }

            static void ConnectLayers(List<Neuron> pre_layer, List<Neuron> post_layer)
            {
                foreach (Neuron pre in pre_layer)
                {
                    foreach (Neuron post in post_layer)
                    {
                        ConnectNeurons(pre, post);
                    }
                }

                static void ConnectNeurons(Neuron pre, Neuron post)
                {
                    for (double delay_i = 16; delay_i > 0; delay_i--)
                    {
                        post.SynapsesIn.Add(new Synapse(pre, .0, delay_i + 1.0));
                    }
                    pre.NeuronsPost.Add(post);
                }
            }

            void InitializeWeights()
            {
                // Set random weights
                foreach (Neuron n in Layers[(int)Layer.Hidden])
                {
                    foreach (Synapse synapse in n.SynapsesIn)
                    {
                        synapse.Weight = Rnd.NextDouble(-.5, 1.0);
                    }
                }

                foreach (Synapse synapse in Layers[(int)Layer.Output].First().SynapsesIn)
                {
                    if (synapse.NeuronPre.Name == "hidden 5")
                    {
                        synapse.Weight = Rnd.NextDouble(-.5, 0.0);
                    }
                    else
                    {
                        synapse.Weight = Rnd.NextDouble(0.0, 1.0);
                    }
                }
            }
        }

        public void Clear()
        {
            foreach (List<Neuron> layer in Layers)
            {
                foreach (Neuron neuron in layer)
                {
                    neuron.Spikes.Clear();
                }
            }
        }

        public void LoadSample(Sample sample)
        {
            if (sample.Input.Count != Layers[(int)Layer.Input].Count)
            {
                throw new ArgumentException($"Invalid sample input count {sample.Input.Count}, expected {Layers[(int)Layer.Input].Count - 1}");
            }

            for (int i = 0; i < sample.Input.Count; i++)
            {
                Layers[(int)Layer.Input][i].Fire(sample.Input[i]);
            }

            Layers[(int)Layer.Output][0].Clamped = sample.Output;
        }

        public void Forward(double maxTime, double timestep)
        {
            IEnumerable<Neuron> neuronsOutNoSpikes = Layers[(int)Layer.Output].Where(n => n.Spikes.Count == 0);
            for (double time = 0.0; time < maxTime && neuronsOutNoSpikes.Any(); time += timestep)
            {
                foreach (List<Neuron> layer in Layers)
                {
                    foreach (Neuron neuron in layer)
                    {
                        neuron.Forward(time);
                    }
                }
            }
        }

        public double Predict(Sample sample, double maxTime, double timestep)
        {
            Clear();
            LoadSample(sample);
            Forward(maxTime, timestep);

            return Layers[(int)Layer.Output].First().Spikes.FirstOrDefault();
        }
    }
}
