using SpikepropSharp.Utility;
using System.Text.Json;

namespace SpikepropSharp.Components
{
    public enum Layer
    {
        Input = 0,
        Hidden = 1,
        Output = 2
    }

    public sealed class Network
    {
        public Neuron[][] Layers { get; private set; }
        public double CurrentError { get; set; } = double.MaxValue;

        private Random Rnd { get; }
        private string[] NamesInput { get; set; }
        private string[] NamesHidden { get; set; }
        private string[] NamesOutput { get; set; }

        public Network(Random rnd)
        {
            Rnd = rnd;
            Layers = new Neuron[3][];
        }

        public void Create(string[] namesInput, string[] namesHidden, string[] namesOutput)
        {
            NamesInput = namesInput;
            NamesHidden = namesHidden;
            NamesOutput = namesOutput;

            Layers[(int)Layer.Input] = CreateLayer(namesInput);
            Layers[(int)Layer.Hidden] = CreateLayer(namesHidden);
            Layers[(int)Layer.Output] = CreateLayer(namesOutput);

            ConnectLayers(Layers[(int)Layer.Input], Layers[(int)Layer.Hidden]);
            ConnectLayers(Layers[(int)Layer.Hidden], Layers[(int)Layer.Output]);

            InitializeWeights();

            static Neuron[] CreateLayer(string[] keys)
            {
                List<Neuron> layer = new();
                foreach (string key in keys)
                {
                    layer.Add(new Neuron(key));
                }
                return layer.ToArray();
            }

            static void ConnectLayers(Neuron[] pre_layer, Neuron[] post_layer)
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
                    List<Synapse> synPost = post.SynapsesIn.ToList();
                    for (double delay_i = 16; delay_i > 0; delay_i--)
                    {
                        synPost.Add(new Synapse(pre, 0, delay_i + 1.0));
                    }
                    post.SynapsesIn = synPost.ToArray();

                    List<Neuron> neuronsPost = pre.NeuronsPost.ToList();
                    neuronsPost.Add(post);
                    pre.NeuronsPost = neuronsPost.ToArray();
                }
            }

            void InitializeWeights()
            {
                foreach (Neuron n in Layers[(int)Layer.Hidden])
                {
                    foreach (Synapse synapse in n.SynapsesIn)
                    {
                        synapse.Weight = Rnd.NextDouble(-0.5, 1);
                    }
                }

                foreach (Synapse synapse in Layers[(int)Layer.Output][0].SynapsesIn)
                {
                    if (synapse == Layers[(int)Layer.Output][0].SynapsesIn.Last())
                    {
                        synapse.Weight = Rnd.NextDouble(-0.5, 0);
                    }
                    else
                    {
                        synapse.Weight = Rnd.NextDouble(0, 1);
                    }
                }
            }
        }

        public void Clear()
        {
            for (int l = 0; l < Layers.Length; l++)
            {
                for (int n = 0; n < Layers[l].Length; n++)
                {
                    Layers[l][n].Spikes.Clear();
                }
            }
        }

        public void LoadSample(Sample sample)
        {
            if (sample.Input.Length != Layers[(int)Layer.Input].Length)
            {
                throw new ArgumentException($"Invalid sample input count {sample.Input.Length}, expected {Layers[(int)Layer.Input].Length - 1}");
            }

            for (int i = 0; i < sample.Input.Length; i++)
            {
                Layers[(int)Layer.Input][i].Fire(sample.Input[i]);
            }

            Layers[(int)Layer.Output][0].Clamped = sample.Output;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="tMax">For how many timesteps to run this forward simulation</param>
        /// <param name="timestep">For how much to increase t each time</param>
        public void Forward(double tMax, double timestep)
        {
            bool NotAllOutputsSpiked() => Layers[(int)Layer.Output].Where(n => n.Spikes.Count == 0).Any();

            for (double t = 0; t < tMax && NotAllOutputsSpiked(); t += timestep)
            {
                for (int l = 0; l < Layers.Length; l++)
                {
                    for (int n = 0; n < Layers[l].Length; n++)
                    {
                        Layers[l][n].Forward(t);
                    }
                }
            }
        }

        public double Predict(Sample sample, double tMax, double timestep)
        {
            Clear();
            LoadSample(sample);
            Forward(tMax, timestep);

            return Layers[(int)Layer.Output][0].Spikes.FirstOrDefault();
        }

        public void SaveWeightsAndDelays(string path)
        {
            (List<double> weights, List<double> delays) = GetWeightsAndDelays();
            File.WriteAllText(path, JsonSerializer.Serialize(new List<double>[] { weights, delays }));
        }

        public (List<double> weights, List<double> delays) GetWeightsAndDelays()
        {
            List<double> weights = new();
            List<double> delays = new();

            for (int l = 0; l < Layers.Length; l++)
            {
                for (int n = 0; n < Layers[l].Length; n++)
                {
                    for (int s = 0; s < Layers[l][n].SynapsesIn.Length; s++)
                    {
                        weights.Add(Layers[l][n].SynapsesIn[s].Weight);
                        delays.Add(Layers[l][n].SynapsesIn[s].Delay);
                    }
                }
            }

            return (weights, delays);
        }

        public void LoadWeightsAndDelays(string path)
        {
            List<double>[] weightsAndDelays = JsonSerializer.Deserialize<List<double>[]>(File.ReadAllText(path))!;
            SetWeightsAndDelays(weightsAndDelays[0], weightsAndDelays[1]);
        }

        public void SetWeightsAndDelays(List<double> weights, List<double> delays)
        {
            int synI = 0;
            foreach (Neuron[] layer in Layers)
            {
                foreach (Neuron neuron in layer)
                {
                    foreach (Synapse synapse in neuron.SynapsesIn)
                    {
                        synapse.Weight = weights[synI];
                        synapse.Delay = delays[synI];
                        synI++;
                    }
                }
            }
        }

        public Network Clone()
        {
            (List<double> weights, List<double> delays) = GetWeightsAndDelays();

            Network result = new(Rnd);
            result.Create(NamesInput, NamesHidden, NamesOutput);
            result.SetWeightsAndDelays(weights, delays);

            return result;
        }
    }
}
