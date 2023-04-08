using SpikepropSharp.Components;
using SpikepropSharp.Utility;
using System;
using System.Linq;
using static SpikepropSharp.ComponentsRecurrent.Network;
using System.Reflection.Emit;

namespace SpikepropSharp.ComponentsRecurrent
{
    public sealed class Network
    {
        public class Synapse
        {
            public double weight;
            public double delay;
            public double delta_weight;
            public List<double> dt_dws = new();
            public double u_m;
            public double u_s;

            public Synapse(double weight, double delay)
            {
                this.weight = weight;
                this.delay = delay;
            }
        };

        public class DPreSpike
        {
            public List<double> dpostts = new();
            public double u_m = 0;
            public double u_s = 0;

            public DPreSpike() { }
        }

        public class Connection
        {
            public Neuron neuron;
            public Neuron post_neuron;
            public List<Synapse> synapses = new();
            public List<DPreSpike> dpre_spikes = new();

            public Connection(Neuron neuron)
            {
                this.neuron = neuron;
            }
        }

        public struct NeuronSpike
        {
            public Neuron neuron;
            public double time;

            public NeuronSpike(Neuron neuron, double time) : this()
            {
                this.neuron = neuron;
                this.time = time;
            }

            public static bool operator ==(NeuronSpike s1, NeuronSpike s2) => s1.Equals(s2);

            public static bool operator !=(NeuronSpike s1, NeuronSpike s2) => !s1.Equals(s2);
        }

        public struct SynapseSpike
        {
            public Neuron neuron;
            public double weight;
            public double time;

            public SynapseSpike(Neuron neuron, double weight, double time)
            {
                this.neuron = neuron;
                this.weight = weight;
                this.time = time;
            }

            public static bool operator <(SynapseSpike a, SynapseSpike b) => a.time > b.time;
            public static bool operator >(SynapseSpike a, SynapseSpike b) => a.time < b.time;
        }

        public struct SpikeRecord
        {
            public Neuron neuron;
            public double time;
            public int index;

            public SpikeRecord(Neuron neuron, double time, int index)
            {
                this.neuron = neuron;
                this.time = time;
                this.index = index;
            }
        }

        public List<Neuron>[] Layers { get; private set; }
        public double CurrentError { get; set; } = double.MaxValue;
        
        private Random Rnd { get; }

        public Network(Random rnd)
        {
            Rnd = rnd;
        }

        public void Create(int inCount, int hiddenCount, int outCount)
        {
            Layers[(int)Layer.Input] = CreateLayer(inCount);
            Layers[(int)Layer.Hidden] = CreateLayer(hiddenCount);
            Layers[(int)Layer.Output] = CreateLayer(outCount);

            ConnectLayers(Layers[(int)Layer.Input], Layers[(int)Layer.Hidden]);
            ConnectLayers(Layers[(int)Layer.Hidden], Layers[(int)Layer.Output]);

            InitializeSynapses();
            connect_outgoing();

            static List<Neuron> CreateLayer(int neuronCount)
            {
                List<Neuron> layer = new();
                for (int i = 0; i < neuronCount; i++)
                {
                    layer.Add(new Neuron());
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
                    Connection incoming_connection = new(pre);
                    post.incoming_connections.Add(incoming_connection);
                    for (double delay_i = 16; delay_i > 0; delay_i--)
                    {
                        incoming_connection.synapses.Add(new Synapse(.0, delay_i + 1.0 + 1e-10));
                    }
                }
            }

            void InitializeSynapses()
            {
                foreach (List<Neuron> layer in Layers)
                {
                    foreach (Neuron n in layer)
                    {
                        foreach (Connection incoming_connection in n.incoming_connections)
                        {
                            foreach (Synapse synapse in incoming_connection.synapses)
                            {
                                synapse.weight = Rnd.NextDouble(-.025, .05);
                                synapse.delay *= 2;
                            }
                        }
                    }
                }
            }

            void connect_outgoing()
            {
                foreach (List<Neuron> layer in Layers)
                {
                    connect_outgoing_layer(layer);
                }
            }

            void connect_outgoing_layer(List<Neuron> layer)
            {
                foreach (Neuron neuron in layer)
                {
                    for (int conI = 0; conI < neuron.incoming_connections.Count; conI++)
                    {
                        neuron.incoming_connections[conI].post_neuron = neuron;
                        neuron.incoming_connections[conI].neuron.outgoing_connections.Add(neuron.incoming_connections[conI]);
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
                    neuron.spikes.Clear();
                }
            }
        }

        public int FirstSpikeResult(List<List<Neuron>> network)
        {
            return network.Last().IndexOf(network.Last().Min(Comparer<Neuron>.Create(CompareNeurons))!);

            static int CompareNeurons(Neuron a, Neuron b)
            {
                if (!a.spikes.Any()) return 1;
                if (!b.spikes.Any()) return -1;
                return a.spikes.First().time.CompareTo(b.spikes.First().time);
            }
        }

        public double ComputeLoss()
        {
            double lossPattern = 0;
            foreach (Neuron neuron in Layers.Last())
            {
                if (!neuron.spikes.Any())
                {
                    continue;
                }

                lossPattern += 0.5 * Math.Pow(neuron.spikes.First().time - neuron.clamped, 2);
            }
            return lossPattern;
        }

        public void LoadSample(Sample sample)
        {
            if (sample.Input.Length != Layers[(int)Layer.Input].Count)
            {
                throw new ArgumentException($"Invalid sample input count {sample.Input.Length}, expected {Layers[(int)Layer.Input].Count - 1}");
            }

            for (int i = 0; i < sample.Input.Length; i++)
            {
                //Layers[(int)Layer.Input][i].Fire(sample.Input[i]);
            }

            Layers[(int)Layer.Output][0].clamped = sample.Output;
        }

        public List<SpikeRecord> forward_propagate(double pattern)
        {
            Clear();
            Events events = new();

            bool NotAllOutputsSpiked() => Layers[(int)Layer.Output].Where(n => n.spikes.Count == 0).Any();

            while (NotAllOutputsSpiked() && events.active())
                events.process_event();

            return events.actual_spikes;
        }
    }
}
