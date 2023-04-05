using SpikepropSharp.Components;

namespace SpikepropSharp.Utility
{
    internal static class XorHelper
    {
        private const double SPIKE_TIME_INPUT = 6;
        private const double SPIKE_TIME_TRUE = 10;
        private const double SPIKE_TIME_FALSE = 16;

        private static List<Sample> GetDataset() => new()
        {
            new Sample(new List<double>() { 0, 0, 0 }, SPIKE_TIME_FALSE), // 0
            new Sample(new List<double>() { 0, SPIKE_TIME_INPUT, 0 }, SPIKE_TIME_TRUE), // 1
            new Sample(new List<double>() { SPIKE_TIME_INPUT, 0, 0 }, SPIKE_TIME_TRUE), // 1
            new Sample(new List<double>() { SPIKE_TIME_INPUT, SPIKE_TIME_INPUT, 0 }, SPIKE_TIME_FALSE)  // 0
        };

        private static bool ConvertSpikeTimeToResult(double prediction) =>
            new List<double>() { SPIKE_TIME_TRUE, SPIKE_TIME_FALSE }
            .OrderBy(item => Math.Abs(prediction - item))
            .First() 
            == SPIKE_TIME_TRUE;

        private static Network CreateNetwork(Random rnd)
        {
            Network network = new(rnd);
            network.Create(
                namesInput: new[] { "input 1", "input 2", "bias" },
                namesHidden: new[] { "hidden 1", "hidden 2", "hidden 3", "hidden 4", "hidden 5" },
                namesOutput: new[] { "output" }
            );

            return network;
        }

        public static void RunTest(Random rnd, int trials, int epochs, int testRuns, double maxTime, double timestep, double learningRate)
        {
            Console.WriteLine("Running XOR test\n");

            double AvgNrOfEpochs = 0;
            // Multiple trials for statistics
            for (int trial = 0; trial < trials; ++trial)
            {
                Network network = CreateNetwork(rnd);
                Neuron output_neuron = network.Layers[(int)Layer.Output].First();

                // Main training loop
                for (int epoch = 0; epoch < epochs; ++epoch)
                {
                    double sumSquaredError = 0;
                    foreach (Sample sample in GetDataset())
                    {
                        network.Clear();
                        network.LoadSample(sample);
                        network.Forward(maxTime, timestep);
                        if (output_neuron.Spikes.Count == 0)
                        {
                            Console.WriteLine("No output spikes! Replacing with different trial.");
                            trial -= 1;
                            sumSquaredError = epoch = (int)1e9;
                            break;
                        }
                        sumSquaredError += 0.5 * Math.Pow(output_neuron.Spikes.First() - output_neuron.Clamped, 2);

                        // Backward propagation and changing weights (no batch-mode)
                        foreach (List<Neuron> layer in network.Layers)
                        {
                            foreach (Neuron neuron in layer)
                            {
                                neuron.ComputeDeltaWeights(learningRate);
                                foreach (Synapse synapse in neuron.SynapsesIn)
                                {
                                    synapse.Weight += synapse.WeightDelta;
                                    synapse.WeightDelta = 0.0;
                                }
                            }
                        }
                    }
                    Console.Write(trial);
                    Console.Write(" ");
                    Console.Write(epoch);
                    Console.Write(" ");
                    Console.Write(sumSquaredError);
                    Console.Write("\n");

                    // Stopping criterion
                    if (sumSquaredError < 1.0)
                    {
                        AvgNrOfEpochs = (AvgNrOfEpochs * trial + epoch) / (trial + 1);
                        break;
                    }
                }

                // Test
                ConfusionMatrix cm = new();
                for (int testRun = 0; testRun < testRuns; testRun++)
                {
                    foreach (Sample sample in GetDataset())
                    {
                        double predictionRaw = network.Predict(sample, maxTime, timestep);
                        bool prediction = ConvertSpikeTimeToResult(predictionRaw);
                        bool label = ConvertSpikeTimeToResult(sample.Output);

                        if (prediction)
                        {
                            // TP
                            if (label)
                            {
                                cm.TruePositives++;
                            }
                            // FP
                            else
                            {
                                cm.FalsePositives++;
                            }
                        }
                        else
                        {
                            // FN
                            if (label)
                            {
                                cm.FalseNegatives++;
                            }
                            // TN
                            else
                            {
                                cm.TrueNegatives++;
                            }
                        }
                    }
                }

                Console.WriteLine("#############################################################################");
                Console.WriteLine($"TRIAL {trial} TEST RESULT");
                Console.WriteLine(cm.ToString());
                Console.WriteLine("#############################################################################");
            }

            Console.Write("Average nr of epochs per trial: ");
            Console.WriteLine(AvgNrOfEpochs);
            Console.ReadLine();
        }
    }
}
