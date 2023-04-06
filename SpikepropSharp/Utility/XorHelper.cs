using SpikepropSharp.Components;

namespace SpikepropSharp.Utility
{
    public static class XorHelper
    {
        private const int TRIALS = 10;
        private const int EPOCHS = 1000;
        private const int TEST_RUNS = 100;
        private const double TIMESTEP = 0.1;
        private const double T_MAX = 40;
        private const double LEARNING_RATE = 1e-2;

        private const double SPIKE_TIME_INPUT = 6;
        private const double SPIKE_TIME_TRUE = 10;
        private const double SPIKE_TIME_FALSE = 16;

        /// <summary>
        /// [input 1, input 2, bias] = SPIKE_TIME_TRUE/SPIKE_TIME_FALSE
        /// </summary>
        /// <returns></returns>
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
            // {2 + 1, 5, 1}
            Network network = new(rnd);
            network.Create(
                namesInput: new[] { "input 1", "input 2", "bias" },
                namesHidden: new[] { "hidden 1", "hidden 2", "hidden 3", "hidden 4", "hidden 5" },
                namesOutput: new[] { "output" }
            );

            return network;
        }

        public static void RunTest(Random rnd)
        {
            Console.WriteLine("Running XOR test\n");

            double AvgNrOfEpochs = 0;
            // Multiple trials for statistics
            Parallel.For(0, TRIALS, new ParallelOptions() { MaxDegreeOfParallelism = Environment.ProcessorCount }, trial =>
            {
                ConsoleColor color = GetColorForIndex(trial);
                Network network = CreateNetwork(rnd);
                Neuron output_neuron = network.Layers[(int)Layer.Output].First();

                // Main training loop
                for (int epoch = 0; epoch < EPOCHS; ++epoch)
                {
                    double sumSquaredError = 0;
                    foreach (Sample sample in GetDataset())
                    {
                        network.Clear();
                        network.LoadSample(sample);
                        network.Forward(T_MAX, TIMESTEP);
                        if (output_neuron.Spikes.Count == 0)
                        {
                            Console.ForegroundColor = color;
                            Console.WriteLine($"[T{trial}] No output spikes! Replacing with different trial.");
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
                                neuron.ComputeDeltaWeights(LEARNING_RATE);
                                foreach (Synapse synapse in neuron.SynapsesIn)
                                {
                                    synapse.Weight += synapse.WeightDelta;
                                    synapse.WeightDelta = 0.0;
                                }
                            }
                        }
                    }
                    Console.ForegroundColor = color;
                    Console.WriteLine($"[T{trial}] ep:{epoch} er:{sumSquaredError}");

                    // Stopping criterion
                    if (sumSquaredError < 1.0)
                    {
                        AvgNrOfEpochs = (AvgNrOfEpochs * trial + epoch) / (trial + 1);
                        break;
                    }
                }

                // Test
                ConfusionMatrix cm = new();
                for (int testRun = 0; testRun < TEST_RUNS; testRun++)
                {
                    foreach (Sample sample in GetDataset())
                    {
                        double predictionRaw = network.Predict(sample, T_MAX, TIMESTEP);
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

                Console.ForegroundColor = color;
                Console.WriteLine("#############################################################################");
                Console.WriteLine($"TRIAL {trial} TEST RESULT");
                Console.WriteLine(cm.ToString());
                Console.WriteLine("#############################################################################");
            });

            Console.Write("Average nr of epochs per trial: ");
            Console.WriteLine(AvgNrOfEpochs);
            Console.WriteLine("\n#############################################################################");
            Console.WriteLine("Done");
            Console.ReadLine();

            static ConsoleColor GetColorForIndex(int i) =>
                (ConsoleColor)Enum.GetValues(typeof(ConsoleColor)).GetValue(i + 1)!;
        }
    }
}
