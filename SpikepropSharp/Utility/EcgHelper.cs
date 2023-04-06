using MatFileHandler;
using SpikepropSharp.Components;
using System.Diagnostics;

namespace SpikepropSharp.Utility
{
    public static class EcgHelper
    {
        // Data
        private const string DATA_DIR_PATH = "Data";
        private const int SAMPLE_INDEX = 0;
        private const double SOD_SAMPLING_THRESHOLD = 0.5;
        private const double SPIKE_TIME_INPUT = 6;
        private const double SPIKE_TIME_TRUE = 10;
        private const double SPIKE_TIME_FALSE = 16;

        // Network
        private const int INPUT_SIZE = 10;
        private const int T_MAX = 30;

        private static double[] EcgSignalSpikesTrain { get; set; } = null!;
        private static double[] EcgSignalLabelsTrain { get; set; } = null!;

        private static void LoadData(int sampleIndex = SAMPLE_INDEX, double sodSamplingThreshold = SOD_SAMPLING_THRESHOLD)
        {
            double[] ecgSignalRaw = LoadMatlabEcgData(Path.Combine(DATA_DIR_PATH, $"ecg_{sampleIndex}.mat"), "signal");
            EcgSignalSpikesTrain = ApplySodSampling(ecgSignalRaw, sodSamplingThreshold).ToArray();

            double[] ecgSignalLabelsRaw = LoadMatlabEcgData(Path.Combine(DATA_DIR_PATH, $"ecg_{sampleIndex}_ann.mat"), "ann").ToArray();
            EcgSignalLabelsTrain = ConvertAnnotationTimestampsToLabels(ecgSignalLabelsRaw).ToArray();

            /// <summary>
            /// Loads a field of the specified matlab file.
            /// Returns the first column of the specified field as the given type
            /// </summary>
            /// <param name="filePath">The path to the matlab file</param>
            /// <param name="fieldName">The name of the data field within the matlab file</param>
            /// <param name="scaleFactor">A factor to scale the matlab values with</param>
            /// <typeparam name="T">The type of the data within the specified field</typeparam>
            /// <returns></returns>
            /// <exception cref="FormatException"></exception>
            static double[] LoadMatlabEcgData(string filePath, string fieldName)
            {
                using FileStream fileStream = new(filePath, FileMode.Open);
                MatFileReader reader = new(fileStream);
                IMatFile matFile = reader.Read();

                return matFile[fieldName].Value.ConvertToDoubleArray()!;
            }

            /// <summary>
            /// Applies send-on-delta sampling on the given ecg signal and returns the resulting spikes.
            /// Returns it as a set of spike times
            /// </summary>
            /// <param name="ecgSignal"></param>
            /// <param name="threshold"></param>
            static List<double> ApplySodSampling(double[] ecgSignal, double threshold)
            {
                double yLevel = 0;
                List<double> spikes = new();

                for (int i = 0; i < ecgSignal.Length; i++)
                {
                    double diff = ecgSignal[i] - yLevel;
                    // Positive or negative spike
                    if (Math.Abs(diff) > threshold)
                    {
                        spikes.Add(i);
                        yLevel += Math.Sign(diff) * threshold;
                    }
                    //else
                    //{
                    //    spikes.Add(i);
                    //}
                }

                return spikes;
            }

            /// <summary>
            /// 
            /// </summary>
            /// <param name="annotationTimestamps"></param>
            /// <returns></returns>
            static List<double> ConvertAnnotationTimestampsToLabels(double[] annotationTimestamps)
            {
                List<double> spikes = new();
                spikes.AddRange(annotationTimestamps);

                return spikes;
            }
        }

        private static Sample[] GetDataset(Random rnd, int datasetSize = 6)
        {
            if (datasetSize % 2 != 0)
            {
                throw new ArgumentException("DatasetSize has to be even", nameof(datasetSize));
            }

            Sample[] dataset = new Sample[datasetSize];
            for (int i = 0; i < dataset.Length; i++)
            {
                int t = 0;
                bool sampleIsTrue = i % 2 == 0;

                // Get a random label timestamp
                if (sampleIsTrue)
                {
                    // Get a random label time
                    double labelT = EcgSignalLabelsTrain[rnd.Next(EcgSignalLabelsTrain.Length)];

                    // Set t to be equal or less than labelT
                    t = (int)labelT - rnd.Next(rnd.Next(INPUT_SIZE / 2));
                }
                // Get a random non label timestamp
                else
                {
                    while (t == 0)
                    {
                        // Get a random input time
                        t = rnd.Next((int)EcgSignalSpikesTrain.Last() - INPUT_SIZE);

                        // Set t to be equal or less than that input time
                        t -= rnd.Next(rnd.Next(INPUT_SIZE / 2));

                        // Make sure the sample isn't true
                        if (EcgSignalLabelsTrain.FirstOrDefault(l => l >= t && l < t + INPUT_SIZE) != default)
                        {
                            t = 0;
                        }
                    }
                }

                // Get the input
                List<double> input = new();
                while (input.Count < INPUT_SIZE)
                {
                    input.Add(EcgSignalSpikesTrain.Contains(t++) ? SPIKE_TIME_INPUT : 0);
                }
                // Bias
                input.Add(0);

                dataset[i] = new Sample(input, sampleIsTrue ? SPIKE_TIME_TRUE : SPIKE_TIME_FALSE);
            }

            Shuffle(rnd, dataset);
            return dataset;

            static T[] Shuffle<T>(Random rnd, T[] array)
            {
                int n = array.Length;
                while (n > 1)
                {
                    int k = rnd.Next(n--);
                    (array[k], array[n]) = (array[n], array[k]);
                }

                return array;
            }
        }

        private static bool ConvertSpikeTimeToResult(double prediction) =>
            Math.Abs(prediction - SPIKE_TIME_TRUE) < Math.Abs(prediction - SPIKE_TIME_FALSE);

        private static Network CreateNetwork(Random rnd)
        {
            string[] inputNeurons = new string[INPUT_SIZE + 1];
            for (int i = 0; i < inputNeurons.Length - 1; i++)
            {
                inputNeurons[i] = $"input {i}";
            }
            inputNeurons[^1] = "bias";

            // {2 + 1, 5, 1}
            Network network = new(rnd);
            network.Create(
                namesInput: inputNeurons,
                namesHidden: new[] { "hidden 1", "hidden 2", "hidden 3", "hidden 4", "hidden 5" },
                namesOutput: new[] { "output" }
            );

            return network;
        }

        public static void RunTest(Random rnd, int trials, int epochs, int testRuns, double timestep, double learningRate)
        {
            Console.WriteLine("Loading data...");
            LoadData();

            Console.WriteLine("Running ECG test\n");

            double AvgNrOfEpochs = 0;
            // Multiple trials for statistics
            Parallel.For(0, trials, new ParallelOptions() { MaxDegreeOfParallelism = Environment.ProcessorCount }, trial =>
            {
                ConsoleColor color = GetColorForIndex(trial);

                Network network = CreateNetwork(rnd);
                Neuron output_neuron = network.Layers[(int)Layer.Output].First();

                // Main training loop
                for (int epoch = 0; epoch < epochs; ++epoch)
                {
                    Stopwatch sw = new();
                    sw.Start();

                    double sumSquaredError = 0;
                    foreach (Sample sample in GetDataset(rnd))
                    {
                        network.Clear();
                        network.LoadSample(sample);
                        network.Forward(T_MAX, timestep);
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
                                neuron.ComputeDeltaWeights(learningRate);
                                foreach (Synapse synapse in neuron.SynapsesIn)
                                {
                                    synapse.Weight += synapse.WeightDelta;
                                    synapse.WeightDelta = 0.0;
                                }
                            }
                        }
                    }
                    Console.ForegroundColor = color;
                    Console.WriteLine($"[T{trial}] ep:{epoch} er:{sumSquaredError} t:{sw.Elapsed:mm\\:ss}");

                    // Stopping criterion
                    if (sumSquaredError < 1.0)
                    {
                        AvgNrOfEpochs = (AvgNrOfEpochs * trial + epoch) / (trial + 1);
                        break;
                    }

                    if (epoch % 2 == 0)
                    {
                        Test(network, color, trial, epoch);
                    }
                }

                Test(network, color, trial, epochs - 1);
            });

            Console.Write("Average nr of epochs per trial: ");
            Console.WriteLine(AvgNrOfEpochs);
            Console.WriteLine("\n#############################################################################");
            Console.WriteLine("Done");
            Console.ReadLine();

            static ConsoleColor GetColorForIndex(int i) =>
                (ConsoleColor)Enum.GetValues(typeof(ConsoleColor)).GetValue(i + 1)!;

            void Test(Network network, ConsoleColor color, int trial, int epoch)
            {
                Console.WriteLine($"[T{trial}] Running {testRuns} tests ...");

                // Test
                ConfusionMatrix cm = new();
                for (int testRun = 0; testRun < testRuns; testRun++)
                {
                    foreach (Sample sample in GetDataset(rnd))
                    {
                        double predictionRaw = network.Predict(sample, T_MAX, timestep);
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
                Console.WriteLine($"TRIAL {trial} EPOCH {epoch} TEST RESULT");
                Console.WriteLine(cm.ToString());
                Console.WriteLine("#############################################################################");
            }
        }
    }
}
