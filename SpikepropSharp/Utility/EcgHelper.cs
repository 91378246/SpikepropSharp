using MatFileHandler;
using SpikepropSharp.Components;
using SpikepropSharp.Data;
using System.Diagnostics;

namespace SpikepropSharp.Utility
{
    public static class EcgHelper
    {
        // Data
        private const string DATA_DIR_PATH = "Data";
        private const int SAMPLE_INDEX = 0;
        private const int DATASET_TRAIN_SIZE = 10;
        private const int DATASET_VALIDATE_SIZE = 1000;
        private const double SOD_SAMPLING_THRESHOLD = 0.5;
        private const double SPIKE_TIME_INPUT = 6;
        private const double SPIKE_TIME_TRUE = 10;
        private const double SPIKE_TIME_FALSE = 16;

        // Network
        private const int INPUT_SIZE = 10;
        private const int HIDDEN_SIZE = 10;
        private const int T_MAX = 30;
        private const int TRIALS = 1;
        private const int EPOCHS = 100;
        private const int TEST_RUNS = 100;
        private const double TIMESTEP = 0.1;
        private const double LEARNING_RATE = 1e-2;

        private static Dictionary<double, bool> EcgSignalSpikesTrain { get; set; } = null!;
        private static double[] EcgSignalLabelsTrain { get; set; } = null!;

        private static void LoadData(int sampleIndex = SAMPLE_INDEX, double sodSamplingThreshold = SOD_SAMPLING_THRESHOLD)
        {
            double[] ecgSignalRaw = LoadMatlabEcgData(Path.Combine(DATA_DIR_PATH, $"ecg_{sampleIndex}.mat"), "signal");
            EcgSignalSpikesTrain = ApplySodSampling(ecgSignalRaw, sodSamplingThreshold);

            double[] ecgSignalLabelsRaw = LoadMatlabEcgData(Path.Combine(DATA_DIR_PATH, $"ecg_{sampleIndex}_ann.mat"), "ann").ToArray();
            EcgSignalLabelsTrain = ConvertAnnotationTimestampsToLabels(ecgSignalLabelsRaw).ToArray();

            /// <summary>
            /// Applies send-on-delta sampling on the given ecg signal and returns the resulting spikes.
            /// Returns it as a set of spike times
            /// </summary>
            /// <param name="ecgSignal"></param>
            /// <param name="threshold"></param>
            static Dictionary<double, bool> ApplySodSampling(double[] ecgSignal, double threshold)
            {
                double yLevel = 0;
                Dictionary<double, bool> spikes = new();

                for (int i = 0; i < ecgSignal.Length; i++)
                {
                    double diff = ecgSignal[i] - yLevel;
                    if (Math.Abs(diff) > threshold)
                    {
                        spikes[i] = Math.Sign(diff) > 0;
                        yLevel += Math.Sign(diff) * threshold;
                    }
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
        private static double[] LoadMatlabEcgData(string filePath, string fieldName)
        {
            using FileStream fileStream = new(filePath, FileMode.Open);
            MatFileReader reader = new(fileStream);
            IMatFile matFile = reader.Read();

            return matFile[fieldName].Value.ConvertToDoubleArray()!;
        }

        private static Sample[] GetDataset(Random rnd, int datasetSize = DATASET_TRAIN_SIZE)
        {
            if (datasetSize % 2 != 0)
            {
                throw new ArgumentException("DatasetSize has to be even", nameof(datasetSize));
            }

            Sample[] dataset = new Sample[datasetSize];
            for (int i = 0; i < dataset.Length; i++)
            {
                int t = 0;
                bool sampleIsTrue = i < 3;

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
                        t = rnd.Next((int)EcgSignalSpikesTrain.Last().Key - INPUT_SIZE);

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
                    input.Add(EcgSignalSpikesTrain.ContainsKey(t++) ? SPIKE_TIME_INPUT : 0);
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

        private static Sample[] GetValidationDataset()
        {
            Sample[] dataset = new Sample[DATASET_VALIDATE_SIZE];
            for (int i = 0; i < dataset.Length; i++)
            {
                int t = i * INPUT_SIZE;
                List<double> input = new();
                while (input.Count < INPUT_SIZE)
                {
                    input.Add(EcgSignalSpikesTrain.ContainsKey(t++) ? SPIKE_TIME_INPUT : 0);
                }
                // Bias
                input.Add(0);


                dataset[i] = new Sample(
                    input: input,
                    output: EcgSignalLabelsTrain.FirstOrDefault(l => l >= t && l < t + INPUT_SIZE) != default ? SPIKE_TIME_TRUE : SPIKE_TIME_FALSE);
            }

            return dataset;
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

            string[] hiddenNeurons = new string[HIDDEN_SIZE];
            for (int i = 0; i < hiddenNeurons.Length; i++)
            {
                hiddenNeurons[i] = $"hidden {i}";
            }

            Network network = new(rnd);
            network.Create(
                namesInput: inputNeurons,
                namesHidden: hiddenNeurons,
                namesOutput: new[] { "output" }
            );

            return network;
        }

        public static void RunTest(Random rnd, bool runTestsInBetween = false, bool loadPrevWeights = false)
        {
            Console.WriteLine("Loading data...");
            LoadData();

            Console.WriteLine("Running ECG test\n");

            double AvgNrOfEpochs = 0;

            // Multiple trials for statistics
            Network[] networks = new Network[TRIALS];
            Dictionary<Network, List<double>> errors = new();
            Parallel.For(0, TRIALS, new ParallelOptions() { MaxDegreeOfParallelism = Environment.ProcessorCount }, trial =>
            {
                ConsoleColor color = GetColorForIndex(trial);

                networks[trial] = CreateNetwork(rnd);
                errors[networks[trial]] = new List<double>();
                Neuron output_neuron = networks[trial].Layers[(int)Layer.Output].First();
                double lowestError = double.MaxValue;

                // Load a prev saved one
                string prevWeightsAndDelaysFile = $"network_{trial}_070423.json";
                if (loadPrevWeights && File.Exists(prevWeightsAndDelaysFile))
                {
                    networks[trial].LoadWeightsAndDelays(prevWeightsAndDelaysFile);
                    Console.WriteLine($"[T{trial}] loaded weights and delays from {prevWeightsAndDelaysFile}");
                }

                // Main training loop
                for (int epoch = 0; epoch < EPOCHS; ++epoch)
                {
                    Stopwatch sw = new();
                    sw.Start();

                    double sumSquaredError = 0;
                    foreach (Sample sample in GetDataset(rnd))
                    {
                        networks[trial].Clear();
                        networks[trial].LoadSample(sample);
                        networks[trial].Forward(T_MAX, TIMESTEP);
                        if (output_neuron.Spikes.Count == 0)
                        {
                            Console.ForegroundColor = color;
                            Console.WriteLine($"[T{trial}] No output spikes! Replacing with different trial.");
                            trial -= 1;
                            sumSquaredError = epoch = (int)1e9;
                            break;
                        }
                        sumSquaredError += 0.5 * Math.Pow(output_neuron.Spikes.First() - output_neuron.Clamped, 2);

                        // Backward propagation
                        foreach (List<Neuron> layer in networks[trial].Layers)
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
                    Console.WriteLine($"[T{trial}] ep:{epoch} er:{sumSquaredError} t:{sw.Elapsed:mm\\:ss}");
                    networks[trial].CurrentError = sumSquaredError;
                    errors[networks[trial]].Add(sumSquaredError);

                    if (sumSquaredError < lowestError)
                    {
                        networks[trial].SaveWeightsAndDelays($"network_{trial}_{DateTime.Now:ddMMyy}.json");
                    }

                    // Stopping criterion
                    if (sumSquaredError < 1.0)
                    {
                        AvgNrOfEpochs = (AvgNrOfEpochs * trial + epoch) / (trial + 1);
                        break;
                    }

                    if (runTestsInBetween && epoch % 2 == 0)
                    {
                        Test(networks[trial], color, trial, epoch);
                    }

                    if (epoch % 10 == 0)
                    {
                        Validate(networks[trial]);
                    }
                }

                Test(networks[trial], color, trial, EPOCHS - 1);
            });

            Console.Write("Average nr of epochs per trial: ");
            Console.WriteLine(AvgNrOfEpochs);
            Console.WriteLine("\n#############################################################################");

            Network bestNetwork = networks.OrderBy(network => network.CurrentError).First();
            Validate(bestNetwork);

            Console.WriteLine("Done");
            Console.ReadLine();

            static ConsoleColor GetColorForIndex(int i) =>
                (ConsoleColor)Enum.GetValues(typeof(ConsoleColor)).GetValue(i + 2)!;

            void Test(Network network, ConsoleColor color, int trial, int epoch)
            {
                Console.WriteLine($"[T{trial}] Running {TEST_RUNS} tests ...");

                // Test
                ConfusionMatrix cm = new();
                for (int testRun = 0; testRun < TEST_RUNS; testRun++)
                {
                    foreach (Sample sample in GetDataset(rnd))
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
                Console.WriteLine($"TRIAL {trial} EPOCH {epoch} TEST RESULT");
                Console.WriteLine(cm.ToString());
                Console.WriteLine("#############################################################################");
            }

            Dictionary<double, bool> GetRange(Dictionary<double, bool> dict, int startIndex, int endIndex) =>
                dict.OrderBy(d => d.Key).Skip(startIndex).Take(endIndex - startIndex + 1).ToDictionary(k => k.Key, v => v.Value);

            void Validate(Network bestNetwork)
            {
                Console.WriteLine("Validating ... ");
                double lastTVal = (DATASET_VALIDATE_SIZE + 1) * INPUT_SIZE;
                double[] eegRaw = LoadMatlabEcgData(Path.Combine(DATA_DIR_PATH, $"ecg_{SAMPLE_INDEX}.mat"), "signal").ToArray()[..(int)lastTVal];
                double[] ecgLabelsRaw = LoadMatlabEcgData(Path.Combine(DATA_DIR_PATH, $"ecg_{SAMPLE_INDEX}_ann.mat"), "ann").Where(t => t < lastTVal).ToArray();
                Dictionary<double, bool> eegSignalsSpikeTrain = EcgSignalSpikesTrain.Where(kv => kv.Key <= lastTVal).ToDictionary(x => x.Key, x => x.Value);
                ValidationResult result = new(errors[bestNetwork].ToArray(), eegRaw, ecgLabelsRaw, eegSignalsSpikeTrain);
                int sampleI = 0;
                foreach (Sample sample in GetValidationDataset())
                {
                    double predictionRaw = bestNetwork.Predict(sample, T_MAX, TIMESTEP);
                    bool prediction = ConvertSpikeTimeToResult(predictionRaw);
                    bool label = ConvertSpikeTimeToResult(sample.Output);

                    result.Predictions.Add(new Prediction(sampleI * INPUT_SIZE, sampleI * INPUT_SIZE + INPUT_SIZE, prediction, label));
                    sampleI++;
                }
                result.Save(plot: true);
            }
        }
    }
}
