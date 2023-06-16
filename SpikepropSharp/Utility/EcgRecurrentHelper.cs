using MatFileHandler;
using SpikepropSharp.Components;
using SpikepropSharp.Data;
using Network = SpikepropSharp.ComponentsRecurrent.Network;
using Neuron = SpikepropSharp.ComponentsRecurrent.Neuron;

namespace SpikepropSharp.Utility
{
	public static class EcgRecurrentHelper
    {
        // Data
        private const string DATA_DIR_PATH = "Data";
        private const int SAMPLE_INDEX = 0;
        private const int DATASET_TRAIN_SIZE = 10;
        private const int DATASET_VALIDATE_SIZE = 1000;
        private const double SOD_SAMPLING_THRESHOLD = 1;
        private const double SPIKE_TIME_INPUT = 6;
        private const double SPIKE_TIME_TRUE = 10;
        private const double SPIKE_TIME_FALSE = 16;

        // Network
        private const int INPUT_SIZE = 2;
        private const int HIDDEN_SIZE = 5;
        private const int OUTPUT_SIZE = 2;
        private const int T_MAX = 40;
        private const int TRIALS = 1;
        private const int EPOCHS = 100;
        private const int TEST_RUNS = 100;
        private const double TIMESTEP = 0.1;
        private const double LEARNING_RATE = 1e-2;

        private static Dictionary<double, bool> EcgSignalSpikesTrain { get; set; } = null!;
        private static double[] EcgSignalLabelsTrain { get; set; } = null!;

        private static void LoadData(int sampleIndex = SAMPLE_INDEX)
        {
            double[] ecgSignalRaw = LoadMatlabEcgData(Path.Combine(DATA_DIR_PATH, $"ecg_{sampleIndex}.mat"), "signal");
            EcgSignalSpikesTrain = ApplySodSampling(ecgSignalRaw, SOD_SAMPLING_THRESHOLD);

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
            // if (datasetSize % 2 != 0)
            // {
            //     throw new ArgumentException("DatasetSize has to be even", nameof(datasetSize));
            // }

            Sample[] dataset = new Sample[datasetSize];
            for (int i = 0; i < dataset.Length; i++)
            {
                int t = 0;
                //bool sampleIsTrue = i % 2 == 0;

                //// Get a random label timestamp
                //if (sampleIsTrue)
                //{
                //    // Get a random label time
                //    double labelT = EcgSignalLabelsTrain[rnd.Next(EcgSignalLabelsTrain.Length)];

                //    // Set t to be equal or less than labelT
                //    t = (int)labelT - rnd.Next(rnd.Next(INPUT_SIZE / 2));
                //}
                //// Get a random non label timestamp
                //else
                //{
                //    while (t == 0)
                //    {
                //        // Get a random input time
                //        t = rnd.Next((int)EcgSignalSpikesTrain.Last().Key - INPUT_SIZE);

                //        // Set t to be equal or less than that input time
                //        t -= rnd.Next(rnd.Next(INPUT_SIZE / 2));

                //        // Make sure the sample isn't true
                //        if (EcgSignalLabelsTrain.FirstOrDefault(l => l >= t && l < t + INPUT_SIZE) != default)
                //        {
                //            t = 0;
                //        }
                //    }
                //}

                // Get a random t
                t = rnd.Next((int)EcgSignalSpikesTrain.Last().Key - INPUT_SIZE);

                // Get the input
                List<double> input = new();
                while (input.Count < INPUT_SIZE)
                {
                    input.Add(EcgSignalSpikesTrain.ContainsKey(t++) ? SPIKE_TIME_INPUT : 0);
                }
                // Bias
                input.Add(0);

                bool sampleIsTrue = EcgSignalLabelsTrain.FirstOrDefault(l => l >= t - INPUT_SIZE && l < t) != default;
                dataset[i] = new Sample(input.ToArray(), sampleIsTrue ? SPIKE_TIME_TRUE : SPIKE_TIME_FALSE);
            }

            // Shuffle(rnd, dataset);
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

                dataset[i] = new Sample(
                    input: input.ToArray(),
                    output: EcgSignalLabelsTrain.FirstOrDefault(l => l >= t && l < t + INPUT_SIZE) != default ? SPIKE_TIME_TRUE : SPIKE_TIME_FALSE);
            }

            return dataset;
        }

        private static bool ConvertSpikeTimeToResult(double prediction) =>
            Math.Abs(prediction - SPIKE_TIME_TRUE) < Math.Abs(prediction - SPIKE_TIME_FALSE);

        private static Network CreateNetwork(Random rnd)
        {
            Network network = new(rnd);
            network.Create(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);

            return network;
        }

        public static void RunTest(Random rnd, bool runTestsInBetween = true, bool loadPrevWeights = false)
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
                //string prevWeightsAndDelaysFile = $"network_{trial}_070423.json";
                //if (loadPrevWeights && File.Exists(prevWeightsAndDelaysFile))
                //{
                //    networks[trial].LoadWeightsAndDelays(prevWeightsAndDelaysFile);
                //    Console.WriteLine($"[T{trial}] loaded weights and delays from {prevWeightsAndDelaysFile}");
                //}

                //for (int epoch = 0; epoch < 100; ++epoch)
                //{
                //    var spike_patterns_decimated = decimate_events(spike_patterns_train, 200, random_gen);
                //    double loss_batch = 0;
                //    double loss_epoch = 0;
                //    int error_epoch = 0;
                //    foreach (auto[pattern_i, pattern] in ranges.views.enumerate(spike_patterns_decimated))
                //    {
                //        // forward
                //        var spikes = networks[trial].forward_propagate(pattern);

                //        // update logs
                //        loss_batch += networks[trial].compute_loss(network);
                //        if (networks[trial].first_spike_result(network) != pattern.label)
                //        {
                //            error_epoch++;
                //        }

                //        // backprop
                //        networks[trial].backprop(spikes, learning_rate);

                //        // per batch change weights and report logs
                //        if ((pattern_i + 1) % batch_size == 0 || pattern_i + 1 == spike_patterns_decimated.size())
                //        {
                //            foreach (var layer in networks[trial].Layers)
                //            {
                //                foreach (var n in layer)
                //                {
                //                    foreach (var incoming_connection in n.incoming_connections)
                //                    {
                //                        foreach (var synapse in incoming_connection.synapses)
                //                        {
                //                            synapse.weight += synapse.delta_weight;
                //                            synapse.delta_weight = 0.0;
                //                        }
                //                    }
                //                }
                //            }
                //            Console.Write("batch loss after pattern ");
                //            Console.Write(pattern_i + 1);
                //            Console.Write(" ");
                //            Console.Write(loss_batch / (pattern_i % batch_size + 1));
                //            Console.Write("\n");
                //            loss_epoch += loss_batch;
                //            loss_batch = 0;
                //        }
                //    }
                //    // report epoch logs
                //    Console.Write("train loss  after epoch ");
                //    Console.Write(epoch);
                //    Console.Write(" ");
                //    Console.Write(loss_epoch / spike_patterns_train.size());
                //    Console.Write("\n");
                //    Console.Write("train error after epoch ");
                //    Console.Write(epoch);
                //    Console.Write(" ");
                //    Console.Write(100 * (double)error_epoch / spike_patterns_train.size());
                //    Console.Write(" %");
                //    Console.Write("\n");
                //    {
                //        double loss_validation = 0;
                //        int error_validation = 0;
                //        foreach (var pattern in spike_patterns_validation_decimated)
                //        {
                //            networks[trial].forward_propagate(pattern);
                //            loss_validation += networks[trial].compute_loss(network);
                //            if (networks[trial].first_spike_result(network) != pattern.label)
                //            {
                //                ++error_validation;
                //            }
                //        }
                //        Console.Write("validation loss  after epoch ");
                //        Console.Write(epoch);
                //        Console.Write(" ");
                //        Console.Write(loss_validation / spike_patterns_validation.size());
                //        Console.Write("\n");
                //        Console.Write("validation error after epoch ");
                //        Console.Write(epoch);
                //        Console.Write(" ");
                //        Console.Write(100 * (double)error_validation / spike_patterns_validation.size());
                //        Console.Write(" %");
                //        Console.Write("\n");
                //    }
                //}          
            });

            Console.Write("Average nr of epochs per trial: ");
            Console.WriteLine(AvgNrOfEpochs);
            Console.WriteLine("\n#############################################################################");

            Network bestNetwork = networks.OrderBy(network => network.CurrentError).First();
            PrintConfiguration();
            Test(bestNetwork, ConsoleColor.Gray, -1, EPOCHS - 1);
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
                        double predictionRaw = 0; // network.Predict(sample, T_MAX, TIMESTEP);
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

            void PrintConfiguration()
            {
                Console.WriteLine("#############################################################################");
                Console.WriteLine("CONFIG DATA:");
                Console.WriteLine($"\tSAMPLE_INDEX = {SAMPLE_INDEX}");
                Console.WriteLine($"\tDATASET_TRAIN_SIZE = {DATASET_TRAIN_SIZE}");
                Console.WriteLine($"\tDATASET_VALIDATE_SIZE = {DATASET_VALIDATE_SIZE}");
                Console.WriteLine($"\tSOD_SAMPLING_THRESHOLD = {SOD_SAMPLING_THRESHOLD}");
                Console.WriteLine($"\tSPIKE_TIME_INPUT = {SPIKE_TIME_INPUT}");
                Console.WriteLine($"\tSPIKE_TIME_TRUE = {SPIKE_TIME_TRUE}");
                Console.WriteLine($"\tSPIKE_TIME_FALSE = {SPIKE_TIME_FALSE}");
                Console.WriteLine("CONFIG NETWORK:");
                Console.WriteLine($"\tINPUT_SIZE = {INPUT_SIZE}");
                Console.WriteLine($"\tHIDDEN_SIZE = {HIDDEN_SIZE}");
                Console.WriteLine($"\tT_MAX = {T_MAX}");
                Console.WriteLine($"\tTRIALS = {TRIALS}");
                Console.WriteLine($"\tEPOCHS = {EPOCHS}");
                Console.WriteLine($"\tTEST_RUNS = {TEST_RUNS}");
                Console.WriteLine($"\tTIMESTEP = {TIMESTEP}");
                Console.WriteLine($"\tLEARNING_RATE = {LEARNING_RATE}");
                Console.WriteLine("#############################################################################");
            }

            void Validate(Network bestNetwork)
            {
                Console.WriteLine("Validating ... ");
                double lastTVal = (DATASET_VALIDATE_SIZE + 1) * INPUT_SIZE;
                double[] eegRaw = LoadMatlabEcgData(Path.Combine(DATA_DIR_PATH, $"ecg_{SAMPLE_INDEX}.mat"), "signal").ToArray()[..(int)lastTVal];
                double[] ecgLabelsRaw = LoadMatlabEcgData(Path.Combine(DATA_DIR_PATH, $"ecg_{SAMPLE_INDEX}_ann.mat"), "ann").Where(t => t < lastTVal).ToArray();
                Dictionary<double, bool> eegSignalsSpikeTrain = EcgSignalSpikesTrain.Where(kv => kv.Key <= lastTVal).ToDictionary(x => x.Key, x => x.Value);
                ValidationResult result = new(errors[bestNetwork].ToArray(), eegRaw, eegSignalsSpikeTrain);
                int sampleI = 0;
                foreach (Sample sample in GetValidationDataset())
                {
                    double predictionRaw = 0; // bestNetwork.Predict(sample, T_MAX, TIMESTEP);
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
