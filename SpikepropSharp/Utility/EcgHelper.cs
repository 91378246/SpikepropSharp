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
        private const int SAMPLE_INDEX_TRAIN = 0;
        private const int SAMPLE_INDEX_TEST = 1;
        private const int DATASET_TRAIN_SIZE = 4;
        private const int DATASET_VALIDATE_SIZE = 1000;
        private const double SOD_SAMPLING_THRESHOLD = 1;
        private const double SPIKE_TIME_INPUT = 6;
        private const double SPIKE_TIME_TRUE = 10;
        private const double SPIKE_TIME_FALSE = 16;

        // Network
        private const int INPUT_SIZE = 10;                          // 10 -> 15: Accuracy reduces, train time strongly increases
        private const int HIDDEN_SIZE = 1;                          // 2 -> 3: Reduces accuracy by about 5%, increases train time by about 50%, 2 -> 1: Reduces accuracy by about 4%
        private const int T_MAX = 40;                               // 40 -> 30: Reduces accuracy by about 4%, decreases train time by about 30%
        private const int TRIALS = 1;
        private const int EPOCHS = 500;
        private const int TEST_RUNS = 100;
        private const double TIMESTEP = 0.1;
        private const double LEARNING_RATE = 1e-2;                  // 1e-2 -> 1e-3:
        private const double ADAPTIVE_LEARNING_RATE_FACTOR = 0.01;

        private static Dictionary<double, bool> EcgSignalSpikesTrain { get; set; } = null!;
        private static double[] EcgSignalLabelsTrain { get; set; } = null!;
        private static Dictionary<double, bool> EcgSignalSpikesTest { get; set; } = null!;
        private static double[] EcgSignalLabelsTest { get; set; } = null!;

        private static void LoadData()
        {
            // Train
            double[] ecgSignalRaw = LoadMatlabEcgData(Path.Combine(DATA_DIR_PATH, $"ecg_{SAMPLE_INDEX_TRAIN}.mat"), "signal");
            EcgSignalSpikesTrain = ApplySodSampling(ecgSignalRaw, SOD_SAMPLING_THRESHOLD);

            double[] ecgSignalLabelsRaw = LoadMatlabEcgData(Path.Combine(DATA_DIR_PATH, $"ecg_{SAMPLE_INDEX_TRAIN}_ann.mat"), "ann").ToArray();
            EcgSignalLabelsTrain = ConvertAnnotationTimestampsToLabels(ecgSignalLabelsRaw).ToArray();

            // Test
            ecgSignalRaw = LoadMatlabEcgData(Path.Combine(DATA_DIR_PATH, $"ecg_{SAMPLE_INDEX_TRAIN}.mat"), "signal");
            EcgSignalSpikesTest = ApplySodSampling(ecgSignalRaw, SOD_SAMPLING_THRESHOLD);

            ecgSignalLabelsRaw = LoadMatlabEcgData(Path.Combine(DATA_DIR_PATH, $"ecg_{SAMPLE_INDEX_TRAIN}_ann.mat"), "ann").ToArray();
            EcgSignalLabelsTest = ConvertAnnotationTimestampsToLabels(ecgSignalLabelsRaw).ToArray();

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

        private static Sample[] GetDataset(Random rnd, bool test = false, int datasetSize = DATASET_TRAIN_SIZE)
        {
            // if (datasetSize % 2 != 0)
            // {
            //     throw new ArgumentException("DatasetSize has to be even", nameof(datasetSize));
            // }

            Sample[] dataset = new Sample[datasetSize];
            for (int i = 0; i < dataset.Length; i++)
            {
                int t = 0;
                bool sampleIsTrue = i % 2 == 0;

                // Get a random label timestamp
                if (sampleIsTrue)
                {
                    // Get a random label time
                    double labelT = (test ? EcgSignalLabelsTest : EcgSignalLabelsTrain)[rnd.Next((test ? EcgSignalLabelsTest : EcgSignalLabelsTrain).Length)];

                    // Set t to be equal or less than labelT
                    t = (int)labelT - rnd.Next(rnd.Next(INPUT_SIZE / 2));
                }
                // Get a random non label timestamp
                else
                {
                    while (t == 0)
                    {
                        // Get a random input time
                        t = rnd.Next((int)(test ? EcgSignalSpikesTest : EcgSignalSpikesTrain).Last().Key - INPUT_SIZE);

                        // Set t to be equal or less than that input time
                        t -= rnd.Next(rnd.Next(INPUT_SIZE / 2));

                        // Make sure the sample isn't true
                        if ((test ? EcgSignalLabelsTest : EcgSignalLabelsTrain).FirstOrDefault(l => l >= t && l < t + INPUT_SIZE) != default)
                        {
                            t = 0;
                        }
                    }
                }

                // Get the input
                List<double> input = new();
                while (input.Count < INPUT_SIZE)
                {
                    input.Add((test ? EcgSignalSpikesTest : EcgSignalSpikesTrain).ContainsKey(t++) ? SPIKE_TIME_INPUT : 0);
                }
                // Bias
                input.Add(0);

                // bool sampleIsTrue = EcgSignalLabelsTrain.FirstOrDefault(l => l >= t - INPUT_SIZE && l < t) != default;
                dataset[i] = new Sample(input.ToArray(), sampleIsTrue ? SPIKE_TIME_TRUE : SPIKE_TIME_FALSE);
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
                    input.Add(EcgSignalSpikesTest.ContainsKey(t++) ? SPIKE_TIME_INPUT : 0);
                }
                // Bias
                input.Add(0);


                dataset[i] = new Sample(
                    input: input.ToArray(),
                    output: EcgSignalLabelsTest.FirstOrDefault(l => l >= t && l < t + INPUT_SIZE) != default ? SPIKE_TIME_TRUE : SPIKE_TIME_FALSE);
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

        public static void RunTest(Random rnd, bool runTestsInBetween = true, bool loadPrevWeights = false)
        {
            Console.WriteLine("Loading data...");
            LoadData();

            Console.WriteLine("Running ECG test\n");

            double avgNrOfEpochs = 0;

            // Multiple trials for statistics
            Network[] networks = new Network[TRIALS];
            Network networkBest = new(rnd) { CurrentError = double.MaxValue };
            Dictionary<int, List<double>> errors = new();
            for (int trial = 0; trial < TRIALS; trial++)
            {
                ConsoleColor color = GetColorForIndex(trial);

                networks[trial] = CreateNetwork(rnd);
                errors[trial] = new List<double>();
                Neuron output_neuron = networks[trial].Layers[(int)Layer.Output][0];

                // Load a prev saved one
                string prevWeightsAndDelaysFile = $"network_0_060523.json";
                if (loadPrevWeights && File.Exists(prevWeightsAndDelaysFile))
                {
                    networks[trial].LoadWeightsAndDelays(prevWeightsAndDelaysFile);
                    Console.WriteLine($"[T{trial}] loaded weights and delays from {prevWeightsAndDelaysFile}");

                    Test(networks[trial], color, trial, -1);
                    Validate(networks[trial], trial);
                    Debugger.Break();
                }

                Stopwatch swFullTraining= new();
                swFullTraining.Start();

                // Main training loop
                int epochsQuarter = EPOCHS / 4;
                double adaptedLearningRate = LEARNING_RATE;
                for (int epoch = 0; epoch < EPOCHS; ++epoch)
                {
                    Stopwatch swEpoch = new();
                    swEpoch.Start();

                    double sumSquaredError = 0;
                    // Debug.WriteLine($"LR[{epoch}]: {adaptedLearningRate}");

                    Sample[] samples = GetDataset(rnd);
                    for (int sampleI = 0; sampleI < samples.Length; sampleI++)
                    {
                        // Debug.WriteLine($"Processing sample {sampleI + 1}/{DATASET_TRAIN_SIZE}");

                        // Forward propagation
                        networks[trial].Clear();
                        networks[trial].LoadSample(samples[sampleI]);
                        networks[trial].Forward(T_MAX, TIMESTEP);
                        if (output_neuron.Spikes.Count == 0)
                        {
                            Console.ForegroundColor = color;
                            Console.WriteLine($"[T{trial}] No output spikes! Replacing with different trial.");
                            trial -= 1;
                            sumSquaredError = epoch = (int)1e9;
                            break;
                        }
                        sumSquaredError += 0.5 * Math.Pow(output_neuron.Spikes[0] - output_neuron.FixedOutput, 2);

                        // Backward propagation
                        for (int l = networks[trial].Layers.Length - 1; l >= 1; l--)
                        {
                            for (int n = 0; n < networks[trial].Layers[l].Length; n++)
                            {
                                networks[trial].Layers[l][n].ComputeDeltaWeights(adaptedLearningRate);
                                for (int synI = 0; synI < networks[trial].Layers[l][n].SynapsesIn.Length; synI++)
                                {
                                    networks[trial].Layers[l][n].SynapsesIn[synI].Weight += networks[trial].Layers[l][n].SynapsesIn[synI].WeightDelta;
                                    networks[trial].Layers[l][n].SynapsesIn[synI].WeightDelta = 0;
                                }
                            }
                        }
                    }
                    Console.ForegroundColor = color;
                    Console.WriteLine($"[T{trial}] ep:{epoch} er:{sumSquaredError} t:{swEpoch.Elapsed:mm\\:ss}");
                    networks[trial].CurrentError = sumSquaredError;
                    errors[trial].Add(sumSquaredError);

                    if (sumSquaredError < networkBest.CurrentError)
                    {
                        networkBest = networks[trial].Clone();
                        networks[trial].SaveWeightsAndDelays($"network_{trial}_{DateTime.Now:ddMMyy}.json");
                    }

                    // Stopping criterion
                    if (sumSquaredError < 0.5)
                    {
                        avgNrOfEpochs = (avgNrOfEpochs * trial + epoch) / (trial + 1);
                        break;
                    }

                    if (epoch != 0 && epoch % epochsQuarter == 0)
                    {
                        // Test and validate
                        if (runTestsInBetween)
                        {
                            Test(networks[trial], color, trial, epoch);
                        }

                        Validate(networks[trial], trial);

                        // Adaptive learning rate
                        double oldLr = adaptedLearningRate;
                        adaptedLearningRate *= ADAPTIVE_LEARNING_RATE_FACTOR;
                        Console.WriteLine($"[T{trial}] Learning rate adapted: {oldLr} => {adaptedLearningRate}");
                    }                   
                }

                Console.WriteLine($"[T{trial}] finished after {swFullTraining.Elapsed:hh\\:mm\\:ss}");
            }

            Console.WriteLine($"Average nr of epochs per trial: {avgNrOfEpochs}");
            Console.WriteLine("\n#############################################################################");

            PrintConfiguration();
            Test(networkBest, ConsoleColor.Gray, -1, EPOCHS - 1);
            Validate(networkBest, 0);

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
                    Sample[] samples = GetDataset(rnd, test: true);
                    for (int sampleI = 0; sampleI < samples.Length; sampleI++)
                    {
                        double predictionRaw = network.Predict(samples[sampleI], T_MAX, TIMESTEP);
                        bool prediction = ConvertSpikeTimeToResult(predictionRaw);
                        bool label = ConvertSpikeTimeToResult(samples[sampleI].Output);

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

            void PrintConfiguration()
            {
                Console.WriteLine("#############################################################################");
                Console.WriteLine("CONFIG DATA:");
                Console.WriteLine($"\tSAMPLE_INDEX_TRAIN = {SAMPLE_INDEX_TRAIN}");
                Console.WriteLine($"\tSAMPLE_INDEX_TEST = {SAMPLE_INDEX_TEST}");
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
                Console.WriteLine($"\tADAPTIVE_LEARNING_RATE_FACTOR = {ADAPTIVE_LEARNING_RATE_FACTOR}");
                Console.WriteLine("#############################################################################");
            }

            void Validate(Network bestNetwork, int trial)
            {
                Console.WriteLine("Validating ... ");
                double lastTVal = (DATASET_VALIDATE_SIZE + 1) * INPUT_SIZE;
                double[] eegRaw = LoadMatlabEcgData(Path.Combine(DATA_DIR_PATH, $"ecg_{SAMPLE_INDEX_TEST}.mat"), "signal").ToArray()[..(int)lastTVal];
                double[] ecgLabelsRaw = LoadMatlabEcgData(Path.Combine(DATA_DIR_PATH, $"ecg_{SAMPLE_INDEX_TEST}_ann.mat"), "ann").Where(t => t < lastTVal).ToArray();
                Dictionary<double, bool> eegSignalsSpikeTrain = EcgSignalSpikesTest.Where(kv => kv.Key <= lastTVal).ToDictionary(x => x.Key, x => x.Value);
                ValidationResult result = new(errors[trial].ToArray(), eegRaw, ecgLabelsRaw, eegSignalsSpikeTrain);
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
