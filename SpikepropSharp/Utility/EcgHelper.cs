using SpikepropSharp.Components;
using SpikepropSharp.Data;
using System.Diagnostics;

namespace SpikepropSharp.Utility
{
	public static class EcgHelper
	{
		// 92% bei Hidden = 2, Epochs = 250, 0.5 sampling
		// 90% bei Hidden = 2, Epochs = 150, 0.1 sampling
		// 86% bei Hidden = 3, Epochs = , 0.1 sampling
		// 82% bei Hidden = 2, Epochs = 250, 0.05 sampling
		// 88% bei Hidden = 3, Epochs = 250, 0.5 sampling


		// Network
		private const int INPUT_SIZE = 10;                          // 10 -> 15: Accuracy reduces, train time strongly increases
		private const int HIDDEN_SIZE = 2;                          // 2 -> 3: Reduces accuracy by about 5%, increases train time by about 50%, 2 -> 1: Reduces accuracy by about 4%
		private const int T_MAX = 40;                               // 40 -> 30: Reduces accuracy by about 4%, decreases train time by about 30%
		private const int EPOCHS = 500;
		private const int VAL_RUNS = 10;
		private const double TIMESTEP = 0.1;
		private const double LEARNING_RATE = 1e-2;
		private const double ADAPTIVE_LEARNING_RATE_FACTOR = 0.1;

		private static DataManager DataManager { get; set; } = null!;

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
			DataManager = new(loadEcgData: true);

			Console.WriteLine("Running ECG test\n");

			// Multiple trials for statistics
			Network network = CreateNetwork(rnd);
			Network networkBest = new(rnd) { CurrentError = double.MaxValue };
			Neuron output_neuron = network.Layers[(int)Layer.Output][0];
			List<double> errors = new();

			// Load a prev saved one
			string prevWeightsAndDelaysFile = $"network_0_070523.json";
			if (loadPrevWeights && File.Exists(prevWeightsAndDelaysFile))
			{
				network.LoadWeightsAndDelays(prevWeightsAndDelaysFile);
				ConsoleExtensions.WriteLine($"Loaded weights and delays from {prevWeightsAndDelaysFile}", ConsoleColor.Red);

				Validate(rnd, network, -1);
				Test(errors.ToArray(), network, plot: true);
				Debugger.Break();
			}

			Stopwatch swFullTraining = new();
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

				Sample[] samples = DataManager.GetRndSamples(rnd, INPUT_SIZE, DataSet.Train);
				for (int sampleI = 0; sampleI < samples.Length; sampleI++)
				{
					// Debug.WriteLine($"Processing sample {sampleI + 1}/{DATASET_TRAIN_SIZE}");

					// Forward propagation
					network.Clear();
					network.LoadSample(samples[sampleI]);
					network.Forward(T_MAX, TIMESTEP);
					if (output_neuron.Spikes.Count == 0)
					{
						ConsoleExtensions.WriteLine("No output spikes! Replacing with different trial.", ConsoleColor.Red);
						sumSquaredError = epoch = (int)1e9;
						break;
					}
					sumSquaredError += 0.5 * Math.Pow(output_neuron.Spikes[0] - output_neuron.FixedOutput, 2);

					// Backward propagation
					for (int l = network.Layers.Length - 1; l >= 1; l--)
					{
						for (int n = 0; n < network.Layers[l].Length; n++)
						{
							network.Layers[l][n].ComputeDeltaWeights(adaptedLearningRate);
							for (int synI = 0; synI < network.Layers[l][n].SynapsesIn.Length; synI++)
							{
								network.Layers[l][n].SynapsesIn[synI].Weight += network.Layers[l][n].SynapsesIn[synI].WeightDelta;
								network.Layers[l][n].SynapsesIn[synI].WeightDelta = 0;
							}
						}
					}
				}

				ConsoleExtensions.WriteLine($"ep:{epoch:0000} er:{Math.Round(sumSquaredError, 3):00.000} t:{swEpoch.Elapsed:mm\\:ss\\:fff}", 
					sumSquaredError < networkBest.CurrentError ? ConsoleColor.Green : ConsoleColor.Gray);
				network.CurrentError = sumSquaredError;
				errors.Add(sumSquaredError);

				if (sumSquaredError < networkBest.CurrentError)
				{
					networkBest.CurrentError = sumSquaredError;
					//networkBest = network.Clone();
					network.SaveWeightsAndDelays($"network_{DateTime.Now:ddMMyy}.json");
				}

				if (epoch != 0 && (epoch % 100 == 0 || epoch % epochsQuarter == 0))
				{
					// Test and validate
					if (runTestsInBetween)
					{
						Console.WriteLine($"Validating after {swFullTraining.Elapsed:hh\\:mm\\:ss\\:fff}");
						double accuracy = Validate(rnd, network, epoch);

						// Stopping criterion
						if (accuracy >= 0.95)
						{
							ConsoleExtensions.WriteLine("Stopping criterion reached", ConsoleColor.Red);
							break;
						}
					}

					Test(errors.ToArray(), network, plot: epoch % epochsQuarter == 0);

					// Adaptive learning rate
					double oldLr = adaptedLearningRate;
					adaptedLearningRate *= ADAPTIVE_LEARNING_RATE_FACTOR;
					Console.WriteLine($"Learning rate adapted: {oldLr} => {adaptedLearningRate}");
				}
			}

			Console.WriteLine($"Finished after {swFullTraining.Elapsed:hh\\:mm\\:ss\\:fff}");
			Console.WriteLine("\n#############################################################################");

			PrintConfiguration();
			Validate(rnd, networkBest, EPOCHS - 1);
			Test(errors.ToArray(), networkBest, plot: true);

			Console.WriteLine("Done");
			Console.ReadLine();
		}

		private static double Validate(Random rnd, Network network, int epoch)
		{
			Console.WriteLine($"Running {VAL_RUNS} validations ...");

			// Test
			ConfusionMatrix cm = new();
			for (int valRun = 0; valRun < VAL_RUNS; valRun++)
			{
				Sample[] samples = DataManager.GetRndSamples(rnd, INPUT_SIZE, DataSet.Validate);
				for (int sampleI = 0; sampleI < samples.Length; sampleI++)
				{
					double predictionRaw = network.Predict(samples[sampleI], T_MAX, TIMESTEP);
					bool prediction = DataManager.ConvertSpikeTimeToResult(predictionRaw);
					bool label = DataManager.ConvertSpikeTimeToResult(samples[sampleI].Output);

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
			Console.WriteLine($"EPOCH {epoch} VALIDATION RESULT");
			Console.WriteLine(cm.ToString());
			Console.WriteLine("#############################################################################");

			return cm.Accuracy;
		}

		private static void PrintConfiguration()
		{
			Console.WriteLine("#############################################################################");
			Console.WriteLine("CONFIG DATA:");
			Console.WriteLine("CONFIG NETWORK:");
			Console.WriteLine($"\tINPUT_SIZE = {INPUT_SIZE}");
			Console.WriteLine($"\tHIDDEN_SIZE = {HIDDEN_SIZE}");
			Console.WriteLine($"\tT_MAX = {T_MAX}");
			Console.WriteLine($"\tEPOCHS = {EPOCHS}");
			Console.WriteLine($"\tTEST_RUNS = {VAL_RUNS}");
			Console.WriteLine($"\tTIMESTEP = {TIMESTEP}");
			Console.WriteLine($"\tLEARNING_RATE = {LEARNING_RATE}");
			Console.WriteLine($"\tADAPTIVE_LEARNING_RATE_FACTOR = {ADAPTIVE_LEARNING_RATE_FACTOR}");
			Console.WriteLine("#############################################################################");
		}

		private static void Test(double[] errors, Network bestNetwork, bool plot)
		{
			Console.WriteLine("Testing ... ");
			Sample[] testDataSet = DataManager.GetSamples(0, INPUT_SIZE, DataSet.Test);
			double[] ecgRaw = DataManager.GetRawData(DataSet.Test, testDataSet.Length * INPUT_SIZE, preprocess: true);
			Dictionary<double, bool> ecgSignalsSpikeTrain = DataManager.GetSpikesOfDataset(DataSet.Test, testDataSet.Length);
			ValidationResult result = new(
				errors: errors,
				ecgRaw: ecgRaw,
				ecgSignalSpikesTrain: ecgSignalsSpikeTrain.Where(s => s.Key < ecgRaw.Length).ToDictionary(k => k.Key, v => v.Value)
			);

			int sampleI = 0;
			foreach (Sample sample in testDataSet)
			{
				double predictionRaw = bestNetwork.Predict(sample, T_MAX, TIMESTEP);
				bool prediction = DataManager.ConvertSpikeTimeToResult(predictionRaw);
				bool label = DataManager.ConvertSpikeTimeToResult(sample.Output);

				result.Predictions.Add(new Prediction(sampleI * INPUT_SIZE, sampleI * INPUT_SIZE + INPUT_SIZE, prediction, label));
				sampleI++;
			}
			result.Save(plot: plot);
		}
	}
}
