using MatFileHandler;
using SpikepropSharp.Components;

namespace SpikepropSharp.Utility;

public enum DataSet
{
	Train,
	Test,
	Validate
}

internal class DataManager
{
	// Constants
	private const string DATA_DIR_PATH = "Data/Samples";
	private const int SAMPLE_INDEX_TRAIN = 0;
	private const int DATASET_TRAIN_SIZE = 4;
	private const int SAMPLE_INDEX_VAL = 1;
	private const int DATASET_VAL_SIZE = 100;
	private const int SAMPLE_INDEX_TEST = 2;
	private const int DATASET_TEST_SIZE = 500;
	private const double SOD_SAMPLING_THRESHOLD = 0.5;
	private const double SPIKE_TIME_INPUT = 6;
	private const double SPIKE_TIME_TRUE = 10;
	private const double SPIKE_TIME_FALSE = 20;

	/// <summary>
	/// Contains the pre-processed data.<br/>
	/// [spikeTime:double] = (positive/negative spike):bool<br/>
	/// [labelTime:double]
	/// </summary>
	private Dictionary<DataSet, DataSetRecord> DataSets { get; } = new();
	private record DataSetRecord(Dictionary<double, bool> Spikes, double[] Labels);

	public DataManager(bool loadEcgData)
	{
		if (loadEcgData)
		{
			LoadEcgData();
		}
	}

	private void LoadEcgData()
	{
		DataSets[DataSet.Train] = CreateDataSetRecord(SAMPLE_INDEX_TRAIN);
		DataSets[DataSet.Validate] = CreateDataSetRecord(SAMPLE_INDEX_VAL);
		DataSets[DataSet.Test] = CreateDataSetRecord(SAMPLE_INDEX_TEST);

		static DataSetRecord CreateDataSetRecord(int sampleIndex)
		{
			if (!Directory.Exists(DATA_DIR_PATH))
			{
				Directory.CreateDirectory(DATA_DIR_PATH);
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine($"[ERROR] {DATA_DIR_PATH} is empty");
				Console.Read();
				Environment.Exit(1);
            }

			string unlabelledPath = Path.Combine(DATA_DIR_PATH, $"ecg_{sampleIndex}.mat");
            if (!File.Exists(unlabelledPath))
            {
                Directory.CreateDirectory(DATA_DIR_PATH);
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine($"[ERROR] Failed to find {unlabelledPath}");
                Console.Read();
                Environment.Exit(1);
            }

			string labelledPath = Path.Combine(DATA_DIR_PATH, $"ecg_{sampleIndex}_ann.mat");
            if (!File.Exists(labelledPath))
            {
                Directory.CreateDirectory(DATA_DIR_PATH);
				Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine($"[ERROR] Failed to find {labelledPath}");
                Console.Read();
                Environment.Exit(1);
            }

            double[] ecgSignalRaw = LoadMatlabEcgData(unlabelledPath, "signal");
			return new(
				Spikes: DataPreprocessor.ApplySodSampling(PreprocessEcgSignal(ecgSignalRaw), SOD_SAMPLING_THRESHOLD),
				Labels: LoadMatlabEcgData(labelledPath, "ann").ToArray()
			);
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

	private static double[] PreprocessEcgSignal(double[] ecgSignalRaw) =>
		DataPreprocessor.Denoise(ecgSignalRaw);

	public static double[] GetRawData(DataSet dataSet, int count, bool preprocess)
	{
		int sampleIndex = dataSet switch { DataSet.Train => SAMPLE_INDEX_TRAIN, DataSet.Validate => SAMPLE_INDEX_VAL, DataSet.Test => SAMPLE_INDEX_TEST };
		double[] ecgSignalRaw = LoadMatlabEcgData(Path.Combine(DATA_DIR_PATH, $"ecg_{sampleIndex}.mat"), "signal")[..count];

		return preprocess ? PreprocessEcgSignal(ecgSignalRaw) : ecgSignalRaw;
	}

	public Dictionary<double, bool> GetSpikesOfDataset(DataSet dataSet, int count) =>
		DataSets[dataSet].Spikes.Take(count).ToDictionary(k => k.Key, v => v.Value);

	public double[] GetLabelsOfDataset(DataSet dataSet, int count) =>
		DataSets[dataSet].Labels[..count];

	/// <summary>
	/// Returns a set of samples
	/// </summary>
	/// <param name="startIndex"></param>
	/// <param name="count"></param>
	/// <param name="sampleSize"></param>
	/// <param name="dataSet"></param>
	/// <returns></returns>
	public Sample[] GetSamples(int startIndex, int sampleSize, DataSet dataSet)
	{
		int dataSetSize = dataSet switch { DataSet.Train => DATASET_TRAIN_SIZE, DataSet.Validate => DATASET_VAL_SIZE, DataSet.Test => DATASET_TEST_SIZE };
		Sample[] dataset = new Sample[dataSetSize];
		for (int sampleI = startIndex; sampleI < startIndex + dataset.Length; sampleI++)
		{
			// Get the input
			List<double> input = new();
			bool isTrue = false;
			for (int t = 0; t < sampleSize; t++)
			{
				int signalTime = sampleI * sampleSize + t;
				input.Add(DataSets[dataSet].Spikes.ContainsKey(signalTime) ? SPIKE_TIME_INPUT : 0);
				isTrue = isTrue || DataSets[dataSet].Labels.Contains(signalTime);
			}
			// Bias
			input.Add(0);

			dataset[sampleI - startIndex] = new Sample(
				input: input.ToArray(),
				output: isTrue ? SPIKE_TIME_TRUE : SPIKE_TIME_FALSE);
		}
		return dataset;
	}

	/// <summary>
	/// Returns a set of randomly picked samples
	/// </summary>
	/// <param name="rnd"></param>
	/// <param name="sampleSize"></param>
	/// <param name="sampleSize"></param>
	/// <returns></returns>
	public Sample[] GetRndSamples(Random rnd, int sampleSize, DataSet dataSet)
	{
		Sample[] dataset = new Sample[dataSet switch { DataSet.Train => DATASET_TRAIN_SIZE, DataSet.Validate => DATASET_VAL_SIZE, DataSet.Test => DATASET_TEST_SIZE }];
		for (int i = 0; i < dataset.Length; i++)
		{
			int t = 0;
			bool sampleIsTrue = i % 2 == 0;

			// Get a random label timestamp
			if (sampleIsTrue)
			{
				// Get a random label time
				double labelT = DataSets[dataSet].Labels[rnd.Next(DataSets[dataSet].Labels.Length)];

				// Set t to be equal or less than labelT
				t = (int)labelT - rnd.Next(rnd.Next(sampleSize / 2));
			}
			// Get a random non label timestamp
			else
			{
				while (t == 0)
				{
					// Get a random input time
					t = rnd.Next((int)DataSets[dataSet].Spikes.Last().Key - sampleSize);

					// Set t to be equal or less than that input time
					t -= rnd.Next(rnd.Next(sampleSize / 2));

					// Make sure the sample isn't true
					if (DataSets[dataSet].Labels.FirstOrDefault(l => l >= t && l < t + sampleSize) != default)
					{
						t = 0;
					}
				}
			}

			// Get the input
			List<double> input = new();
			while (input.Count < sampleSize)
			{
				input.Add(DataSets[dataSet].Spikes.ContainsKey(t++) ? SPIKE_TIME_INPUT : 0);
			}
			// Bias
			input.Add(0);

			dataset[i] = new Sample(input.ToArray(), sampleIsTrue ? SPIKE_TIME_TRUE : SPIKE_TIME_FALSE);
		}

		DataPreprocessor.Shuffle(rnd, dataset);
		return dataset;
	}

	public static bool ConvertSpikeTimeToResult(double prediction) =>
		Math.Abs(prediction - SPIKE_TIME_TRUE) < Math.Abs(prediction - SPIKE_TIME_FALSE);
}
