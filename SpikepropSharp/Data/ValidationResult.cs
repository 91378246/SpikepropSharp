using System.Diagnostics;
using System.Text.Json;

namespace SpikepropSharp.Data
{
    public sealed class ValidationResult
    {
        private const string PYTHON_PATH = @"C:\Users\janha\AppData\Local\Programs\Python\Python310\python.exe";

        public double[] EegRaw { get; }
        public double[] EegLabelsRaw { get; }
        public Dictionary<double, bool> EcgSignalSpikesTrain { get; }
        public List<Prediction> Predictions { get; }

        public ValidationResult(double[] eegRaw, double[] eegLabelsRaw, Dictionary<double, bool> ecgSignalSpikesTrain)
        {
            EegRaw = eegRaw;
            EegLabelsRaw = eegLabelsRaw;
            EcgSignalSpikesTrain = ecgSignalSpikesTrain;
            Predictions = new();
        }

        public void Save(bool plot)
        {
            string validationFilePath = $"{DateTime.Now:yy.MM.dd.HH.mm.ss}-val_res.json";
            File.WriteAllText(validationFilePath, JsonSerializer.Serialize(this));

            Console.WriteLine($"Validation file saved to {validationFilePath}");

            if (plot)
            {
                RunPythonScript(Path.GetFullPath(validationFilePath));
            }
        }

        private static string RunPythonScript(string dataFilePath)
        {
            string pythonFilePath = Path.GetFullPath("Data/visualize.py");
            string result = "";
            ProcessStartInfo start = new()
            {
                FileName = PYTHON_PATH,
                Arguments = $"{pythonFilePath} {dataFilePath}",
                UseShellExecute = false,
                WorkingDirectory = ""
,
                RedirectStandardOutput = true
            };

            using (Process process = Process.Start(start))
            {
                using StreamReader reader = process.StandardOutput;
                result = reader.ReadToEnd();
                Console.WriteLine(result);
            }

            return result;
        }
    }
}
