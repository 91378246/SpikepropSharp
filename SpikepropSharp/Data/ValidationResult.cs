using System.Diagnostics;
using System.Text.Json;

namespace SpikepropSharp.Data
{
    public sealed class ValidationResult
    {
        private const string PYTHON_PATH_0 = @"C:\Users\janha\AppData\Local\Programs\Python\Python311\python.exe";
        private const string PYTHON_PATH_1 = @"C:\Users\janha\AppData\Local\Programs\Python\Python310\python.exe";

        public double[] Errors { get; }
        public double[] EegRaw { get; }
        public Dictionary<double, bool> EcgSignalSpikesTrain { get; }
        public List<Prediction> Predictions { get; }

        public ValidationResult(double[] errors, double[] ecgRaw, Dictionary<double, bool> ecgSignalSpikesTrain)
        {
            Errors = errors;
            EegRaw = ecgRaw;
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

        private static async void RunPythonScript(string dataFilePath)
        {
            string pythonFilePath = Path.GetFullPath("Data/visualize.py");
            ProcessStartInfo start = new()
            {
                FileName = File.Exists(PYTHON_PATH_0) ? PYTHON_PATH_0 : PYTHON_PATH_1,
                Arguments = $"{pythonFilePath} {dataFilePath}",
                UseShellExecute = false,
                WorkingDirectory = ""
,
                RedirectStandardOutput = true
            };

            using Process process = Process.Start(start);
            using StreamReader reader = process.StandardOutput;
            string output = await reader.ReadToEndAsync();

            if (!string.IsNullOrWhiteSpace(output))
            {
                Console.WriteLine(output);
            }
        }
    }
}
