using System.Text.Json;

namespace SpikepropSharp.Data
{
    public sealed class ValidationResult
    {
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

        public void Save()
        {
            string validationFilePath = $"{DateTime.Now:yy.MM.dd.HH.mm.ss}-val_res.json";
            File.WriteAllText(validationFilePath, JsonSerializer.Serialize(this));

            Console.WriteLine($"Validation file saved to {validationFilePath}");
        }
    }
}
