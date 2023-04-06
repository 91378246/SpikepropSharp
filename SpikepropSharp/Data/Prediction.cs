namespace SpikepropSharp.Data
{
    public sealed class Prediction
    {
        public double TStart { get; set; }
        public double TEnd { get; set; }
        public bool PredictionResult { get; set; }
        public bool Label { get; set; }

        public Prediction(double tStart, double tEnd, bool predictionResult, bool label)
        {
            TStart = tStart;
            TEnd = tEnd;
            PredictionResult = predictionResult;
            Label = label;
        }
    }
}
