using SpikepropSharp.Components;

namespace SpikepropSharp.Utility
{
    internal static class EcgHelper
    {
        private const double SPIKE_TIME_TRUE = 10;
        private const double SPIKE_TIME_FALSE = 16;

        private static List<Sample> GetDataset()
        {
            throw new NotImplementedException();
        }

        private static bool ConvertSpikeTimeToResult(double prediction) =>
            new List<double>() { SPIKE_TIME_TRUE, SPIKE_TIME_FALSE }
            .OrderBy(item => Math.Abs(prediction - item))
            .First()
            == SPIKE_TIME_TRUE;

        private static Network CreateNetwork(Random rnd)
        {
            throw new NotImplementedException();
        }

        public static void RunTest(Random rnd, int trials, int epochs, int testRuns, double maxTime, double timestep, double learningRate)
        {
            Console.WriteLine("Running ECG test\n");
            throw new NotImplementedException();
        }
    }
}
