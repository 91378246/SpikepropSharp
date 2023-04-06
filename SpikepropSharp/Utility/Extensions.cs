namespace SpikepropSharp.Utility
{
    public static class Extensions
    {
        public static double NextDouble(this Random rnd, double max) =>
            rnd.NextDouble(0, max);

        public static double NextDouble(this Random rnd, double min, double max) =>
            rnd.NextDouble() * (max - min) + min;
    }
}
