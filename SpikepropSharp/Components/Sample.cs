namespace SpikepropSharp.Components
{
    internal sealed class Sample
    {
        public List<double> Input { get; set; } = new(3);
        public double Output { get; set; }

        public Sample(List<double> input, double output)
        {
            Input = input;
            Output = output;
        }
    }
}
