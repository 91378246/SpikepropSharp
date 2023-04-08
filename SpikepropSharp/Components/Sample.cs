namespace SpikepropSharp.Components;

public sealed class Sample
{
    public double[] Input { get; set; }
    public double Output { get; set; }

    public Sample(double[] input, double output)
    {
        Input = input;
        Output = output;
    }
}
