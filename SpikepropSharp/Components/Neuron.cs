namespace SpikepropSharp.Components;

public sealed class Neuron
{
	/// <summary>
	/// Defines the raise of ε in ms
	/// </summary>
	private const double TAU_M = 4;
	/// <summary>
	/// Defines the decay of ε in ms
	/// </summary>
	private const double TAU_S = 2;
	/// <summary>
	/// Defines the decay of η in ms
	/// </summary>
	private const double TAU_R = 4;
	private const double THRESHOLD = 1;

	public string Name { get; }
	public double FixedOutput { get; set; }
	public Synapse[] SynapsesIn { get; set; } = Array.Empty<Synapse>();
	public Neuron[] NeuronsPost { get; set; } = Array.Empty<Neuron>();
	public List<double> Spikes { get; set; } = new();

	public Neuron(string name)
	{
		Name = name;
	}

	public void AddPostNeuron(Neuron neuronPost)
	{
		List<Neuron> neuronsPost = NeuronsPost.ToList();
		neuronsPost.Add(neuronPost);
		NeuronsPost = neuronsPost.ToArray();
	}

	/// <summary>
	/// Eq. 4<br/>
	/// ε = The contribution of one synaptic terminal on the potential of the postsynaptic neuron caused by a presynaptic spike
	/// </summary>
	/// <param name="s">t - spikeT - delay</param>
	/// <returns></returns>
	private static double Epsilon(double s) =>
		s < 0 ? 0 : Math.Exp(-s / TAU_M) - Math.Exp(-s / TAU_S);

	/// <summary>
	/// Eq. 4 derived<br/>
	/// ε = The contribution of one synaptic terminal on the potential of the postsynaptic neuron caused by a presynaptic spike
	/// </summary>
	/// <param name="s">t - spikeT - delay</param>
	/// <returns></returns>
	private static double EpsilonDerived(double s) =>
		s < 0 ? 0 : -Math.Exp(-s / TAU_M) / TAU_M + Math.Exp(-s / TAU_S) / TAU_S;

	/// <summary>
	/// Eq. 5<br/>
	/// η = Exponential decay, modeling the behavioral of the neuron after it fired
	/// </summary>
	/// <param name="s">t - spikeT - delay</param>
	/// <returns></returns>
	private static double Eta(double s) =>
		s < 0 ? 0 : -Math.Exp(-s / TAU_R);

	/// <summary>
	/// Eq. 5 derived<br/>
	/// η = Exponential decay, modeling the behavioral of the neuron after it fired
	/// </summary>
	/// <param name="s">t - spikeT - delay</param>
	/// <returns></returns>
	private static double EtaDerived(double s) =>
		s < 0 ? 0 : Math.Exp(-s / TAU_R) / TAU_R;

	public void Fire(double time) =>
		Spikes.Add(time);

	/// <summary>
	/// Eq. 2
	/// </summary>
	/// <param name="time"></param>
	public void Forward(double time)
	{
		if (ComputePotential(time) > THRESHOLD)
		{
			Fire(time);
		}
	}

	/// <summary>
	/// Eq. 3
	/// </summary>
	/// <param name="time"></param>
	/// <returns></returns>
	private double ComputePotential(double time)
	{
		double result = 0;

		// Foreach syn in
		for (int synI = 0; synI < SynapsesIn.Length; synI++)
		{
			// Foreach spike of the pre neuron of the syn in
			for (int spikeI = 0; spikeI < SynapsesIn[synI].NeuronPre.Spikes.Count; spikeI++)
			{
				result += SynapsesIn[synI].Weight * Epsilon(time - SynapsesIn[synI].NeuronPre.Spikes[spikeI] - SynapsesIn[synI].Delay);
			}
		}

		for (int spikeI = 0; spikeI < Spikes.Count; spikeI++)
		{
			result += Eta(time - Spikes[spikeI]);
		}

		return result;
	}

	/// <summary>
	/// Eq. 9
	/// </summary>
	/// <param name="learningRate"></param>
	public void ComputeDeltaWeights(double learningRate)
	{
		Parallel.For(0, SynapsesIn.Length, new ParallelOptions() { MaxDegreeOfParallelism = Environment.ProcessorCount }, synI =>
		{
			for (int spikeI = 0; spikeI < Spikes.Count; spikeI++)
			{
				SynapsesIn[synI].WeightDelta -= learningRate * ComputeDeDt(Spikes[spikeI]) * ComputeDtDw(SynapsesIn[synI], Spikes[spikeI]);
			}
		});
	}

	/// <summary>
	/// Eq. 10<br/>
	/// dt/dw = Derivation of the firing time with respect to the weight
	/// </summary>
	/// <param name="synapse"></param>
	/// <param name="spikeThis"></param>
	/// <returns></returns>
	private double ComputeDtDw(Synapse synapse, double spikeThis) =>
		-ComputeDuDw(synapse, spikeThis) / ComputeDuDt(spikeThis);

	/// <summary>
	/// Eq. 11<br/>
	/// du/dw = Derivation of the potential with respect to the weight
	/// </summary>
	/// <param name="synapse"></param>
	/// <param name="spikeThis"></param>
	/// <returns></returns>
	private double ComputeDuDw(Synapse synapse, double spikeThis)
	{
		double result = 0;

		// Foreach spike from the pre neuron of the syn
		for (int spikeI = 0; spikeI < synapse.NeuronPre.Spikes.Count; spikeI++)
		{
			result += Epsilon(spikeThis - synapse.NeuronPre.Spikes[spikeI] - synapse.Delay);
		}

		// Foreach spike from this neuron
		for (int spikeI = 0; spikeI < Spikes.Count; spikeI++)
		{
			if (Spikes[spikeI] < spikeThis)
			{
				result += -EtaDerived(spikeThis - Spikes[spikeI]) * ComputeDtDw(synapse, Spikes[spikeI]);
			}
		}

		return result;
	}

	/// <summary>
	/// Eq. 12<br/>
	/// du/dt = Derivation of the potential with respect to the spike time
	/// </summary>
	/// <param name="spikeThis"></param>
	/// <returns></returns>
	private double ComputeDuDt(double spikeThis)
	{
		double result = 0;

		// Foreach syn in
		for (int synI = 0; synI < SynapsesIn.Length; synI++)
		{
			// Foreach spike of the pre neuron of the syn in
			for (int spikeI = 0; spikeI < SynapsesIn[synI].NeuronPre.Spikes.Count; spikeI++)
			{
				result += SynapsesIn[synI].Weight * EpsilonDerived(spikeThis - SynapsesIn[synI].NeuronPre.Spikes[spikeI] - SynapsesIn[synI].Delay);
			}
		}

		// Foreach spike from this neuron
		for (int spikeI = 0; spikeI < Spikes.Count; spikeI++)
		{
			if (Spikes[spikeI] < spikeThis)
			{
				result += EtaDerived(spikeThis - Spikes[spikeI]);
			}
		}

		if (result < 0.1)
		{
			result = 0.1;
		}

		return result;
	}

	/// <summary>
	/// Eq. 13<br/>
	/// dE/dt = Derivation of the network error with respect to the spike time
	/// </summary>
	/// <param name="spikeThis"></param>
	/// <returns></returns>
	private double ComputeDeDt(double spikeThis)
	{
		if (FixedOutput > 0)
		{
			if (spikeThis == Spikes[0])
			{
				return spikeThis - FixedOutput;
			}
		}

		double result = 0;

		// Foreach neuron post
		for (int neuronI = 0; neuronI < NeuronsPost.Length; neuronI++)
		{
			// Foreach spike of that neuron post
			for (int spikeI = 0; spikeI < NeuronsPost[neuronI].Spikes.Count; spikeI++)
			{
				if (NeuronsPost[neuronI].Spikes[spikeI] > spikeThis)
				{
					result += NeuronsPost[neuronI].ComputeDeDt(NeuronsPost[neuronI].Spikes[spikeI]) * ComputeDposttDt(spikeThis, NeuronsPost[neuronI], NeuronsPost[neuronI].Spikes[spikeI]);
				}
			}
		}
		return result;
	}

	/// <summary>
	/// Eq. 14<br/>
	/// dt/dt = Derivation of the post with respect to the pre synaptic spike time
	/// </summary>
	/// <param name="spikeThis"></param>
	/// <param name="neuronPost"></param>
	/// <param name="spikePost"></param>
	/// <returns></returns>
	private double ComputeDposttDt(double spikeThis, Neuron neuronPost, double spikePost) =>
		-ComputeDpostuDt(spikeThis, neuronPost, spikePost) / neuronPost.ComputeDuDt(spikePost);

	/// <summary>
	/// Eq. 15<br/>
	/// du/dt = Derivation of the potential during a postsynaptic spike with respect to the pre synaptic spike time 
	/// </summary>
	/// <param name="spikeThis"></param>
	/// <param name="neuronPost"></param>
	/// <param name="spikePost"></param>
	/// <returns></returns>
	private double ComputeDpostuDt(double spikeThis, Neuron neuronPost, double spikePost)
	{
		double result = 0;

		// Foreach syn in of the given neuron post
		for (int synI = 0; synI < neuronPost.SynapsesIn.Length; synI++)
		{
			if (neuronPost.SynapsesIn[synI].NeuronPre == this)
			{
				result -= neuronPost.SynapsesIn[synI].Weight * EpsilonDerived(spikePost - spikeThis - neuronPost.SynapsesIn[synI].Delay);
			}
		}

		// Foreach spike from the given neuron post
		for (int spikeI = 0; spikeI < neuronPost.Spikes.Count; spikeI++)
		{
			if (neuronPost.Spikes[spikeI] < spikePost)
			{
				result -= EtaDerived(spikePost - neuronPost.Spikes[spikeI]) * ComputeDposttDt(spikeThis, neuronPost, neuronPost.Spikes[spikeI]);
			}
		}
		return result;
	}

}
