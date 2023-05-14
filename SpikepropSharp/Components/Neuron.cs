namespace SpikepropSharp.Components;

public sealed class Neuron
{
	private const double TAU_M = 4;
	private const double TAU_S = 2;
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
	/// Eq. 4
	/// </summary>
	/// <param name="s"></param>
	/// <returns></returns>
	private static double Epsilon(double s) =>
		s < 0 ? 0 : Math.Exp(-s / TAU_M) - Math.Exp(-s / TAU_S);

	private static double EpsilonDerived(double s) =>
		s < 0 ? 0 : -Math.Exp(-s / TAU_M) / TAU_M + Math.Exp(-s / TAU_S) / TAU_S;

	/// <summary>
	/// Eq. 5
	/// </summary>
	/// <param name="s"></param>
	/// <returns></returns>
	private static double Eta(double s) =>
			s < 0 ? 0 : -Math.Exp(-s / TAU_R);

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
		if (ComputeU(time) > THRESHOLD)
		{
			Fire(time);
		}
	}

	/// <summary>
	/// Eq. 3
	/// </summary>
	/// <param name="time"></param>
	/// <returns></returns>
	private double ComputeU(double time)
	{
		double u = 0;
		// Foreach syn in
		for (int synI = 0; synI < SynapsesIn.Length; synI++)
		{
			// Foreach spike of the pre neuron of the syn in
			for (int spikeI = 0; spikeI < SynapsesIn[synI].NeuronPre.Spikes.Count; spikeI++)
			{
				u += SynapsesIn[synI].Weight * Epsilon(time - SynapsesIn[synI].NeuronPre.Spikes[spikeI] - SynapsesIn[synI].Delay);
			}
		}

		for (int spikeI = 0; spikeI < Spikes.Count; spikeI++)
		{
			u += Eta(time - Spikes[spikeI]);
		}

		return u;
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
	/// dt/dw
	/// Eq. 10
	/// </summary>
	/// <param name="synapse"></param>
	/// <param name="spikeThis"></param>
	/// <returns></returns>
	private double ComputeDtDw(Synapse synapse, double spikeThis)
	{
		return -ComputeDuDw(synapse, spikeThis) / ComputeDuDt(spikeThis);
	}

	/// <summary>
	/// du/dw
	/// Eq. 11
	/// </summary>
	/// <param name="synapse"></param>
	/// <param name="spikeThis"></param>
	/// <returns></returns>
	private double ComputeDuDw(Synapse synapse, double spikeThis)
	{
		double duDw = 0;

		// Foreach spike from the pre neuron of the syn
		for (int spikeI = 0; spikeI < synapse.NeuronPre.Spikes.Count; spikeI++)
		{
			duDw += Epsilon(spikeThis - synapse.NeuronPre.Spikes[spikeI] - synapse.Delay);
		}

		// Foreach spike from this neuron
		for (int spikeI = 0; spikeI < Spikes.Count; spikeI++)
		{
			if (Spikes[spikeI] < spikeThis)
			{
				duDw += -EtaDerived(spikeThis - Spikes[spikeI]) * ComputeDtDw(synapse, Spikes[spikeI]);
			}
		}

		return duDw;
	}

	/// <summary>
	/// du/dt
	/// Eq. 12
	/// </summary>
	/// <param name="spikeThis"></param>
	/// <returns></returns>
	private double ComputeDuDt(double spikeThis)
	{
		double duDt = 0;

		// Foreach syn in
		for (int synI = 0; synI < SynapsesIn.Length; synI++)
		{
			// Foreach spike of the pre neuron of the syn in
			for (int spikeI = 0; spikeI < SynapsesIn[synI].NeuronPre.Spikes.Count; spikeI++)
			{
				duDt += SynapsesIn[synI].Weight * EpsilonDerived(spikeThis - SynapsesIn[synI].NeuronPre.Spikes[spikeI] - SynapsesIn[synI].Delay);
			}
		}

		// Foreach spike from this neuron
		for (int spikeI = 0; spikeI < Spikes.Count; spikeI++)
		{
			if (Spikes[spikeI] < spikeThis)
			{
				duDt += EtaDerived(spikeThis - Spikes[spikeI]);
			}
		}

		if (duDt < 0.1)
		{
			duDt = 0.1;
		}

		return duDt;
	}

	/// <summary>
	/// dE/dt
	/// Eq. 13
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

		double deDt = 0;

		// Foreach neuron post
		for (int neuronI = 0; neuronI < NeuronsPost.Length; neuronI++)
		{
			// Forach spike of thaht neuron post
			for (int spikeI = 0; spikeI < NeuronsPost[neuronI].Spikes.Count; spikeI++)
			{
				if (NeuronsPost[neuronI].Spikes[spikeI] > spikeThis)
				{
					deDt += NeuronsPost[neuronI].ComputeDeDt(NeuronsPost[neuronI].Spikes[spikeI]) * ComputeDposttDt(spikeThis, NeuronsPost[neuronI], NeuronsPost[neuronI].Spikes[spikeI]);
				}
			}
		}
		return deDt;
	}

	/// <summary>
	/// dt/dt
	/// Eq. 14
	/// </summary>
	/// <param name="spikeThis"></param>
	/// <param name="neuronPost"></param>
	/// <param name="spikePost"></param>
	/// <returns></returns>
	private double ComputeDposttDt(double spikeThis, Neuron neuronPost, double spikePost) =>
		-ComputeDpostuDt(spikeThis, neuronPost, spikePost) / neuronPost.ComputeDuDt(spikePost);

	/// <summary>
	/// du/dt
	/// Eq. 15
	/// </summary>
	/// <param name="spikeThis"></param>
	/// <param name="neuronPost"></param>
	/// <param name="spikePost"></param>
	/// <returns></returns>
	private double ComputeDpostuDt(double spikeThis, Neuron neuronPost, double spikePost)
	{
		double dpostuDt = 0;

		// Foreach syn in of the given neuron post
		for (int synI = 0; synI < neuronPost.SynapsesIn.Length; synI++)
		{
			if (neuronPost.SynapsesIn[synI].NeuronPre == this)
			{
				dpostuDt -= neuronPost.SynapsesIn[synI].Weight * EpsilonDerived(spikePost - spikeThis - neuronPost.SynapsesIn[synI].Delay);
			}
		}

		// Foreach spike from the given neuron post
		for (int spikeI = 0; spikeI < neuronPost.Spikes.Count; spikeI++)
		{
			if (neuronPost.Spikes[spikeI] < spikePost)
			{
				dpostuDt -= EtaDerived(spikePost - neuronPost.Spikes[spikeI]) * ComputeDposttDt(spikeThis, neuronPost, neuronPost.Spikes[spikeI]);
			}
		}
		return dpostuDt;
	}

}
