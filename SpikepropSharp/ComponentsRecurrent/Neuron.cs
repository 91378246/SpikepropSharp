using static SpikepropSharp.ComponentsRecurrent.Network;

namespace SpikepropSharp.ComponentsRecurrent
{
    public sealed class Neuron
    {
        public List<Connection> incoming_connections = new();
        public List<Connection> outgoing_connections = new();
        public List<Spike> spikes = new();
        public const double tau = 4.0;
        public double u_m;
        public double u_s;
        public double last_update = 0.0;
        public double last_spike = 0.0;
        public double clamped = 0.0;
        public string key = "";

        public Neuron(string key_ = "neuron")
        {
            this.key = key_;
        }

        public class Spike
        {
            public double time;
            public double dE_dt;

            public Spike(double time, double dE_dt)
            {
                this.time = time;
                this.dE_dt = dE_dt;
            }
        }

        public void clear()
        {
            spikes.Clear();
            for (int conI = 0; conI < incoming_connections.Count; conI++)
            {
                incoming_connections[conI].dpre_spikes.Clear();

                for (int synI = 0; synI < incoming_connections[conI].synapses.Count; synI++)
                {
                    incoming_connections[conI].synapses[synI].dt_dws.Clear();
                    incoming_connections[conI].synapses[synI].u_m = 0;
                    incoming_connections[conI].synapses[synI].u_s = 0;
                }
            }
            u_m = u_s = last_update = last_spike = 0;
        }

        public void update_potentials(double time)
        {
            var update = Math.Exp(-(time - last_update) / tau);
            u_m *= update;
            u_s *= update * update;
            last_update = time;
        }

        // compute exact future firing time (should document the derivation of this formula)
        public double compute_future_spike()
        {
            double D = u_m * u_m + 4 * u_s;
            if (D > 0)
            {
                double expdt = (-u_m - Math.Sqrt(D)) / (2 * u_s);
                if (expdt > 0)
                {
                    double predict_spike = -Math.Log(expdt) * tau;
                    if (predict_spike > 0)
                    {
                        return last_update + predict_spike;
                    }
                }
            }
            return 0.0; // should perhaps use std::optional
        }

        // Forward propagate and store gradients for backpropagation. For now
        // keeping this as one function. To be refactored.
        public void incoming_spike(double time, double weight)
        {
            update_potentials(time);
            u_m += weight;
            u_s -= weight;
        }

        public void spike(double time)
        {
            update_potentials(time);

            store_gradients(time);
            spikes.Add(new Spike(time, 0));
            last_spike = time;
            u_m -= 1.0;
        }

        public void store_gradients(double spike_time)
        {
            double du_dt = -(u_m + u_s * 2) / tau;
            if (du_dt < .1) // handling discontinuity circumstance 1 Sec 3.2
            {
                du_dt = .1;
            }

            double spike_diff_exp_m = Math.Exp(-(spike_time - last_spike) / tau);
            double spike_diff_exp_s = spike_diff_exp_m * spike_diff_exp_m;

            for (int conI = 0; conI < incoming_connections.Count; conI++)
            {
                for (int spikeI = 0; spikeI < incoming_connections[conI].dpre_spikes.Count; spikeI++)
                {
                    incoming_connections[conI].dpre_spikes[spikeI].u_m *= spike_diff_exp_m;
                    incoming_connections[conI].dpre_spikes[spikeI].u_s *= spike_diff_exp_s;
                }

                for (int synI = 0; synI < incoming_connections[conI].synapses.Count; synI++)
                {
                    // update_synapse_potentials
                    incoming_connections[conI].synapses[synI].u_m *= spike_diff_exp_m;
                    incoming_connections[conI].synapses[synI].u_s *= spike_diff_exp_s;
                    for (int spikeI = 0; spikeI < Math.Min(incoming_connections[conI].neuron.spikes.Count, incoming_connections[conI].dpre_spikes.Count); spikeI++)
                    {
                        double s = spike_time - incoming_connections[conI].neuron.spikes[spikeI].time - incoming_connections[conI].synapses[synI].delay;
                        if (incoming_connections[conI].neuron.spikes[spikeI].time + incoming_connections[conI].synapses[synI].delay > last_spike && s >= 0) // pre-spike came between previous and this spike
                        {
                            var u_m1 = Math.Exp(-s / tau);
                            var u_s1 = -u_m1 * u_m1;
                            incoming_connections[conI].synapses[synI].u_m += u_m1;
                            incoming_connections[conI].synapses[synI].u_s += u_s1;
                            incoming_connections[conI].dpre_spikes[spikeI].u_m += incoming_connections[conI].synapses[synI].weight * u_m1;
                            incoming_connections[conI].dpre_spikes[spikeI].u_s += incoming_connections[conI].synapses[synI].weight * u_s1;
                        }
                    }
                    incoming_connections[conI].synapses[synI].dt_dws.Add(-(incoming_connections[conI].synapses[synI].u_m + incoming_connections[conI].synapses[synI].u_s) / du_dt);
                    incoming_connections[conI].synapses[synI].u_m -= spike_diff_exp_m / tau * incoming_connections[conI].synapses[synI].dt_dws.Last();
                }
                foreach (var dpre_spike in incoming_connections[conI].dpre_spikes)
                {
                    dpre_spike.dpostts.Add(-(dpre_spike.u_m + dpre_spike.u_s * 2) / tau / du_dt);
                    dpre_spike.u_m -= spike_diff_exp_m / tau * dpre_spike.dpostts.Last();
                }
            }
        }

        // Compute needed weight changes, and backpropagate to incoming
        // connections.
        public void backprop_spike(int spike_i, double learning_rate)
        {
            Spike spike = spikes[spike_i];
            if (clamped > 0 && spike_i == 0) // output neuron
            {
                spike.dE_dt = spike.time - clamped;
            }
            for (int conI = 0; conI < incoming_connections.Count; conI++)
            {
                for (int synI = 0; synI < incoming_connections[conI].synapses.Count; synI++)
                {
                    incoming_connections[conI].synapses[synI].delta_weight -= learning_rate * spike.dE_dt * incoming_connections[conI].synapses[synI].dt_dws.ElementAt(spike_i);
                }

                for (int spikeI = 0; spikeI < Math.Min(incoming_connections[conI].neuron.spikes.Count, incoming_connections[conI].dpre_spikes.Count); spikeI++)
                {
                    incoming_connections[conI].neuron.spikes[spikeI].dE_dt += spike.dE_dt * incoming_connections[conI].dpre_spikes[spikeI].dpostts.ElementAt(spike_i);
                }
            }
        }

    }
}
