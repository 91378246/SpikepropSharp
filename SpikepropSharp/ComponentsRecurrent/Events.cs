using static SpikepropSharp.ComponentsRecurrent.Network;

namespace SpikepropSharp.ComponentsRecurrent
{
    public sealed class Events
    {
        public PriorityQueue<SynapseSpike, double> synapse_spikes = new();
        public List<NeuronSpike> predicted_spikes = new List<NeuronSpike>();
        public List<SpikeRecord> actual_spikes = new List<SpikeRecord>();

        public bool active()
        {
            return !(predicted_spikes.Count == 0 && synapse_spikes.Count == 0);
        }

        // Process the next event in the queue. This might be either a spike
        // ariving at a neuron (named synapse_spike), or a neuron spiking.
        public void process_event()
        {
            if (!active())
            {
                return;
            }
            // which one first
            // compute_earliest_neuron_spike
            NeuronSpike neuron_spike = predicted_spikes.MaxBy(p => p.time);
            Neuron updated_neuron;
            // bit of ugly logic to determine which type of event is first
            if (predicted_spikes.Count == 0 || (synapse_spikes.Count != 0 && synapse_spikes.Peek().time < neuron_spike.time))
            { // process synapse
                var synapse_spike = synapse_spikes.Dequeue();
                updated_neuron = synapse_spike.neuron;
                // find neuron's existing fire-time
                //ORIGINAL LINE: neuron_spike = std::ranges::find_if(predicted_spikes, [updated_neuron](const auto& n) noexcept ({return updated_neuron == n.neuron;}));
                neuron_spike = predicted_spikes.FirstOrDefault(p => p.neuron == updated_neuron);
                // update neuron
                updated_neuron.incoming_spike(synapse_spike.time, synapse_spike.weight);
            }
            else
            { // process neuron
              // update post_synapses
                updated_neuron = neuron_spike.neuron;
                // record
                actual_spikes.Add(new SpikeRecord(updated_neuron, neuron_spike.time, updated_neuron.spikes.Count));
                foreach (Connection outgoing_connection in updated_neuron.outgoing_connections)
                {
                    foreach (Synapse post_synapse in outgoing_connection.synapses)
                    {
                        synapse_spikes.Enqueue(new SynapseSpike(outgoing_connection.post_neuron, post_synapse.weight, neuron_spike.time + post_synapse.delay), double.MaxValue - (neuron_spike.time + post_synapse.delay));
                    }
                }
                // update neuron itself, including gradients
                updated_neuron.spike(neuron_spike.time);
            }
            // remove affected neuron's spike
            if (neuron_spike != predicted_spikes.Last())
            {
                predicted_spikes.Remove(neuron_spike);
            }

            // check for new spike
            double future_spike = updated_neuron.compute_future_spike();
            if (future_spike > 0)
            {
                predicted_spikes.Add(new NeuronSpike(updated_neuron, future_spike));
            }
        }

        public void backprop(List<SpikeRecord> actual_spikes, double learning_rate)
        {
            actual_spikes.Reverse();
            foreach (var spike in actual_spikes)
            {
                spike.neuron.backprop_spike(spike.index, learning_rate);
            }
        }
    }
}
