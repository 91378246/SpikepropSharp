namespace SpikepropSharp.Api.Endpoints;

internal static class Network
{
    public static void RegisterEndopoints(WebApplication app, Components.Network network)
    {
        app.MapGet("/size", () => GetSize(network));
        app.MapGet("/parameters", () => GetParameters(network));
    }

    /// <summary>
    /// /size
    /// </summary>
    /// <param name="network"></param>
    /// <returns></returns>
    private static int[] GetSize(Components.Network network)
    {
        int[] result = new int[3];
        for (int i = 0; i < result.Length; i++)
        {
            result[i] = network.Layers[i].Length;
        }

        return result;
    }

    /// <summary>
    /// /parameters
    /// </summary>
    /// <param name="network"></param>
    /// <returns></returns>
    private static Dictionary<string, object> GetParameters(Components.Network network)
    {
        Dictionary<string, object> parameters = new();
        parameters["Network"] = null!;
        parameters["Neuron"] = null!;

        return parameters;
    }
}

