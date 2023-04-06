namespace SpikepropSharp.Api.Endpoints;

internal static class Data
{
    public static void RegisterEndopoints(WebApplication app)
    {
        app.MapGet("/inputSample", () =>
        {
            IEnumerable<object> forecast = null!;
            return forecast;
        });
    }

    private static Dictionary<string, object> GetInputSample()
    {
        const int sampleIndex = 3;
        throw new NotImplementedException();
    }
}

