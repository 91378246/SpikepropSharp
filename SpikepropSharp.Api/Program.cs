using SpikepropSharp.Api.Endpoints;

WebApplicationBuilder builder = WebApplication.CreateBuilder(args);

builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();

WebApplication app = builder.Build();
if (app.Environment.IsDevelopment())
{
    app.UseSwagger();
    app.UseSwaggerUI();
}

Random rnd = new(1);
SpikepropSharp.Components.Network network = new(rnd);
Data.RegisterEndopoints(app);
Network.RegisterEndopoints(app, network);

app.Run();