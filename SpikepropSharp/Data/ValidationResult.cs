using Microsoft.Win32;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text.Json;

namespace SpikepropSharp.Data
{
	public sealed class ValidationResult
	{
		public double[] Errors { get; }
		public double[] EegRaw { get; }
		public Dictionary<double, bool> EcgSignalSpikesTrain { get; }
		public List<Prediction> Predictions { get; }

		public ValidationResult(double[] errors, double[] ecgRaw, Dictionary<double, bool> ecgSignalSpikesTrain)
		{
			Errors = errors;
			EegRaw = ecgRaw;
			EcgSignalSpikesTrain = ecgSignalSpikesTrain;
			Predictions = new();
		}

		public void Save(bool plot)
		{
			string validationFilePath = $"{DateTime.Now:yy.MM.dd.HH.mm.ss}-val_res.json";
			File.WriteAllText(validationFilePath, JsonSerializer.Serialize(this));

			Console.WriteLine($"Validation file saved to {validationFilePath}");
			if (plot)
			{
				if (!RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
				{
					Console.WriteLine("Automated Python path finding is only available on Windows");
				}
				else
				{
					RunPythonScript(Path.GetFullPath(validationFilePath));
				}
			}
		}

		private static async void RunPythonScript(string dataFilePath)
		{
			string pythonFilePath = Path.GetFullPath("Data/visualize.py");
			ProcessStartInfo start = new()
			{
				FileName = GetPythonPath(),
				Arguments = $"{pythonFilePath} {dataFilePath}",
				UseShellExecute = false,
				WorkingDirectory = ""
,
				RedirectStandardOutput = true
			};

			using Process process = Process.Start(start);
			using StreamReader reader = process.StandardOutput;
			string output = await reader.ReadToEndAsync();

			if (!string.IsNullOrWhiteSpace(output))
			{
				Console.WriteLine(output);
			}
		}

		[System.Diagnostics.CodeAnalysis.SuppressMessage("Interoperability", "CA1416:Validate platform compatibility", Justification = "Only works on Windows")]
		private static string? GetPythonPath(string requiredVersion = "", string maxVersion = "")
		{
			string[] possiblePythonLocations = new string[] {
				@"HKLM\SOFTWARE\Python\PythonCore\",
				@"HKCU\SOFTWARE\Python\PythonCore\",
				@"HKLM\SOFTWARE\Wow6432Node\Python\PythonCore\"
			};

			//Version number, install path
			Dictionary<string, string> pythonLocations = new();

			foreach (string possibleLocation in possiblePythonLocations)
			{
				string regKey = possibleLocation[..4];
				string actualPath = possibleLocation[5..];
				RegistryKey theKey = regKey == "HKLM" ? Registry.LocalMachine : Registry.CurrentUser;
				RegistryKey? theValue = theKey.OpenSubKey(actualPath);

				foreach (string val in theValue?.GetSubKeyNames() ?? Array.Empty<string>())
				{
					if (theValue?.OpenSubKey(val) is RegistryKey productKey)
					{
						try
						{
							string? pythonExePath = productKey.OpenSubKey("InstallPath")?.GetValue("ExecutablePath")?.ToString();
							// string? pythonExePath = productKey.OpenSubKey("InstallPath")?.GetValue("")?.ToString();

							if (!string.IsNullOrWhiteSpace(pythonExePath))
							{
								Console.WriteLine("Found python with version " + val);
								pythonLocations.Add(val.ToString(), pythonExePath);
							}
						}
						catch { }
					}
				}
			}

			if (pythonLocations.Count > 0)
			{
				Version desiredVersion = new(requiredVersion == "" ? "0.0.1" : requiredVersion);
				Version maxPVersion = new(maxVersion == "" ? "999.999.999" : maxVersion);

				string highestVersionPath = "";

				foreach (KeyValuePair<string, string> pVersion in pythonLocations)
				{
					int index = pVersion.Key.IndexOf("-");
					string formattedVersion = index > 0 ? pVersion.Key[..index] : pVersion.Key;

					Version thisVersion = new(formattedVersion);
					int comparison = desiredVersion.CompareTo(thisVersion);
					int maxComparison = maxPVersion.CompareTo(thisVersion);

					if (comparison <= 0)
					{
						if (maxComparison >= 0)
						{
							desiredVersion = thisVersion;
							highestVersionPath = pVersion.Value;
						}
					}
				}

				return highestVersionPath;
			}

			Console.WriteLine("[ERROR] Python not found");
			return null;
		}
	}
}
