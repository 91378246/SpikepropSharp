namespace SpikepropSharp.Utility
{
	internal static class ConsoleExtensions
	{
		public static void WriteLine(string text, ConsoleColor color) 
		{
			ConsoleColor currentColor = Console.ForegroundColor;
			Console.ForegroundColor = color;
			Console.WriteLine(text);
			Console.ForegroundColor = currentColor;
		}
	}
}
