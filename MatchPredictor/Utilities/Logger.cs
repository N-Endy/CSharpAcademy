using Spectre.Console;

namespace Utilities
{
    public static class Logger
    {
        public static void Log(string message)
        {
            AnsiConsole.MarkupLine(message);
        }
    }
}