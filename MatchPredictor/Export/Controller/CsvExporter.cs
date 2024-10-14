using System.Globalization;
using Analysis.Controller;
using Shared.Models;
using CsvHelper;

namespace Export.Controller;
public class CsvExporter : ICsvExporter
{
    private readonly IDataAnalyzer _dataAnalyzer;
    private readonly string _resourcesFolder;

    public CsvExporter(IDataAnalyzer dataAnalyzer)
    {
        _dataAnalyzer = dataAnalyzer;

        // Set the path to the Resources folder in the project
        string projectDirectory = Directory.GetParent(Directory.GetCurrentDirectory())?.FullName ?? string.Empty;
        _resourcesFolder = Path.Combine(projectDirectory, "Resources");

        // Ensure the Resources folder exists
        Directory.CreateDirectory(_resourcesFolder);
    }

    public void ExportBothTeamScore()
    {
        var bothTeamsScoreMatches = _dataAnalyzer.BothTeamsScore();
        WriteToCsv(bothTeamsScoreMatches, "BothTeamsScore.csv");
    }

    public void ExportDraw()
    {
        var drawMatches = _dataAnalyzer.Draw();
        WriteToCsv(drawMatches, "Draw.csv");
    }

    public void ExportOverThreeGoals()
    {
        var overThreeGoalsMatches = _dataAnalyzer.OverThreeGoals();
        WriteToCsv(overThreeGoalsMatches, "OverThreeGoals.csv");
    }

    public void ExportOverTwoGoals()
    {
        var overTwoGoalsMatches = _dataAnalyzer.OverTwoGoals();
        WriteToCsv(overTwoGoalsMatches, "OverTwoGoals.csv");
    }

    public void ExportStraightWin()
    {
        var straightWinMatches = _dataAnalyzer.StraightWin();
        WriteToCsv(straightWinMatches, "StraightWin.csv");
    }

    public void ExportUnderTwoGoals()
    {
        var underTwoGoalsMatches = _dataAnalyzer.UnderTwoGoals();
        WriteToCsv(underTwoGoalsMatches, "UnderTwoGoals.csv");
    }

    public void ExportDrawOrUnderTwoGoals()
    {
        var drawOrUnderTwoGoalsMatches = _dataAnalyzer.DrawOrUnder2();
        WriteToCsv(drawOrUnderTwoGoalsMatches, "DrawOrUnderTwoGoals.csv");
    }

    private void WriteToCsv(IEnumerable<MatchData> matches, string fileName)
    {
        string filePath = Path.Combine(_resourcesFolder, fileName);

        using var writer = new StreamWriter(filePath);
        using var csv = new CsvWriter(writer, CultureInfo.InvariantCulture);
        csv.WriteRecords(matches);

        Console.WriteLine($"CSV file '{fileName}' written successfully to {filePath}");
    }
}