using Shared.Models;
using Data.Repository;
using OfficeOpenXml;
using Utilities;

namespace Data.Service;
public class DataSeeder : IDataSeeder
{
    private readonly string _filePath;
    private readonly IMatchDataRepository _matchDataRepository;

    public DataSeeder(IMatchDataRepository matchDataRepository)
    {
        _matchDataRepository = matchDataRepository;

        string projectDirectory = Directory.GetParent(Directory.GetCurrentDirectory())?.FullName ?? string.Empty;
        _filePath = Path.Combine(projectDirectory, "Resources/predictions.xlsx");
    }

    public async Task ExtractMatchDatasetToDatabase()
    {
        var extractedData = new List<MatchData>();

        try
        {
            // If filePath is not found
            if (!File.Exists(_filePath))
            {
                Logger.Log("[bold][red]Excel file not found at path: " + _filePath + "[/][/]");
                return;
            }

            
            // If the Excel file is empty, return
            if (new FileInfo(_filePath).Length == 0)
            {
                Logger.Log("[bold][yellow]Excel file is empty.[/][/]");
                return;
            }

            // Read data from the downloaded Excel file and extract relevant information
            using var package = new ExcelPackage(new FileInfo(_filePath));

            if (package.Workbook.Worksheets.Count > 0)
            {
                var worksheet = package.Workbook.Worksheets[0];

                if (worksheet.Dimension == null || worksheet.Cells.Any(cell => cell.Value == null))
                {
                    Logger.Log("[bold][yellow]Excel file has no data.[/][/]");
                }

                var rowCount = worksheet.Dimension?.Rows ?? 0;

                for (var row = 2; row <= rowCount; row++)
                {
                    // Some flat fields may be null and that's okay
                    var matchData = new MatchData
                    {
                        Date = worksheet.Cells[row, 5].Value?.ToString(),
                        League = worksheet.Cells[row, 4].Value?.ToString(),
                        HomeTeam = worksheet.Cells[row, 2].Value?.ToString(),
                        AwayTeam = worksheet.Cells[row, 3].Value?.ToString(),
                        HomeWin = double.TryParse(worksheet.Cells[row, 6].Value?.ToString(), out double homeWin) ? homeWin : 0,
                        Draw = double.TryParse(worksheet.Cells[row, 7].Value?.ToString(), out double draw) ? draw : 0,
                        AwayWin = double.TryParse(worksheet.Cells[row, 8].Value?.ToString(), out double awayWin) ? awayWin : 0,
                        OverOneGoal = double.TryParse(worksheet.Cells[row, 14].Value?.ToString(), out double overOneGoal) ? overOneGoal : 0,
                        OverTwoGoals = double.TryParse(worksheet.Cells[row, 18].Value?.ToString(), out double overTwoGoals) ? overTwoGoals : 0,
                        OverThreeGoals = double.TryParse(worksheet.Cells[row, 22].Value?.ToString(), out double overThreeGoals) ? overThreeGoals : 0,
                        OverFourGoals = double.TryParse(worksheet.Cells[row, 24].Value?.ToString(), out double overFourGoals) ? overFourGoals : 0,
                        UnderOneGoal = double.TryParse(worksheet.Cells[row, 30].Value?.ToString(), out double underOneGoal) ? underOneGoal : 0,
                        UnderTwoGoals = double.TryParse(worksheet.Cells[row, 34].Value?.ToString(), out double underTwoGoals) ? underTwoGoals : 0,
                        UnderThreeGoals = double.TryParse(worksheet.Cells[row, 38].Value?.ToString(), out double underThreeGoals) ? underThreeGoals : 0,
                        UnderFourGoals = double.TryParse(worksheet.Cells[row, 40].Value?.ToString(), out double underFourGoals) ? underFourGoals : 0
                    };

                    extractedData.Add(matchData);
                }
            }
            else
            {
                Logger.Log("[bold][yellow]Excel file has no worksheets.[/][/]");
            }

            // Add extracted data to the database
            await _matchDataRepository.AddMatchData(extractedData);
        }
        catch (Exception e)
        {
            Logger.Log($"[bold][red]Error processing Excel file: {e.Message}[/][/]");
        }
    }
}