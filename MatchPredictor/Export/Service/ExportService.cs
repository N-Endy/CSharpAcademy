using Shared.Models;
using Export.Controller;

namespace Export.Service;
public class ExportService : IExportService
{
    private readonly IExcelExporter _excelExporter;
    private readonly ICsvExporter _csvExporter;

    public ExportService(IExcelExporter excelExporter, ICsvExporter csvExporter)
    {
        _excelExporter = excelExporter;
        _csvExporter = csvExporter;
    }

    public void ExportToCsv()
    {
        _csvExporter.ExportStraightWin();
        _csvExporter.ExportDraw();
        _csvExporter.ExportOverTwoGoals();
        _csvExporter.ExportUnderTwoGoals();
        _csvExporter.ExportBothTeamScore();
        _csvExporter.ExportOverThreeGoals();
    }

    public void ExportToExcel(IEnumerable<MatchData> matches, string filePath)
    {
        //_excelExporter.Export(matches, filePath);
    }
}