using Shared.Models;

namespace Export.Service;
public interface IExportService
{
    void ExportToExcel(IEnumerable<MatchData> matches, string filePath);
    void ExportToCsv();
}