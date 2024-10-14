namespace Export.Controller;
public interface ICsvExporter
{
    void ExportStraightWin();
    void ExportDraw();
    void ExportOverTwoGoals();
    void ExportUnderTwoGoals();
    void ExportBothTeamScore();
    void ExportOverThreeGoals();
    void ExportDrawOrUnderTwoGoals();
}