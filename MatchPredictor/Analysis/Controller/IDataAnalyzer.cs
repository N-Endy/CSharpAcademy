using Shared.Models;

namespace Analysis.Controller;
public interface IDataAnalyzer
{
    IEnumerable<MatchData> StraightWin();
    IEnumerable<MatchData> OverTwoGoals();
    IEnumerable<MatchData> UnderTwoGoals();
    IEnumerable<MatchData> Draw();
    IEnumerable<MatchData> BothTeamsScore();
    IEnumerable<MatchData> OverThreeGoals();
}