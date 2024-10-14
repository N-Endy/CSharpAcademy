using Shared.Models;
using Data.Repository;

namespace Analysis.Controller;
public class DataAnalyzer : IDataAnalyzer
{
    private readonly IMatchDataRepository _repository;

    public DataAnalyzer(IMatchDataRepository repository)
    {
        _repository = repository;
    }

    public IEnumerable<MatchData> BothTeamsScore()
    {
        return _repository.GetMatchData().Result.Where(x => (x.HomeWin > 0.35 && x.AwayWin > 0.35 && x.OverTwoGoals > 0.6) || (x.HomeWin > x.Draw && x.AwayWin > x.Draw && x.OverTwoGoals > 0.6) || (x.HomeWin > x.Draw && x.AwayWin > x.Draw && x.OverThreeGoals > 0.55) || (x.HomeWin > 0.35 && x.AwayWin > 0.35 && x.OverThreeGoals > 0.55));
    }

    public IEnumerable<MatchData> Draw()
    {
        return _repository.GetMatchData().Result.Where(x => (x.Draw > 0.35 && x.HomeWin < 0.34 && x.AwayWin < 0.34 && x.UnderTwoGoals > 0.65) || (x.HomeWin < 0.34 && x.AwayWin < 0.34 && x.UnderThreeGoals > 0.75) || x.Draw > 3.6);
    }

    public IEnumerable<MatchData> OverThreeGoals()
    {
        return _repository.GetMatchData().Result.Where(x => x.OverThreeGoals > 0.65 || x.OverFourGoals > 4.5);
    }

    public IEnumerable<MatchData> OverTwoGoals()
    {
        return _repository.GetMatchData().Result.Where(x => (x.HomeWin > 0.35 && x.AwayWin > 0.35 && x.OverTwoGoals > 0.65) || ((x.HomeWin + x.Draw ) < x.AwayWin && x.OverTwoGoals > 0.65) || ((x.AwayWin + x.Draw ) < x.HomeWin && x.OverTwoGoals > 0.65));
    }

    public IEnumerable<MatchData> StraightWin()
    {
        return _repository.GetMatchData().Result.Where(x => (x.HomeWin > 0.65 && x.OverTwoGoals > 0.6) || (x.AwayWin > 0.67 && x.OverTwoGoals > 0.6));
    }

    public IEnumerable<MatchData> UnderTwoGoals()
    {
        return _repository.GetMatchData().Result.Where(x => x.HomeWin < 0.35 && x.AwayWin < 0.35 && x.UnderTwoGoals > 0.65);
    }

    public IEnumerable<MatchData> DrawOrUnder2()
    {
        return _repository.GetMatchData().Result.Where(x => x.HomeWin > 0.3 && x.Draw > 0.3 && x.AwayWin > 0.3);
    }
}