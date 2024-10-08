using Shared.Models;

namespace Data.Repository;
public interface IMatchDataRepository
{
    Task AddMatchData(List<MatchData> matchData);
    Task<List<MatchData>> GetMatchData();
    void DeleteDatabase();
    void CreateDatabase();
}