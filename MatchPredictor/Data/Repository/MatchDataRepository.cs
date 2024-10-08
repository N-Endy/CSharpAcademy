using Shared.Models;
using Microsoft.EntityFrameworkCore;

namespace Data.Repository;
public class MatchDataRepository : IMatchDataRepository
{
    private readonly FootballContext _context;

    public MatchDataRepository(FootballContext context)
    {
        _context = context;
    }

    public void CreateDatabase()
    {
        _context.Database.EnsureCreated();
    }

    public void DeleteDatabase()
    {
        _context.Database.EnsureDeleted();
    }

    public async Task AddMatchData(List<MatchData> matchData)
    {
        await _context.MatchData.AddRangeAsync(matchData);
        await _context.SaveChangesAsync();
    }

    public async Task<List<MatchData>> GetMatchData()
    {
        return await _context.MatchData.ToListAsync();
    }

}