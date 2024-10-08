using Shared.Models;
using Microsoft.EntityFrameworkCore;

namespace Data;
public class FootballContext : DbContext
{
    public DbSet<MatchData> MatchData { get; set; }

    protected override void OnConfiguring(DbContextOptionsBuilder optionsBuilder)
    {
        optionsBuilder.UseSqlite("Data Source=MatchAnalysis.db");
    }
}