using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace Data.Migrations
{
    /// <inheritdoc />
    public partial class InitialMigration : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.CreateTable(
                name: "MatchData",
                columns: table => new
                {
                    Id = table.Column<int>(type: "INTEGER", nullable: false)
                        .Annotation("Sqlite:Autoincrement", true),
                    Date = table.Column<string>(type: "TEXT", nullable: true),
                    HomeTeam = table.Column<string>(type: "TEXT", nullable: true),
                    AwayTeam = table.Column<string>(type: "TEXT", nullable: true),
                    HomeWin = table.Column<float>(type: "REAL", nullable: true),
                    Draw = table.Column<float>(type: "REAL", nullable: true),
                    AwayWin = table.Column<float>(type: "REAL", nullable: true),
                    OverOneGoal = table.Column<float>(type: "REAL", nullable: true),
                    OverTwoGoals = table.Column<float>(type: "REAL", nullable: true),
                    OverThreeGoals = table.Column<float>(type: "REAL", nullable: true),
                    OverFourGoals = table.Column<float>(type: "REAL", nullable: true),
                    UnderOneGoal = table.Column<float>(type: "REAL", nullable: true),
                    UnderTwoGoals = table.Column<float>(type: "REAL", nullable: true),
                    UnderThreeGoals = table.Column<float>(type: "REAL", nullable: true),
                    UnderFourGoals = table.Column<float>(type: "REAL", nullable: true)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_MatchData", x => x.Id);
                });
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropTable(
                name: "MatchData");
        }
    }
}
