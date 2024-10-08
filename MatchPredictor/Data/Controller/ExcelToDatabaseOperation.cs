using Data.Repository;
using Data.Service;
using Utilities;

namespace Data.Controller;
public class ExcelToDatabaseOperation : IExcelToDatabaseOperation
{
    private readonly IDataSeeder _dataSeeder;
    private readonly IMatchDataRepository _repository;

    public ExcelToDatabaseOperation(IDataSeeder dataSeeder, IMatchDataRepository repository)
    {
        _dataSeeder = dataSeeder;
        _repository = repository;
    }

    public async Task RunOperation()
    {
        await DeleteDatabase();
        await CreateDatabase();
        await ConvertExcelToDatabase();
    }

    public async Task ConvertExcelToDatabase()
    {
        Logger.Log("[lime] Reading Excel file...[/]");
        Logger.Log("[lime] Starting Excel to database conversion...[/]");
        Logger.Log("[lime] Converting Excel to database...[/]");
        await _dataSeeder.ExtractMatchDatasetToDatabase();
        Logger.Log("[lime] Excel to database conversion completed.[/]");
    }

    public Task CreateDatabase()
    {
        Logger.Log("[lime] Creating new database...[/]");
        _repository.CreateDatabase();
        return Task.CompletedTask;
    }

    public Task DeleteDatabase()
    {
        Logger.Log("[lime] Deleting existent database...[/]");
        _repository.DeleteDatabase();
        return Task.CompletedTask;
    }
}