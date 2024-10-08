namespace Data.Controller;
public interface IExcelToDatabaseOperation
{
    Task RunOperation();
    Task ConvertExcelToDatabase();
    Task CreateDatabase();
    Task DeleteDatabase();
}