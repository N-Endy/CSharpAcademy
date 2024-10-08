using WebScraper;
using Utilities;
using Data.Controller;
using Data.Service;
using Data.Repository;
using Data;
using Export.Service;
using Export.Controller;
using Analysis.Controller;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Configuration;
using Email;
using Email.Service;
using Email.Controller;

var builder = Host.CreateApplicationBuilder(args);

ConfigureServices(builder.Services, builder.Configuration);

var host = builder.Build();

await RunProgram(host.Services);

void ConfigureServices(IServiceCollection services, ConfigurationManager configuration)
{
    services.AddSingleton<IWebScraper, SeleniumWebScraper>();
    services.AddSingleton<IMatchDataRepository, MatchDataRepository>();
    services.AddSingleton<IDataSeeder, DataSeeder>();
    services.AddSingleton<IExcelToDatabaseOperation, ExcelToDatabaseOperation>();
    services.AddSingleton<IDataAnalyzer, DataAnalyzer>();
    services.AddSingleton<IExcelExporter, ExcelExporter>();
    services.AddSingleton<ICsvExporter, CsvExporter>();
    services.AddSingleton<IExportService, ExportService>();
    services.AddSingleton<IEmailService, EmailService>();
    services.AddSingleton<IEmailController, EmailController>();
    services.AddSingleton<ConfigReader>();
    services.AddDbContext<FootballContext>();
}

async Task RunProgram(IServiceProvider serviceProvider)
{
    var scraper = serviceProvider.GetRequiredService<IWebScraper>();

    Logger.Log("[lime]Starting Football Match Predictor...[/]");

    await scraper.DownloadExcelFile();
    Logger.Log("[lime]Excel file downloaded and processed successfully.[/]");

    IExcelToDatabaseOperation dataController = serviceProvider.GetRequiredService<IExcelToDatabaseOperation>();
    await dataController.RunOperation();

    Logger.Log("[lime]Data analysis and export started...[/]");

    IExportService exportService = serviceProvider.GetRequiredService<IExportService>();
    exportService.ExportToCsv();
    Logger.Log("[lime]Data analysis and export completed successfully.[/]");

    IEmailController emailController = serviceProvider.GetRequiredService<IEmailController>();
    await emailController.SendEmailAsync();
    Logger.Log("[lime]Email notification sent successfully.[/]");
}