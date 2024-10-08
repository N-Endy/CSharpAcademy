using OpenQA.Selenium;
using OpenQA.Selenium.Chrome;
using OpenQA.Selenium.Support.UI;
using Utilities;

namespace WebScraper;
public class SeleniumWebScraper : IWebScraper
{
    const string _downloadUrl = "https://www.sports-ai.dev/predictions";
    private readonly string _downloadFolder;

    public SeleniumWebScraper()
    {
        string projectDirectory = Directory.GetParent(Directory.GetCurrentDirectory())?.FullName ?? string.Empty;
        _downloadFolder = Path.Combine(projectDirectory, "Resources");
        Directory.CreateDirectory(_downloadFolder); // Ensure the directory exists
        Logger.Log($"[lime]Download folder set to: {_downloadFolder}[/]");
    }

    public async Task DownloadExcelFile()
    {
        try
        {
            var chromeOptions = new ChromeOptions();
            chromeOptions.AddUserProfilePreference("download.default_directory", _downloadFolder);
            chromeOptions.AddUserProfilePreference("download.prompt_for_download", false);
            chromeOptions.AddUserProfilePreference("download.directory_upgrade", true);
            chromeOptions.AddUserProfilePreference("safebrowsing.enabled", true);
            chromeOptions.AddArgument("--no-sandbox");
            chromeOptions.AddArgument("--disable-dev-shm-usage");
            chromeOptions.AddArgument("--remote-debugging-port=9222");
            chromeOptions.AddArgument("--headless");

            var service = ChromeDriverService.CreateDefaultService();
            //service.SuppressInitialDiagnosticInformation = true;
            service.HideCommandPromptWindow = true;

            // Delete previous file if found
            DeletePreviousFile();

            using var driver = new ChromeDriver(service, chromeOptions, TimeSpan.FromSeconds(120));
            driver.Navigate().GoToUrl(_downloadUrl);

            var wait = new WebDriverWait(driver, TimeSpan.FromSeconds(30));
            var downloadButton = wait.Until(d =>
            {
                var element = d.FindElement(By.XPath("//*[@id='__next']/div/main/div[1]/div[1]/div[1]/button"));
                return element.Displayed && element.Enabled ? element : null;
            });
            downloadButton?.Click();
            Logger.Log("[lime]Download button clicked, waiting for file to be downloaded...[/]");

            await CheckFileIsDownloaded();
        }
        catch (Exception ex)
        {
            Logger.Log($"[red]An error occurred: {ex.Message}[/]");
            Logger.Log($"[red]Stack trace: {ex.StackTrace}[/]");
        }
    }

    public async Task CheckFileIsDownloaded()
    {
        string fileName = "predictions.xlsx";
        string[] possiblePaths =
        [
            Path.Combine(_downloadFolder, fileName),
                Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "Downloads", fileName),
                Path.Combine(Directory.GetCurrentDirectory(), fileName)
        ];

        int maxWaitTime = 60; // Increase wait time to 60 seconds
        int waitInterval = 1000; // Check every second
        int totalWaitTime = 0;

        while (totalWaitTime < maxWaitTime * 1000)
        {
            foreach (var path in possiblePaths)
            {
                if (File.Exists(path))
                {
                    Logger.Log($"[lime]File found at: {path}[/]");
                    if (path != Path.Combine(_downloadFolder, fileName))
                    {
                        File.Move(path, Path.Combine(_downloadFolder, fileName), true);
                        Logger.Log($"[lime]File moved to: {Path.Combine(_downloadFolder, fileName)}[/]");
                    }
                    return;
                }
            }
            await Task.Delay(waitInterval);
            totalWaitTime += waitInterval;
        }

        throw new FileNotFoundException($"File {fileName} not found in any expected location after {maxWaitTime} seconds.");
    }

    public void DeletePreviousFile()
    {
        string filePath = Path.Combine(_downloadFolder, "predictions.xlsx");
        if (File.Exists(filePath))
        {
            File.Delete(filePath);
            Logger.Log($"[lime]Previous file deleted: {filePath}[/]");
        }
    }
}