using System.Net;
using System.Net.Mail;
using Email.Service;
using Utilities;

namespace Email;
public class EmailService : IEmailService
{
    private readonly ConfigReader _configReader;
    private readonly string _resourcesFolder;

    public EmailService(ConfigReader configReader)
    {
        _configReader = configReader;

        string projectDirectory = Directory.GetParent(Directory.GetCurrentDirectory())?.FullName ?? string.Empty;
        _resourcesFolder = Path.Combine(projectDirectory, "Resources");
    }

    public async Task SendEmailWithAttachment()
    {
        var emailSettings = _configReader.GetEmailSettings();

        try
        {
            using var client = new SmtpClient(emailSettings.SmtpServer, emailSettings.SmtpPort)
            {
                UseDefaultCredentials = false,
                Credentials = new NetworkCredential(emailSettings.FromAddress, emailSettings.Password),
                EnableSsl = true,
                Timeout = 10000
            };

            var mail = new MailMessage(emailSettings.FromAddress, emailSettings.ToAddress)
            {
                Subject = "Match Data with Attachments",
                Body = "Please find the attached match data files.",
                IsBodyHtml = false
            };

            AddCsvAttachments(mail);

            // Implement retry logic
            int maxRetries = 3;
            for (int i = 0; i < maxRetries; i++)
            {
                try
                {
                    await client.SendMailAsync(mail);
                    Logger.Log($"Email sent to {emailSettings.ToAddress} successfully");
                    return;
                }
                catch (SmtpException ex)
                {
                    Logger.Log($"[yellow]Attempt {i + 1} failed: {ex.Message}[/]");
                    if (i == maxRetries - 1) throw;
                    await Task.Delay(5000); // Wait 5 seconds before retrying
                }
            }
        }
        catch (SmtpException smtpEx)
        {
            Logger.Log($"[red]SMTP error occurred: {smtpEx.Message}[/]");
            Logger.Log($"[red]Status Code: {smtpEx.StatusCode}[/]");
            Logger.Log($"[red]Stack trace: {smtpEx.StackTrace}[/]");
        }
        catch (Exception ex)
        {
            Logger.Log($"[red]An error occurred while sending email: {ex.Message}[/]");
            Logger.Log($"[red]Stack trace: {ex.StackTrace}[/]");
        }
    }

    private void AddCsvAttachments(MailMessage mail)
    {
        var csvFiles = Directory.GetFiles(_resourcesFolder, "*.csv");

        foreach (var csvFile in csvFiles)
        {
            mail.Attachments.Add(new Attachment(csvFile));
        }
    }
}