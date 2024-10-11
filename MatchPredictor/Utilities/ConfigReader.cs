using Microsoft.Extensions.Configuration;


using Shared.Models;

namespace Utilities;
public class ConfigReader
{
    private readonly IConfiguration _configuration;

    public ConfigReader(IConfiguration configuration)
    {
        _configuration = configuration;
    }

    public EmailSettings GetEmailSettings()
    {
        return new EmailSettings
        {
            SmtpServer = _configuration["Email:SmtpServer"] ?? throw new ArgumentNullException("Email:SmtpServer"),
            SmtpPort = int.TryParse(_configuration["Email:SmtpPort"], out var smtpPort) ? smtpPort : throw new ArgumentException("Email:SmtpPort"),
            Password = _configuration["Email:Password"] ?? throw new ArgumentNullException("Email:Password"),
            FromAddress = _configuration["Email:FromAddress"] ?? throw new ArgumentNullException("Email:FromAddress"),
            ToAddresses = _configuration.GetSection("Email:ToAddress").Get<List<string>>() ?? []
        };
    }
}