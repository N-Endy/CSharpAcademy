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
        var emailSettingsSection = _configuration.GetSection("Email") ?? throw new ArgumentNullException("Email section not found in configuration");
        
        var emailSettings = emailSettingsSection.Get<EmailSettings>() ?? throw new ArgumentNullException("Unable to bind Email settings from configuration");
        
        return emailSettings;
    }
}