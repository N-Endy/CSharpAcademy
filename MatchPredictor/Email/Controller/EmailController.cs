
using Email.Service;

namespace Email.Controller;
public class EmailController : IEmailController
{
    private readonly IEmailService _emailService;

    public EmailController(IEmailService emailService)
    {
        _emailService = emailService;
    }

    public async Task SendEmailAsync()
    {
        await _emailService.SendEmailWithAttachment();
    }
}