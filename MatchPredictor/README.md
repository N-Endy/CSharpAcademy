# Football Match Predictor

## Overview
Football Match Predictor is a sophisticated C# application designed to automate the process of downloading, analyzing, and reporting football match predictions. It leverages web scraping, data analysis, and email notifications to provide users with valuable insights into upcoming football matches.

## Features
* Automated web scraping of match prediction data
* Data analysis for various betting scenarios
* CSV export of analyzed data
* Email notifications with attached reports
* Configurable settings via appsettings.json

## System Requirements
* .NET 6.0 or higher
* Chrome browser (for Selenium WebDriver)
* Internet connection

## Installation
1. Clone the repository:
`git clone https://github.com/yourusername/football-match-predictor.git`
2. Navigate to the project directory:
`cd MatchPredictions`
3. Restore NuGet packages:
`dotnet restore`
4. Update the appsettings.json file with your email configuration.

## Configuration
Edit the appsettings.json file to configure the application:

```
{
    "ConnectionStrings": {
        "DefaultConnection": "Data Source=MatchAnalysis.db"
    },
    "Email": {
        "SmtpServer": "smtp.gmail.com",
        "SmtpPort": 587,
        "Password": "your-app-password",
        "FromAddress": "your-email@gmail.com",
        "ToAddress": "recipient@example.com"
    }
}
```

## Usage
Run the application using the following command:
`dotnet run --project MatchPredictions`

The application will perform the following steps:
1. Download the latest match prediction data
2. Process and analyze the data
3. Export analysis results to CSV files
4. Send an email with the exported files attached

## Project Structure
* WebScraper: Handles downloading of prediction data
* Data: Manages database operations and data seeding
* Analysis: Performs data analysis on match predictions
* Export: Exports analyzed data to CSV format
* Email: Handles email composition and sending
* Utilities: Contains utility classes like ConfigReader and Logger

## Key Components
* SeleniumWebScraper: Downloads prediction data using Selenium WebDriver
* DataAnalyzer: Analyzes match data for various betting scenarios
* CsvExporter: Exports analyzed data to CSV files
* EmailService: Composes and sends emails with attachments
* ConfigReader: Reads configuration from appsettings.json

# Customization
To add new analysis criteria:
1. Extend the IDataAnalyzer interface in Analysis/Controller/IDataAnalyzer.cs
2. Implement the new method in Analysis/Controller/DataAnalyzer.cs
3. Add a corresponding export method in Export/Controller/CsvExporter.cs

You can also implement machine learning using ML.NET

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request with your changes.