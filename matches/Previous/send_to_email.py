import pandas as pd
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email import encoders
from datetime import datetime

excel_file = './predictions.xlsx'
df = pd.read_excel(excel_file)

################# EXCEL FILE FUNCTION ######################
# ... (keep the existing code for creating the Excel file)
# Read the Excel file

# Get today's date
today = datetime.now().date()

# Convert 'date' column to datetime, keeping both date and time
df['date'] = pd.to_datetime(df['date'], dayfirst=True)

# Filter matches for today's date
today_matches = df[df['date'].dt.date == today]

# Filter matches based on the specified conditions
filtered_matches = today_matches[
    ((today_matches['1x2_h'].between(0.6, 0.7)) | (today_matches['1x2_a'].between(0.6, 0.7))) &
    (today_matches['o_2.5'] > 0.6)
].copy()

# Select relevant columns
columns_to_keep = ['date', 'league', 'home', 'away', '1x2_h', '1x2_a', 'o_2.5']
result_df = filtered_matches[columns_to_keep]

# Sort the DataFrame by 'o_2.5' in descending order
result_df = result_df.sort_values(by='o_2.5', ascending=False)

# Format the date column to the desired string format
result_df['date'] = result_df['date'].dt.strftime('%d.%m.%Y %H:%M')

# Format probability columns as percentages with one decimal place
for col in ['1x2_h', '1x2_a', 'o_2.5']:
    result_df[col] = result_df[col].apply(lambda x: f"{(x*100):.1f}%")

# Save the result to an Excel file
excel_output_file = 'Today_matches.xlsx'
result_df.to_excel(excel_output_file, index=False)

print(f"Extracted {len(result_df)} matches for today and saved to {excel_output_file}")


################# OVER 3.5 GOALS #################

# Filter matches where either team has >70% probability of under 2.5 goals
high_probability_matches = today_matches[
    (today_matches['o_3.5'] > 0.6)
].copy()

# Select relevant columns
columns_to_keep = ['date', 'league', 'home', 'away', 'o_3.5']
match_df = high_probability_matches[columns_to_keep]

# Sort the DataFrame by 'u_2.5' in descending order
match_df = match_df.sort_values(by='date', ascending=True)

match_df['date'] = match_df['date'].dt.strftime('%d.%m.%Y %H:%M')

# Format 'u_2.5' as percentage with one decimal place
match_df.loc[:, 'o_3.5'] = match_df['o_3.5'].apply(lambda x: f"{(x*100):.1f}%")

# Save the result to a CSV file
csv_output_file = 'Over_3.5_matches.csv'
match_df.to_csv(csv_output_file, index=False)

print(f"Extracted {len(match_df)} matches and saved to {csv_output_file}")



#################### EMAIL METHOD ##################
# After saving the Excel file, add the following code:

def send_email(sender_email, sender_password, recipient_list, subject, body, excel_file, csv_file):
    for recipient in recipient_list:
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = recipient
        msg['Subject'] = subject

        msg.attach(MIMEText(body, 'plain'))

        with open(excel_file, "rb") as attachment:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(attachment.read())
        
        encoders.encode_base64(part)
        part.add_header(
            "Content-Disposition",
            f"attachment; filename= {excel_file}",
        )
        msg.attach(part)
        
        with open(csv_file, "rb") as attachment:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(attachment.read())
        
        encoders.encode_base64(part)
        part.add_header(
            "Content-Disposition",
            f"attachment; filename= {csv_file}",
        )
        msg.attach(part)

        try:
            with smtplib.SMTP('smtp.gmail.com', 587) as server:
                server.starttls()
                server.login(sender_email, sender_password)
                server.send_message(msg)
            print(f"Email sent successfully to {recipient}")
        except Exception as e:
            print(f"Failed to send email to {recipient}. Error: {str(e)}")

# Email configuration
sender_email = "okafornelson9@gmail.com"
sender_password = "hjvg vbut imok ishw"
recipient_list = ["okafornel@gmail.com", "okaford1@gmail.com", "Okaforjerek2030@gmail.com"]
subject = "Excel Matches Report"
body = "Please find attached the Excel Matches report."

# Send the email with the attachment
# send_email(sender_email, sender_password, recipient_list, subject, body, excel_output_file, csv_output_file)
