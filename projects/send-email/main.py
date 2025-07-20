import smtplib
import getpass
from email.mime.text import MIMEText

def send_email():
    sender_addr = 'risky.bkp7@gmail.com'
    pw = getpass.getpass()
    subject = 'Learn.Inspire.Grow'
    msg = '''
        Hi,
        Hope you're doing well. This is a test email message as part of a Coding Mafia project. Learn, inspire, grow!
        
        Thanks,
        Jomin Jose
    '''
    
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(sender_addr, pw)
    
    msg = MIMEText(msg)
    msg['Subject'] = subject
    msg['From'] = sender_addr
    msg['To'] = 'risky.bkp7@gmail.com'
    msg.set_param('importance', 'high value')
    recipients = 'risky.bkp7@gmail.com'
    
    server.sendmail(sender_addr, recipients, msg.as_string())
    
send_email()