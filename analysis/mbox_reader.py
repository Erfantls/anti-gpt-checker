# code inspired by https://gist.github.com/benwattsjones/060ad83efd2b3afc8b229d41f9b246c4

import mailbox
import re
from datetime import datetime, timezone

import bs4

from models.email import EmailGmail, EmailPayload


class GmailMboxMessage:

    def __init__(self, email_data: mailbox.mboxMessage):
        if not isinstance(email_data, mailbox.mboxMessage):
            raise TypeError('Variable must be type mailbox.mboxMessage')
        self.email_data = email_data

    def parse_to_email_model(self) -> EmailGmail:
        email_from = self.email_data['From']
        from_address = email_from.split('<')[1].split('>')[0] if len(email_from.split('<')) > 1 else email_from.replace("<", "").replace(">", "")
        from_name = email_from.split('<')[0].strip() if len(email_from.split('<')) > 1 else None
        to_address = self.email_data['To']
        email_labels = self.email_data['X-Gmail-Labels']
        date_str = self.email_data['Date']
        try:
            date = datetime.strptime(date_str, '%a, %d %b %Y %H:%M:%S %z')
        except ValueError:
            try:
                # Handle case where the timezone is 'GMT' instead of an offset (+HHMM or -HHMM)
                date = datetime.strptime(date_str, '%a, %d %b %Y %H:%M:%S GMT').replace(tzinfo=timezone.utc)
            except ValueError:
                try:
                    # Handle case where the timezone is 'GMT' or '+0000 (UTC)'
                    date_str_cleaned = re.sub(r'\(\w+\)', '', date_str).strip()
                    date = datetime.strptime(date_str_cleaned, '%a, %d %b %Y %H:%M:%S %z').replace(tzinfo=timezone.utc)
                except ValueError:
                    try:
                        date = datetime.strptime(date_str, '%d %b %Y %H:%M:%S %z')
                    except ValueError:
                        date_str_cleaned = (re.sub(r'\(\w+\+\d+:\d+\)', '', date_str).strip()).split(' (')[0]
                        date = datetime.strptime(date_str_cleaned, '%a, %d %b %Y %H:%M:%S %z')

        subject = self.email_data['Subject']
        if not isinstance(subject, str):
            subject = str(subject)
        body = self.email_data._payload
        if not isinstance(body, str):
            is_html = None
            inner_body = []
            self._parse_multipart_body(body, inner_body)
            body = inner_body

        is_html = 'html' in self.email_data['Content-Type']
        return EmailGmail(
            from_address=from_address,
            from_name=from_name,
            to_address=to_address,
            email_labels=email_labels,
            date=date,
            subject=subject,
            body=body,
            is_html=is_html,
            is_spam=None,
            is_ai_generated=None
        )


    def _parse_multipart_body(self, body_to_check, collected_list):
        if self.email_data.is_multipart() and isinstance(body_to_check, list):
            for part in body_to_check:
                inner_is_html = 'html' in '\t'.join([' '.join(subpart) for subpart in part._headers])
                inner_body = part._payload
                if not isinstance(inner_body, str):
                    inner_list = []
                    self._parse_multipart_body(inner_body, inner_list)
                    collected_list.append(EmailPayload(is_html=None, body=inner_list))
                else:
                    collected_list.append(EmailPayload(is_html=inner_is_html, body=inner_body))
        else:
            collected_list.append(EmailPayload(is_html=None, body=None))
