import re
import dateparser

def clean_text(text):
    if not text:
        return ""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def parse_date(date_string):
    if not date_string:
        return None
    try:
        dt = dateparser.parse(date_string)
        if dt:
            return dt.strftime('%Y-%m-%d %H:%M:%S')
    except:
        pass
    return date_string # Return original if parsing fails
