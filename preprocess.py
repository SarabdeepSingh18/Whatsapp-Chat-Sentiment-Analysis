import re
import pandas as pd

def preprocess(data):
    # If data is provided as a list of lines, join them into one string.
    if isinstance(data, list):
        data = "\n".join(line.strip() for line in data)

    # Clean up Unicode formatting issues like narrow spaces or RTL marks
    data = data.replace('\u202f', ' ').replace('\u200e', '').replace('\u200b', '')

    # Regex to support both 24-hour and 12-hour (with am/pm) formats
    pattern = r'\[(\d{2}/\d{2}/\d{2}),\s*(\d{1,2}:\d{2}(?::\d{2})?\s?(?:am|pm|AM|PM)?)\]\s*(.*?):\s*(.*?)(?=\n\[|\Z)'

    matches = re.findall(pattern, data, re.DOTALL)

    if not matches:
        print("No matches found in the data. Please check the data format.")
        return pd.DataFrame()

    dates, times, senders, messages = zip(*matches)

    # Convert to datetime with flexible parsing
    datetimes = []
    for d, t in zip(dates, times):
        try:
            dt = pd.to_datetime(f"{d} {t}", dayfirst=True, errors='coerce')
        except ValueError:
            dt = pd.NaT
        datetimes.append(dt)

    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'time': times,
        'sender': senders,
        'message': messages,
        'datetime': datetimes
    })

    # Drop rows where datetime couldn't be parsed
    df = df.dropna(subset=['datetime'])

    # Add derived columns
    df['year'] = df['datetime'].dt.year
    df['only_date'] = df['datetime'].dt.date
    df['month_num'] = df['datetime'].dt.month
    df['month'] = df['datetime'].dt.month_name()
    df['day_name'] = df['datetime'].dt.day_name()
    df['day'] = df['datetime'].dt.day
    df['hour'] = df['datetime'].dt.hour
    df['minute'] = df['datetime'].dt.minute

    # Period bucketing
    period = []
    for hour in df['hour']:
        if hour == 23:
            period.append(f"{hour}-00")
        elif hour == 0:
            period.append("00-1")
        else:
            period.append(f"{hour}-{hour+1}")
    df['period'] = period

    return df
