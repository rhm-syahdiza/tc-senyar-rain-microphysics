import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime

datdir='data/202511'

df1 = pd.read_csv(f'{datdir}/l0251125.csv', header=1)
df2 = pd.read_csv(f'{datdir}/l0251126.csv', header=1)
df3 = pd.read_csv(f'{datdir}/l0251127.csv', header=1)
df4 = pd.read_csv(f'{datdir}/l0251128.csv', header=1)
df5 = pd.read_csv(f'{datdir}/l0251129.csv', header=1)
df6 = pd.read_csv(f'{datdir}/l0251130.csv', header=1)

df = pd.concat([df1, df2, df3, df4, df5, df6], ignore_index=True)
df['time'] = pd.to_datetime(df['time'], format='%m/%d/%y %I:%M:%S %p')
df = df.sort_values('time')

# PLOT DATA
fig, ax = plt.subplots(figsize=(12, 4))

ax.plot(df['time'], df['SpdAvg10m'], color='teal', linewidth=1.5, label='10-min Wind Speed Avg')
ax.set_title('MAWS Kototabang - 10-Minute Wind Speed Average\n(25 - 30 Nov 2025)', fontsize=14)
ax.set_xlabel('Date & Time', fontsize=12)
ax.set_ylabel('Wind Speed (m/s)', fontsize=12)

# ax2=ax.twinx()
# ax2.plot(df['time'], df['DirAvg10m'], '.', color='tab:red', linewidth=1.5, label='10-min Wind Speed Avg')
# ax2.set_ylabel('Wind Direction', fontsize=12)

ax.set_xlim(datetime.datetime(2025,11,25,0,0,0), datetime.datetime(2025,11,30,23,59,00))
ax.set_ylim(0.0, 5.0)

ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
# ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %H:%M'))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %H'))

plt.xticks(rotation=90, ha='right')

ax.xaxis.set_major_formatter(mdates.DateFormatter('%d%H'))

ax.grid(True, linestyle='--', alpha=0.6)
ax.legend(loc='upper right')

# plt.tight_layout()
plt.subplots_adjust(bottom=0.18)

plt.show()
# plt.savefig('maws_kototabang_wind_speed.png', dpi=300)
