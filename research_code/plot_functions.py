import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
def snr_entropy_rmse(max_decomposition_level, ec_snr_list, ec_rmse_list, ec_appro_entropy_list, temp_snr_list, temp_rmse_list, temp_appro_entropy_list, ph_snr_list, ph_rmse_list, ph_appro_entropy_list):
    level = range(1, max_decomposition_level + 1)
    plt.plot(level, ec_snr_list, marker = '.')
    plt.xlabel('Level', y = 1.02)
    plt.ylabel('Electrical Conductivity SNR(dB)')
    plt.grid(True)
    plt.show()
    
    plt.plot(level, ec_rmse_list, marker = '.')
    plt.xlabel('Level')
    plt.ylabel('Electrical Conductivity RMSE')
    plt.grid(True)
    plt.show()

    
    plt.plot(level, temp_snr_list, marker = '.')
    plt.xlabel('Level')
    plt.ylabel('Temperature SNR(dB)')
    plt.grid(True)
    plt.show()
    
    plt.plot(level, temp_rmse_list, marker = '.')
    plt.xlabel('Level')
    plt.ylabel('Temperature RMSE')
    plt.grid(True)
    plt.show()
    

    
    plt.plot(level, ph_snr_list, marker = '.')
    plt.xlabel('Level')
    plt.ylabel('pH SNR(dB)')
    plt.grid(True)
    plt.show()

    plt.plot(level, ph_rmse_list, marker = '.')
    plt.xlabel('Level')
    plt.ylabel('pH RMSE')
    plt.grid(True)

    plt.show()

def original_data(temp, ph, ec,first_nonzero_index):
    file_path = 'C:\\Users\\USER\\Desktop\\nycu-data\\log(0527)-backup.csv'
    data = pd.read_csv(file_path)
    x = pd.to_datetime(data.iloc[:, 0])
    num_ticks_to_display = 8
    x_min = x.min()
    x_max = x.max()
    x_range = x_max - x_min
    interval = x_range.total_seconds() / (num_ticks_to_display - 1)
    x_ticks_to_display = [x_min + pd.Timedelta(seconds=i * interval) for i in range(num_ticks_to_display)]
    x_tick_labels = [tick.strftime('%H:%M') for tick in x_ticks_to_display]
    plt.figure(figsize=(10, 6))

    plt.subplot(3, 1, 1)
    plt.plot(x[first_nonzero_index:data.shape[0] - 1], temp, color = 'black')
    plt.title('Temperature')
    plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=1))
    plt.xticks(x_ticks_to_display, x_tick_labels, rotation=45)

    plt.subplot(3, 1, 2)
    plt.plot(x[first_nonzero_index:data.shape[0] - 1], ph, color = 'red')
    plt.title('pH')
    plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=1))
    plt.xticks(x_ticks_to_display, x_tick_labels, rotation=45)


    plt.subplot(3, 1, 3)
    plt.plot(x[first_nonzero_index:data.shape[0] - 1], ec, color = 'green')
    plt.title('EC')
    plt.xlabel('Time')
    plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=1))
    plt.xticks(x_ticks_to_display, x_tick_labels, rotation=45)

    plt.tight_layout()

    plt.show()