import fastf1 as ff1
import pandas as pd
import numpy as np

def extractRacePace(session, driver):
    laps = session.laps.pick_driver(driver)
    stints = laps['Stint'].unique()
    stint_paces = []

    for stint in stints:
        stint_laps = laps[laps['Stint'] == stint]
        valid_laps = stint_laps[(stint_laps['LapTime'] < stint_laps['LapTime'].shift(1) + pd.Timedelta(seconds=1)) &
                                (stint_laps['LapTime'] < stint_laps['LapTime'].shift(-1) + pd.Timedelta(seconds=1))]
        if not valid_laps.empty:
            average_pace = valid_laps['LapTime'].mean().total_seconds()
            stint_paces.append(average_pace)

    if stint_paces:
        driver_pace = np.mean(stint_paces)
    else:
        driver_pace = np.nan

    return driver_pace

def extractFastestLap(session, driver):
    laps = session.laps.pick_driver(driver)
    if not laps.empty:
        fastest_lap = laps['LapTime'].min().total_seconds()
    else:
        fastest_lap = np.nan

    return fastest_lap

def extractMaxVelocity(quali_session, driver):
    laps = quali_session.laps.pick_driver(driver)
    if not laps.empty:
        max_speed = laps['SpeedST'].max()
    else:
        max_speed = np.nan

    return max_speed

def get_session_data(year, grand_prix, session_name):
    try:
        session = ff1.get_session(year, grand_prix, session_name)
        session.load()
        if not hasattr(session, 'laps') or session.laps.empty:
            print(f"No lap data available for {session_name} session.")
            return None
    except ValueError as e:
        print(f"Session type '{session_name}' does not exist for {grand_prix} {year}. Skipping this session.")
        return None
    except Exception as e:
        print(f"Error loading session {session_name} for {grand_prix} {year}: {e}")
        return None
    return session

def collect_basic_data_for_season(year, grand_prix):
    quali_session = get_session_data(year, grand_prix, 'Q')
    sprint_session = get_session_data(year, grand_prix, 'S')
    race_session = get_session_data(year, grand_prix, 'R')

    if not quali_session or not sprint_session or not race_session:
        print(f"Skipping {grand_prix} {year} due to missing session data.")
        return None

    data = []
    processed_drivers = set()

    for session in [sprint_session, quali_session]:
        if session is None:
            continue
        for driver in session.drivers:
            if driver in processed_drivers:
                continue
            processed_drivers.add(driver)
            try:
                start_position = race_session.results.loc[race_session.results['DriverNumber'] == driver, 'GridPosition'].values[0]
                finish_position = race_session.results.loc[race_session.results['DriverNumber'] == driver, 'Position'].values[0]

                race_pace = extractRacePace(sprint_session, driver) if sprint_session else np.nan
                fastest_lap = extractFastestLap(sprint_session, driver) if sprint_session else np.nan
                max_velocity = extractMaxVelocity(quali_session, driver) if quali_session else np.nan

                data.append([year, grand_prix, driver, race_pace, fastest_lap, max_velocity, start_position, finish_position])
            except Exception as e:
                print(f"Error processing data for driver {driver} in {grand_prix} {year}: {e}")

    df = pd.DataFrame(data, columns=['Year', 'GrandPrix', 'Driver', 'Race_pace', 'Fastest_Lap', 'Max_Velocity', 'StartPosition', 'FinishPosition'])
    
    df['Race_pace_avg'] = df['Race_pace'].mean()
    df['Fastest_Lap_avg'] = df['Fastest_Lap'].mean()
    df['Max_Velocity_avg'] = df['Max_Velocity'].mean()
    
    df['Race_pace_ratio'] = df['Race_pace_avg'] / df['Race_pace']
    df['Fastest_Lap_ratio'] = df['Fastest_Lap_avg'] / df['Fastest_Lap']
    df['Max_Velocity_ratio'] = df['Max_Velocity'] / df['Max_Velocity_avg']

    # Replace NaN ratios with the average of the available ratios for each driver
    df['Race_pace_ratio'] = df.apply(lambda row: row[['Race_pace_ratio', 'Fastest_Lap_ratio', 'Max_Velocity_ratio']].mean() if np.isnan(row['Race_pace_ratio']) else row['Race_pace_ratio'], axis=1)
    df['Fastest_Lap_ratio'] = df.apply(lambda row: row[['Race_pace_ratio', 'Fastest_Lap_ratio', 'Max_Velocity_ratio']].mean() if np.isnan(row['Fastest_Lap_ratio']) else row['Fastest_Lap_ratio'], axis=1)
    df['Max_Velocity_ratio'] = df.apply(lambda row: row[['Race_pace_ratio', 'Fastest_Lap_ratio', 'Max_Velocity_ratio']].mean() if np.isnan(row['Max_Velocity_ratio']) else row['Max_Velocity_ratio'], axis=1)

    # Remove unwanted columns
    df = df.drop(columns=['Race_pace_avg', 'Fastest_Lap_avg', 'Max_Velocity', 'Max_Velocity_avg', 'Race_pace', 'Fastest_Lap'])

    return df

if __name__ == '__main__':
    year = 2023
    all_data = pd.DataFrame()

    event_schedule = ff1.get_event_schedule(year)
    # rounds_to_process = event_schedule.head(12)  # Get the first 12 rounds

    # event_schedule = ff1.get_event_schedule(year)
    rounds_to_process = event_schedule.iloc[1:15]  # Get rounds from 12 to 20

    for grand_prix in rounds_to_process['EventName']:
        fp2_session = get_session_data(year, grand_prix, 'FP2')
        fp3_session = get_session_data(year, grand_prix, 'FP3')
        
        # Process only if both FP2 and FP3 are missing
        if not fp2_session and not fp3_session:
            race_data = collect_basic_data_for_season(year, grand_prix)
            if race_data is not None:
                all_data = pd.concat([all_data, race_data], ignore_index=True)


    # Optionally, save to CSV
    all_data.to_csv('data/f1_basic_data_2023_last_34_rounds.csv', index=False)

    print("Data has been loaded and processed.")

    # print("Dane zostały załadowane i zapisane do pliku f1_basic_data.csv.")