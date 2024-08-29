import fastf1 as ff1
import pandas as pd


def get_session_data(year, grand_prix, session_name):
    session = ff1.get_session(year, grand_prix, session_name)
    try:
        session.load()
        if not hasattr(session, 'laps') or session.laps.empty:
            print(f"No lap data available for {session_name} session.")
            return None
    except Exception as e:
        print(f"Error loading session {session_name} for {grand_prix} {year}: {e}")
        return None
    return session

def collect_basic_data_for_season(year, grand_prix):
    quali_session = get_session_data(year, grand_prix, 'Q')
    race_session = get_session_data(year, grand_prix, 'R')
    fp2_session = get_session_data(year, grand_prix, 'FP2')
    fp3_session = get_session_data(year, grand_prix, 'FP3')

    if not quali_session or not race_session:
        print(f"Skipping {grand_prix} {year} due to missing data.")
        return None

    data = []
    for driver in quali_session.drivers:
        try:
            start_position = quali_session.results.loc[quali_session.results['DriverNumber'] == driver, 'Position'].values[0]
            finish_position = race_session.results.loc[race_session.results['DriverNumber'] == driver, 'Position'].values[0]

            # Na razie zapisujemy tylko podstawowe dane
            data.append([year, grand_prix, driver, start_position, finish_position])
        except Exception as e:
            print(f"Error processing data for driver {driver} in {grand_prix} {year}: {e}")

    return pd.DataFrame(data, columns=['Year', 'GrandPrix', 'Driver', 'StartPosition', 'FinishPosition'])

if __name__ == '__main__':
    years = [2023]
    all_data = pd.DataFrame()

    for year in years:
        event_names = ff1.get_event_schedule(year).EventName.unique()
        race_data = collect_basic_data_for_season(year, event_names[1])
        print(race_data)

# Zapisanie danych do pliku CSV
all_data.to_csv('data/f1_basic_data.csv', index=False)

print("Dane zostały załadowane i zapisane do pliku f1_basic_data.csv.")
