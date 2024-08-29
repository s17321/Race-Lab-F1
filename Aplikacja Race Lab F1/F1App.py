import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import fastf1 as ff1
import fastf1.plotting
from PIL import Image
import io
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Funkcje do generowania stron
def home():
    st.title("🏎️🏎️🏎️🏎️🏎️🏎️🏎️🏎️🏎️🏎️🏎️ RaceLab F1 - Praca Inżynierska 🏁")
    logo = Image.open("logo_f1.png")
    st.image(logo, use_column_width=True)
    st.write("""
    ## Witaj w aplikacji RaceLab F1!
    RaceLab F1 został stworzony jako część mojej pracy inżynierskiej. Jej celem jest analiza danych Formuły 1 oraz predykcja wyników wyścigów przy użyciu zaawansowanych algorytmów.
    ### Funkcjonalności aplikacji:
    - **Predykcja wyników**: Czy kierowca ukończy w top3 lub top10, za pomocą:
      - Algorytmu Random Forests
      - Sieci neuronowych
    - **Analiza zmian pozycji**: Śledzenie zmian pozycji kierowców podczas wyścigu.
    - **Analiza czasów okrążeń**: Wizualizacja czasów okrążeń w formie scatterplotu.
    - **Porównanie tempa zespołów**: Porównanie prędkości zespołów na różnych etapach wyścigu.
    - **Strategia pit-stopów**: Analiza używanych opon i strategii pit-stopów.
    - **Wizualizacja prędkości**: Przedstawienie prędkości kierowców z interaktywną wizualizacją.
    Skorzystaj z menu nawigacyjnego po lewej stronie, aby przejść do interesującej Cię sekcji. Miłego korzystania!
    """)

def position_changes():
    st.title("Zmiany Pozycji Kierowców")
    st.write("Wybierz rok i Grand Prix, aby zobaczyć zmiany pozycji kierowców podczas wyścigu.")

    # Wybór roku i wyścigu
    year = st.selectbox("Wybierz rok", range(2020, 2025))
    grand_prix = st.selectbox("Wybierz Grand Prix", [
        "Bahrain", "Saudi Arabia", "Australia", "Azerbaijan", "Miami", 
        "Monaco", "Spain", "Canada", "Austria", "Great Britain", 
        "Hungary", "Belgium", "Netherlands", "Italy", "Singapore", 
        "Japan", "Qatar", "United States", "Mexico", "Brazil", "Las Vegas", "Abu Dhabi"
    ])

    if st.button("Generuj wykres"):
        with st.spinner('Pobieranie danych i generowanie wykresu...'):
            try:
                # Załaduj sesję
                session = fastf1.get_session(year, grand_prix, 'R')
                session.load(telemetry=False, weather=False)

                # Ustawienia wykresu
                fig, ax = plt.subplots(figsize=(10, 6))
                fastf1.plotting.setup_mpl(misc_mpl_mods=False)

                # Tworzenie wykresu
                for drv in session.drivers:
                    drv_laps = session.laps.pick_driver(drv)
                    abb = drv_laps['Driver'].iloc[0]
                    color = fastf1.plotting.driver_color(abb)
                    ax.plot(drv_laps['LapNumber'], drv_laps['Position'], label=abb, color=color)

                # Ustawienia osi
                ax.set_ylim([20.5, 0.5])
                ax.set_yticks([1, 5, 10, 15, 20])
                ax.set_xlabel('Lap')
                ax.set_ylabel('Position')
                ax.legend(bbox_to_anchor=(1.0, 1.02))
                plt.tight_layout()

                # Wyświetlenie wykresu w Streamlit
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                st.image(buf)
                plt.close(fig)
            except Exception as e:
                st.error(f'Wystąpił błąd: {e}')

def lap_times():
    st.title("Czasy Okrążeń Kierowców")
    st.write("Wybierz rok i Grand Prix, aby załadować dane.")

    year = st.selectbox("Wybierz rok", range(2020, 2025), key="year")
    grand_prix = st.selectbox("Wybierz Grand Prix", [
        "Bahrain", "Saudi Arabia", "Australia", "Azerbaijan", "Miami", 
        "Monaco", "Spain", "Canada", "Austria", "Great Britain", 
        "Hungary", "Belgium", "Netherlands", "Italy", "Singapore", 
        "Japan", "Qatar", "United States", "Mexico", "Brazil", "Las Vegas", "Abu Dhabi"
    ], key="grand_prix")

    if st.button("Załaduj dane"):
        with st.spinner('Ładowanie danych...'):
            try:
                session = fastf1.get_session(year, grand_prix, 'R')
                session.load(telemetry=False, weather=False)

                drivers = session.laps['Driver'].unique()
                st.session_state.drivers = drivers
                st.session_state.session = session

                driver = st.selectbox("Wybierz kierowcę", drivers, key="driver")

                if st.button("Generuj wykres"):
                    with st.spinner('Generowanie wykresu...'):
                        driver_laps = st.session_state.session.laps.pick_driver(driver).pick_quicklaps().reset_index()

                        # Konwersja 'LapTime' na sekundy
                        driver_laps['LapTime (s)'] = driver_laps['LapTime'].dt.total_seconds()

                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.scatterplot(data=driver_laps,
                                        x="LapNumber",
                                        y="LapTime (s)",
                                        ax=ax,
                                        hue="Compound",
                                        palette=fastf1.plotting.COMPOUND_COLORS,
                                        s=80,
                                        linewidth=0,
                                        legend='auto')

                        ax.set_xlabel("Lap Number")
                        ax.set_ylabel("Lap Time (s)")
                        ax.invert_yaxis()
                        plt.suptitle(f"{driver} Laptimes in the {year} {grand_prix} Grand Prix")
                        plt.grid(color='w', which='major', axis='both')
                        sns.despine(left=True, bottom=True)
                        plt.tight_layout()

                        buf = io.BytesIO()
                        plt.savefig(buf, format='png')
                        st.image(buf)
                        plt.close(fig)
            except Exception as e:
                st.error(f'Wystąpił błąd: {e}')
    elif "drivers" in st.session_state:
        driver = st.selectbox("Wybierz kierowcę", st.session_state.drivers, key="driver_existing")
        if st.button("Generuj wykres"):
            with st.spinner('Generowanie wykresu...'):
                driver_laps = st.session_state.session.laps.pick_driver(driver).pick_quicklaps().reset_index()

                # Konwersja 'LapTime' na sekundy
                driver_laps['LapTime (s)'] = driver_laps['LapTime'].dt.total_seconds()

                fig, ax = plt.subplots(figsize=(10, 6))
                sns.scatterplot(data=driver_laps,
                                x="LapNumber",
                                y="LapTime (s)",
                                ax=ax,
                                hue="Compound",
                                palette=fastf1.plotting.COMPOUND_COLORS,
                                s=80,
                                linewidth=0,
                                legend='auto')

                ax.set_xlabel("Lap Number")
                ax.set_ylabel("Lap Time (s)")
                ax.invert_yaxis()
                plt.suptitle(f"{driver} Laptimes in the {year} {grand_prix} Grand Prix")
                plt.grid(color='w', which='major', axis='both')
                sns.despine(left=True, bottom=True)
                plt.tight_layout()

                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                st.image(buf)
                plt.close(fig)

def team_pace():
    st.title("Porównanie Tempa Zespołów")
    st.write("Wybierz rok i Grand Prix, aby załadować dane.")

    year = st.selectbox("Wybierz rok", range(2020, 2025), key="year_team")
    grand_prix = st.selectbox("Wybierz Grand Prix", [
        "Bahrain", "Saudi Arabia", "Australia", "Azerbaijan", "Miami", 
        "Monaco", "Spain", "Canada", "Austria", "Great Britain", 
        "Hungary", "Belgium", "Netherlands", "Italy", "Singapore", 
        "Japan", "Qatar", "United States", "Mexico", "Brazil", "Las Vegas", "Abu Dhabi"
    ], key="grand_prix_team")

    if st.button("Załaduj dane"):
        with st.spinner('Ładowanie danych...'):
            try:
                session = fastf1.get_session(year, grand_prix, 'R')
                session.load(telemetry=False, weather=False)

                st.session_state.session_team = session

                if st.button("Generuj wykres"):
                    with st.spinner('Generowanie wykresu...'):
                        laps = st.session_state.session_team.laps.pick_quicklaps().reset_index()
                        transformed_laps = laps.copy()
                        transformed_laps['LapTime (s)'] = laps['LapTime'].dt.total_seconds()

                        team_order = (
                            transformed_laps[["Team", "LapTime (s)"]]
                            .groupby("Team")
                            .median()["LapTime (s)"]
                            .sort_values()
                            .index
                        )

                        team_palette = {team: fastf1.plotting.team_color(team) for team in team_order}

                        fig, ax = plt.subplots(figsize=(15, 10))
                        sns.boxplot(
                            data=transformed_laps,
                            x="Team",
                            y="LapTime (s)",
                            hue="Team",
                            order=team_order,
                            palette=team_palette,
                            whiskerprops=dict(color="white"),
                            boxprops=dict(edgecolor="white"),
                            medianprops=dict(color="grey"),
                            capprops=dict(color="white"),
                        )

                        plt.title(f"{year} {grand_prix} Grand Prix - Team Pace Comparison")
                        plt.grid(visible=False)
                        ax.set(xlabel=None)
                        plt.tight_layout()

                        buf = io.BytesIO()
                        plt.savefig(buf, format='png')
                        st.image(buf)
                        plt.close(fig)
            except Exception as e:
                st.error(f'Wystąpił błąd: {e}')
    elif "session_team" in st.session_state:
        if st.button("Generuj wykres"):
            with st.spinner('Generowanie wykresu...'):
                laps = st.session_state.session_team.laps.pick_quicklaps().reset_index()
                transformed_laps = laps.copy()
                transformed_laps['LapTime (s)'] = laps['LapTime'].dt.total_seconds()

                team_order = (
                    transformed_laps[["Team", "LapTime (s)"]]
                    .groupby("Team")
                    .median()["LapTime (s)"]
                    .sort_values()
                    .index
                )

                team_palette = {team: fastf1.plotting.team_color(team) for team in team_order}

                fig, ax = plt.subplots(figsize=(15, 10))
                sns.boxplot(
                    data=transformed_laps,
                    x="Team",
                    y="LapTime (s)",
                    hue="Team",
                    order=team_order,
                    palette=team_palette,
                    whiskerprops=dict(color="white"),
                    boxprops=dict(edgecolor="white"),
                    medianprops=dict(color="grey"),
                    capprops=dict(color="white"),
                )

                plt.title(f"{year} {grand_prix} Grand Prix - Team Pace Comparison")
                plt.grid(visible=False)
                ax.set(xlabel=None)
                plt.tight_layout()

                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                st.image(buf)
                plt.close(fig)

def pit_strategy():
    st.title("Strategia Pit-Stopów")
    st.write("Wybierz rok i Grand Prix, aby załadować dane.")

    # Wybór roku i Grand Prix
    year = st.selectbox("Wybierz rok", range(2020, 2025), key="year_pit")
    grand_prix = st.selectbox("Wybierz Grand Prix", [
        "Bahrain", "Saudi Arabia", "Australia", "Azerbaijan", "Miami", 
        "Monaco", "Spain", "Canada", "Austria", "Great Britain", 
        "Hungary", "Belgium", "Netherlands", "Italy", "Singapore", 
        "Japan", "Qatar", "United States", "Mexico", "Brazil", "Las Vegas", "Abu Dhabi"
    ], key="grand_prix_pit")

    if st.button("Załaduj dane"):
        with st.spinner('Ładowanie danych...'):
            try:
                # Załaduj sesję
                session = fastf1.get_session(year, grand_prix, 'R')
                session.load(telemetry=False, weather=False)

                # Przechowaj sesję w st.session_state
                st.session_state.session_pit = session

                # Wyświetl wiadomość
                st.success('Dane załadowane. Możesz teraz wygenerować wykres.')

            except Exception as e:
                st.error(f'Wystąpił błąd podczas ładowania danych: {e}')

    # Jeśli dane są załadowane
    if "session_pit" in st.session_state:
        if st.button("Generuj wykres"):
            with st.spinner('Generowanie wykresu...'):
                try:
                    # Pobierz dane z sesji
                    session = st.session_state.session_pit
                    laps = session.laps

                    # Oblicz stinty
                    stints = laps[["Driver", "Stint", "Compound", "LapNumber"]]
                    stints = stints.groupby(["Driver", "Stint", "Compound"]).count().reset_index()
                    stints = stints.rename(columns={"LapNumber": "StintLength"})

                    # Pobierz skróty nazwisk kierowców
                    drivers = [session.get_driver(driver)["Abbreviation"] for driver in session.drivers]

                    # Tworzenie wykresu
                    fig, ax = plt.subplots(figsize=(10, len(drivers) / 2))
                    for driver in drivers:
                        driver_stints = stints.loc[stints["Driver"] == driver]

                        previous_stint_end = 0
                        for _, row in driver_stints.iterrows():
                            plt.barh(
                                y=driver,
                                width=row["StintLength"],
                                left=previous_stint_end,
                                color=fastf1.plotting.COMPOUND_COLORS[row["Compound"]],
                                edgecolor="black",
                                fill=True
                            )

                            previous_stint_end += row["StintLength"]

                    plt.title(f"{year} {grand_prix} Grand Prix - Tyre Strategies")
                    plt.xlabel("Lap Number")
                    plt.grid(False)
                    ax.invert_yaxis()
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['left'].set_visible(False)

                    plt.tight_layout()

                    # Wyświetlenie wykresu w Streamlit
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png')
                    st.image(buf)
                    plt.close(fig)

                except Exception as e:
                    st.error(f'Wystąpił błąd podczas generowania wykresu: {e}')

def speed_visualization():
    st.title("Wizualizacja Prędkości Kierowców")
    st.write("Wybierz rok, Grand Prix oraz kierowcę, aby załadować dane i wygenerować wizualizację.")

    # Wybór roku, Grand Prix i kierowcy
    year = st.selectbox("Wybierz rok", range(2020, 2025), key="year_speed")
    grand_prix = st.selectbox("Wybierz Grand Prix", [
        "Bahrain", "Saudi Arabia", "Australia", "Azerbaijan", "Miami", 
        "Monaco", "Spain", "Canada", "Austria", "Great Britain", 
        "Hungary", "Belgium", "Netherlands", "Italy", "Singapore", 
        "Japan", "Qatar", "United States", "Mexico", "Brazil", "Las Vegas", "Abu Dhabi"
    ], key="grand_prix_speed")

    if st.button("Załaduj dane"):
        with st.spinner('Ładowanie danych...'):
            try:
                session = ff1.get_session(year, grand_prix, 'R')
                session.load()

                drivers = session.laps['Driver'].unique()
                st.session_state.drivers_speed = drivers
                st.session_state.session_speed = session

                driver = st.selectbox("Wybierz kierowcę", drivers, key="driver_speed")

                if st.button("Generuj wizualizację"):
                    with st.spinner('Generowanie wizualizacji...'):
                        lap = st.session_state.session_speed.laps.pick_driver(driver).pick_fastest()
                        
                        # Dane telemetryczne
                        x = lap.telemetry['X']
                        y = lap.telemetry['Y']
                        color = lap.telemetry['Speed']

                        # Tworzenie segmentów linii
                        points = np.array([x, y]).T.reshape(-1, 1, 2)
                        segments = np.concatenate([points[:-1], points[1:]], axis=1)

                        # Wykres
                        fig, ax = plt.subplots(sharex=True, sharey=True, figsize=(12, 6.75))
                        fig.suptitle(f'{session.event.name} {year} - {driver} - Speed', size=24, y=0.97)
                        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.12)
                        ax.axis('off')

                        # Ścieżka toru
                        ax.plot(lap.telemetry['X'], lap.telemetry['Y'],
                                color='black', linestyle='-', linewidth=16, zorder=0)

                        # Normalizacja i mapowanie kolorów
                        norm = plt.Normalize(color.min(), color.max())
                        lc = LineCollection(segments, cmap=mpl.cm.plasma, norm=norm,
                                            linestyle='-', linewidth=5)
                        lc.set_array(color)
                        ax.add_collection(lc)

                        # Pasek kolorów
                        cbaxes = fig.add_axes([0.25, 0.05, 0.5, 0.05])
                        normlegend = mpl.colors.Normalize(vmin=color.min(), vmax=color.max())
                        legend = mpl.colorbar.ColorbarBase(cbaxes, norm=normlegend, cmap=mpl.cm.plasma,
                                                           orientation="horizontal")

                        # Wyświetlenie wykresu w Streamlit
                        buf = io.BytesIO()
                        plt.savefig(buf, format='png')
                        st.image(buf)
                        plt.close(fig)
            except Exception as e:
                st.error(f'Wystąpił błąd: {e}')
    elif "drivers_speed" in st.session_state:
        driver = st.selectbox("Wybierz kierowcę", st.session_state.drivers_speed, key="driver_speed_existing")
        if st.button("Generuj wizualizację"):
            with st.spinner('Generowanie wizualizacji...'):
                try:
                    lap = st.session_state.session_speed.laps.pick_driver(driver).pick_fastest()
                    
                    # Dane telemetryczne
                    x = lap.telemetry['X']
                    y = lap.telemetry['Y']
                    color = lap.telemetry['Speed']

                    # Tworzenie segmentów linii
                    points = np.array([x, y]).T.reshape(-1, 1, 2)
                    segments = np.concatenate([points[:-1], points[1:]], axis=1)

                    # Wykres
                    fig, ax = plt.subplots(sharex=True, sharey=True, figsize=(12, 6.75))
                    fig.suptitle(f'{st.session_state.session_speed.event.name} {year} - {driver} - Speed', size=24, y=0.97)
                    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.12)
                    ax.axis('off')

                    # Ścieżka toru
                    ax.plot(lap.telemetry['X'], lap.telemetry['Y'],
                            color='black', linestyle='-', linewidth=16, zorder=0)

                    # Normalizacja i mapowanie kolorów
                    norm = plt.Normalize(color.min(), color.max())
                    lc = LineCollection(segments, cmap=mpl.cm.plasma, norm=norm,
                                        linestyle='-', linewidth=5)
                    lc.set_array(color)
                    ax.add_collection(lc)

                    # Pasek kolorów
                    cbaxes = fig.add_axes([0.25, 0.05, 0.5, 0.05])
                    normlegend = mpl.colors.Normalize(vmin=color.min(), vmax=color.max())
                    legend = mpl.colorbar.ColorbarBase(cbaxes, norm=normlegend, cmap=mpl.cm.plasma,
                                                       orientation="horizontal")

                    # Wyświetlenie wykresu w Streamlit
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png')
                    st.image(buf)
                    plt.close(fig)
                except Exception as e:
                    st.error(f'Wystąpił błąd: {e}')

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

def collect_data_for_prediction(year, grand_prix):
    fp2_session = get_session_data(year, grand_prix, 'FP2')
    fp3_session = get_session_data(year, grand_prix, 'FP3')
    quali_session = get_session_data(year, grand_prix, 'Q')
    sprint_session = get_session_data(year, grand_prix, 'S')
    race_session = get_session_data(year, grand_prix, 'R')

    if not race_session:
        print(f"Skipping {grand_prix} {year} due to missing race session data.")
        return None

    if fp2_session and fp3_session:
        # Przetwarzanie danych dla normalnego weekendu wyścigowego
        return process_regular_weekend(fp2_session, fp3_session, quali_session, race_session, year, grand_prix)
    else:
        # Przetwarzanie danych dla weekendu wyścigowego ze sprintem
        return process_sprint_weekend(sprint_session, quali_session, race_session, year, grand_prix)

def process_regular_weekend(fp2_session, fp3_session, quali_session, race_session, year, grand_prix):
    data = []
    processed_drivers = set()

    for session in [fp2_session, fp3_session, quali_session]:
        if session is None:
            continue
        for driver in session.drivers:
            if driver in processed_drivers:
                continue
            processed_drivers.add(driver)
            try:
                start_position = race_session.results.loc[race_session.results['DriverNumber'] == driver, 'GridPosition'].values[0]

                race_pace = extractRacePace(fp2_session, driver) if fp2_session else np.nan
                fastest_lap = extractFastestLap(fp3_session, driver) if fp3_session else np.nan
                max_velocity = extractMaxVelocity(quali_session, driver) if quali_session else np.nan

                data.append([year, grand_prix, driver, race_pace, fastest_lap, max_velocity, start_position])
            except Exception as e:
                print(f"Error processing data for driver {driver} in {grand_prix} {year}: {e}")

    df = pd.DataFrame(data, columns=['Year', 'GrandPrix', 'Driver', 'Race_pace', 'Fastest_Lap', 'Max_Velocity', 'StartPosition'])
    
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

def process_sprint_weekend(sprint_session, quali_session, race_session, year, grand_prix):
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

                race_pace = extractRacePace(sprint_session, driver) if sprint_session else np.nan
                fastest_lap = extractFastestLap(sprint_session, driver) if sprint_session else np.nan
                max_velocity = extractMaxVelocity(quali_session, driver) if quali_session else np.nan

                data.append([year, grand_prix, driver, race_pace, fastest_lap, max_velocity, start_position])
            except Exception as e:
                print(f"Error processing data for driver {driver} in {grand_prix} {year}: {e}")

    df = pd.DataFrame(data, columns=['Year', 'GrandPrix', 'Driver', 'Race_pace', 'Fastest_Lap', 'Max_Velocity', 'StartPosition'])
    
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

def prediction():
    st.title("Predykcja Wyników Wyścigu")
    st.write("Wybierz rok i Grand Prix, aby załadować dane, a następnie wybierz tryb przewidywania.")

    # Wybór roku i Grand Prix
    year = st.selectbox("Wybierz rok", range(2020, 2025), key="year_prediction")
    grand_prix = st.selectbox("Wybierz Grand Prix", [
        "Bahrain", "Saudi Arabia", "Australia", "Azerbaijan", "Miami", 
        "Monaco", "Spain", "Canada", "Austria", "Great Britain", 
        "Hungary", "Belgium", "Netherlands", "Italy", "Singapore", 
        "Japan", "Qatar", "United States", "Mexico", "Brazil", "Las Vegas", "Abu Dhabi"
    ], key="grand_prix_prediction")

    if st.button("Załaduj dane"):
        with st.spinner('Ładowanie danych...'):
            try:
                # Pobierz i przetwórz dane
                data = collect_data_for_prediction(year, grand_prix)
                if data is not None:
                    st.session_state.prediction_data = data
                    st.success("Dane załadowane pomyślnie.")
                else:
                    st.error("Brak danych dla wybranego wyścigu.")
            except Exception as e:
                st.error(f'Wystąpił błąd: {e}')

    # Jeśli dane zostały załadowane
    if "prediction_data" in st.session_state:
        mode = st.selectbox("Tryb przewidywania", ["Top 10", "Top 3"], key="mode_prediction")

        if st.button("Generuj przewidywania"):
            with st.spinner('Generowanie przewidywań...'):
                try:
                    data = st.session_state.prediction_data
                    features = ['Driver', 'StartPosition', 'Race_pace_ratio', 'Fastest_Lap_ratio', 'Max_Velocity_ratio']
                    X = data[features]

                    # Zakodowanie cechy 'Driver'
                    le = LabelEncoder()
                    X['Driver'] = le.fit_transform(X['Driver'])

                    # Wczytaj odpowiedni model
                    if mode == "Top 10":
                        with open('models/random_forest_classifier_points.pkl', 'rb') as file:
                            model = pickle.load(file)
                        top_n = 10
                    else:
                        with open('models/random_forest_classifier_podium.pkl', 'rb') as file:
                            model = pickle.load(file)
                        top_n = 3

                    # Przewidywania
                    probabilities = model.predict_proba(X)[:, 1]  # Prawdopodobieństwo klasy 1

                    # Wybranie top_n kierowców z najwyższymi prawdopodobieństwami
                    top_indices = np.argsort(probabilities)[-top_n:][::-1]
                    data['Prediction'] = 0
                    data.loc[top_indices, 'Prediction'] = 1

                    # Dodaj emoji do kolumny 'Prediction'
                    emoji_map = {0: "", 1: "🍾" if mode == "Top 3" else "⬆️"}
                    data['Prediction'] = data['Prediction'].map(emoji_map)

                    # Zamiana numerów kierowców na skróty
                    session = st.session_state.session_speed if 'session_speed' in st.session_state else get_session_data(year, grand_prix, 'R')
                    data['Driver'] = data['Driver'].apply(lambda x: session.get_driver(x)['Abbreviation'])

                    # Zresetowanie indeksów, aby zaczynały się od 1
                    data.index = data.index + 1

                    # Wyświetl wyniki
                    st.write("### Przewidywane wyniki:")
                    st.write(data[['Driver', 'StartPosition', 'Prediction']])

                    # Wyświetlenie efektu balonów
                    st.balloons()

                except Exception as e:
                    st.error(f'Wystąpił błąd: {e}')

class RaceNet(nn.Module):
    def __init__(self, input_size: int, classes: int):
        super(RaceNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, classes)
        self.logSoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        output = self.logSoftmax(x)
        return output

def load_neural_network_model(model_path, input_dim):
    model = RaceNet(input_dim, 2)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def prediction_neural_network():
    st.title("Predykcja Wyników Wyścigu - Sieci Neuronowe")
    st.write("Wybierz rok i Grand Prix, aby załadować dane, a następnie wybierz tryb przewidywania.")

    # Wybór roku i Grand Prix
    year = st.selectbox("Wybierz rok", range(2020, 2025), key="year_prediction_nn")
    grand_prix = st.selectbox("Wybierz Grand Prix", [
        "Bahrain", "Saudi Arabia", "Australia", "Azerbaijan", "Miami", 
        "Monaco", "Spain", "Canada", "Austria", "Great Britain", 
        "Hungary", "Belgium", "Netherlands", "Italy", "Singapore", 
        "Japan", "Qatar", "United States", "Mexico", "Brazil", "Las Vegas", "Abu Dhabi"
    ], key="grand_prix_prediction_nn")

    if st.button("Załaduj dane"):
        with st.spinner('Ładowanie danych...'):
            try:
                # Pobierz i przetwórz dane
                data = collect_data_for_prediction(year, grand_prix)
                if data is not None:
                    st.session_state.prediction_data_nn = data
                    st.success("Dane załadowane pomyślnie.")
                else:
                    st.error("Brak danych dla wybranego wyścigu.")
            except Exception as e:
                st.error(f'Wystąpił błąd: {e}')

    # Jeśli dane zostały załadowane
    if "prediction_data_nn" in st.session_state:
        mode = st.selectbox("Tryb przewidywania", ["Top 10", "Top 3"], key="mode_prediction_nn")

        if st.button("Generuj przewidywania"):
            with st.spinner('Generowanie przewidywań...'):
                try:
                    data = st.session_state.prediction_data_nn
                    features = ['Driver', 'StartPosition', 'Race_pace_ratio', 'Fastest_Lap_ratio', 'Max_Velocity_ratio']
                    X = data[features].copy()

                    # Zakodowanie cechy 'Driver'
                    le = LabelEncoder()
                    X['Driver'] = le.fit_transform(X['Driver'])

                    # Standardyzacja cech
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)

                    # Konwersja do tensora
                    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

                    # Wczytaj odpowiedni model sieci neuronowej
                    input_dim = X_tensor.shape[1]
                    if mode == "Top 10":
                        model_path = 'models/points_model.pth'
                        top_n = 10
                    else:
                        model_path = 'models/podium_model.pth'
                        top_n = 3

                    model = load_neural_network_model(model_path, input_dim)

                    # Przewidywania
                    with torch.no_grad():
                        probabilities = model(X_tensor)[:, 1].numpy()  # Prawdopodobieństwo klasy 1

                    # Wybranie top_n kierowców z najwyższymi prawdopodobieństwami
                    top_indices = np.argsort(probabilities)[-top_n:][::-1]
                    data['Prediction'] = 0
                    data.loc[top_indices, 'Prediction'] = 1

                    # Dodaj emoji do kolumny 'Prediction'
                    emoji_map = {0: "", 1: "🍾" if mode == "Top 3" else "⬆️"}
                    data['Prediction'] = data['Prediction'].map(emoji_map)

                    # Zamiana numerów kierowców na skróty
                    session = st.session_state.session_speed if 'session_speed' in st.session_state else get_session_data(year, grand_prix, 'R')
                    data['Driver'] = data['Driver'].apply(lambda x: session.get_driver(x)['Abbreviation'])

                    # Zresetowanie indeksów, aby zaczynały się od 1
                    data.index = data.index + 1

                    # Wyświetl wyniki
                    st.write("### Przewidywane wyniki:")
                    st.write(data[['Driver', 'StartPosition', 'Prediction']])

                    # Wyświetlenie efektu balonów
                    st.balloons()

                except Exception as e:
                    st.error(f'Wystąpił błąd: {e}')

# Menu nawigacyjne
st.sidebar.title("Nawigacja")
page = st.sidebar.radio("Przejdź do", [
    "Strona Główna", 
    "Predykcja - Random Forests", 
    "Predykcja - Sieci Neuronowe", 
    "Zmiany Pozycji", 
    "Czasy Okrążeń", 
    "Porównanie Tempa Zespołów", 
    "Strategia Pit-Stopów", 
    "Wizualizacja Prędkości"
])

# Wywołanie odpowiedniej funkcji w zależności od wybranej strony
if page == "Strona Główna":
    home()
elif page == "Predykcja - Random Forests":
    prediction()
elif page == "Predykcja - Sieci Neuronowe":
    prediction_neural_network()
elif page == "Zmiany Pozycji":
    position_changes()
elif page == "Czasy Okrążeń":
    lap_times()
elif page == "Porównanie Tempa Zespołów":
    team_pace()
elif page == "Strategia Pit-Stopów":
    pit_strategy()
elif page == "Wizualizacja Prędkości":
    speed_visualization()
