import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import io
import streamlit as st
import concurrent.futures

# API
API_URL = "https://decision.cs.taltech.ee/electricity/api/"
DATA_URL = "https://decision.cs.taltech.ee/electricity/data/"

def fetch_metadata():
    response = requests.get(API_URL)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception("Failed to fetch metadata from API.")

def process_entry(entry, target_day):
    hash_ = entry['dataset']
    df = load_csv_from_hash(hash_)
    if df is not None:
        day_df = extract_target_day(df, target_day)
        if day_df is not None and not day_df.empty:
            return day_df[['Periood', 'consumption']]

def load_csv_from_hash(hash_):
    url = DATA_URL + hash_ + ".csv"
    response = requests.get(url)
    if response.status_code == 200:
        response.encoding = 'utf-8-sig'
        text = response.text

        try:
            df_try = pd.read_csv(io.StringIO(text), sep=';', skiprows=4, decimal=',', header=0)
            if 'Periood' not in df_try.columns:
                df_try = pd.read_csv(io.StringIO(text), sep=';', decimal=',', header=0)
            return df_try
        except Exception as e:
            try:
                df_try = pd.read_csv(io.StringIO(text), sep=None, engine='python', decimal=',')
                return df_try
            except Exception as e2:
                return None
    else:
        return None

def extract_target_day(df, target_day: str):
    if 'Periood' not in df.columns:
        return None

    if not pd.api.types.is_datetime64_any_dtype(df['Periood']):
        df['Periood'] = pd.to_datetime(df['Periood'], dayfirst=True, errors='coerce')

    consumption_col = None
    for col in df.columns:
        col_lower = col.lower().strip()
        if 'energia' in col_lower and 'kwh' in col_lower:
            consumption_col = col
            break

    if consumption_col is None:
        return None

    if df[consumption_col].dtype == object:
        df[consumption_col] = pd.to_numeric(df[consumption_col].str.replace(',', '.'), errors='coerce')
    else:
        df[consumption_col] = pd.to_numeric(df[consumption_col], errors='coerce')

    target_date = pd.to_datetime(target_day).date()
    day_df = df[df['Periood'].dt.date == target_date].copy()

    if day_df.empty:
        return None

    if day_df[consumption_col].isna().any():
        return None

    day_df = day_df.rename(columns={consumption_col: 'consumption'})
    return day_df[['Periood', 'consumption']]

def collect_all_one_day(target_day="2024-04-12", sample_size=25):
    metadata = fetch_metadata()

    random_entries = random.sample(metadata, min(sample_size, len(metadata)))

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(process_entry, entry, target_day) for entry in random_entries]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    all_rows = [r for r in results if r is not None]
    if all_rows:
        combined_df = pd.concat(all_rows, ignore_index=True)
        print(f"Combined dataframe shape: {combined_df.shape}")
        return combined_df
    else:
        print("No valid data found for the target day.")
        return pd.DataFrame(columns=["Periood", "consumption"])

# Visualize.
def create_hourly_average_circle(df: pd.DataFrame = None) -> plt.Figure:
    df['Periood'] = pd.to_datetime(df['Periood'], format='%d.%m.%Y %H:%M')
    df['Hour'] = df['Periood'].dt.hour
    hourly_avg = df.groupby('Hour')['consumption'].mean().reset_index()

    theta = 2 * np.pi * ((hourly_avg['Hour']) % 24) / 24
    radii = hourly_avg['consumption']

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, polar=True)

    ax.bar(
        theta,
        radii,
        width=0.26,
        bottom=0,
        color=plt.cm.viridis(radii / radii.max()),
        alpha=0.85
    )

    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_xticks(np.pi / 12 * np.arange(24))
    ax.set_xticklabels([f'{i}:00' for i in range(24)])
    ax.set_yticklabels([])  # Peidab ringi sees olevad y-teljed

    ax.set_title("Average Hourly Energy Usage", va='bottom')
    return fig


def plot_hourly_datapoints(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(16,6))

    ax.scatter(df['Periood'], df['consumption'], color='green', s=20, alpha=0.7, label='Hourly datapoints')

    ax.set_title("Hourly Consumption Data Points", fontsize=18)
    ax.set_xlabel("Timestamp", fontsize=14)
    ax.set_ylabel("Consumption", fontsize=14)

    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend()

    fig.autofmt_xdate()

    return fig


def plot_hourly_by_month(df: pd.DataFrame) -> plt.Figure:
    df_copy = df.copy()
    df_copy['Month'] = df_copy['Periood'].dt.to_period('M')
    df_copy['Hour'] = df_copy['Periood'].dt.hour

    hourly_monthly = df_copy.groupby(['Month', 'Hour'])['consumption'].mean().unstack()

    fig, ax = plt.subplots(figsize=(25, 11))
    hourly_monthly.T.plot(ax=ax)

    ax.set_title('Average Hourly Consumption per Month', fontsize=26)
    ax.set_xlabel('Hour of Day', fontsize=24)
    ax.set_ylabel('Average Consumption', fontsize=20)

    ax.tick_params(axis='both', which='major', labelsize=18)

    legend = ax.legend(title='Month', bbox_to_anchor=(1.01, 1), loc='upper left')
    legend.get_title().set_fontsize(20)
    for text in legend.get_texts():
        text.set_fontsize(20)

    plt.tight_layout()
    return fig


def design(circle_plt, calendar, hourly_by_month):
    st.title("Paljude andmestike üks päev")

    st.subheader("Keskmine igal tunnil")
    st.pyplot(circle_plt)
    st.markdown("_Keskmine tarbimine iga tunni jooksul päevas._")

    st.markdown("---")

    st.subheader("Iga andmepunkt mapitud graafikule")
    with st.container():
        st.pyplot(calendar)
        st.markdown("_Iga punkt tähistab ühe loetud andmestiku antud päeva antud tunni tarbimist._")

    st.markdown("---")

    st.subheader("Keskmine tarbimine igal tunnil")
    with st.container():
        st.pyplot(hourly_by_month)
        st.markdown("_Keskmine energiatarbimine tunniti._")

    st.markdown("<br>", unsafe_allow_html=True)

def main():
    st.title("Paljude andmestike üks päev")

    # Date picker for user to select the day
    selected_date = st.date_input(
        "Vali päev andmete kuvamiseks:",
        value=pd.to_datetime("2024-04-12"),
        min_value=pd.to_datetime("2020-01-01"),
        max_value=pd.to_datetime("2025-12-31"),
    )
    sample_size = st.slider(
        "Vali juhuslikult valitud andmestike arv (5–80):",
        min_value=5,
        max_value=80,
        value=10,
        step=1
    )

    # Convert to string in the format YYYY-MM-DD
    target_day = selected_date.strftime("%Y-%m-%d")

    df = collect_all_one_day(target_day=target_day, sample_size=sample_size)

    if df.empty:
        st.error(f"No valid data found for the target day: {target_day}")
        return

    # Create plots using fetched dataframe
    circle_fig = create_hourly_average_circle(df=df)
    datapoint_fig = plot_hourly_datapoints(df)
    hourly_by_month_fig = plot_hourly_by_month(df)

    # Show plots in Streamlit
    design(circle_fig, datapoint_fig, hourly_by_month_fig)

if __name__ == "__main__":
    main()
