import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import io
import streamlit as st
import calplot

def load_data_from_url(hash: str, base_url: str) -> pd.DataFrame:
    response = requests.get(base_url + hash)
    # Skip initial pointless rows.
    df_read_into_df = pd.read_csv(io.StringIO(response.text), sep=';', skiprows=4, header=0)
    return df_read_into_df

def load_data_from_file(filename) -> pd.DataFrame:
    df_read = pd.read_csv(filename)
    df_read.columns = ['Periood', 'kWh']
    return df_read

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = ['Periood', 'consumption']
    df = df[df['Periood'] != 'Periood']
    df['Periood'] = pd.to_datetime(df['Periood'], dayfirst=True)
    df['consumption'] = df['consumption'].astype(str).str.replace(',', '.')
    df['consumption'] = pd.to_numeric(df['consumption'], errors='coerce')
    df = df.dropna(subset=['consumption']).copy()
    return df

def create_hourly_average_circle(df: pd.DataFrame = None, filename=None) -> plt.Figure:
    if filename is not None:
        df = pd.read_csv(filename)
        df.columns = ['Periood', 'consumption']

    df['Periood'] = pd.to_datetime(df['Periood'], format='%d.%m.%Y %H:%M')
    df['Hour'] = df['Periood'].dt.hour
    hourly_avg = df.groupby('Hour')['consumption'].mean().reset_index()

    theta = 2 * np.pi * ((hourly_avg['Hour'] + 8) % 24) / 24
    radii = hourly_avg['consumption']

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, polar=True)

    radial_min = 0.25
    radial_max = 0.45

    ax.bar(theta, radii - radial_min, width=0.3, bottom=radial_min,
           color=plt.cm.viridis(radii / max(radii)), alpha=0.8)

    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_xticks(np.pi / 12 * np.arange(24))
    ax.set_xticklabels([f'{i}:00' for i in range(24)])
    ax.set_ylim(radial_min, radial_max)
    ax.set_title("Average Hourly Energy Usage", va='bottom')
    return fig


def plot_calendar_heatmap(df_copy: pd.DataFrame):
    df_copy = df_copy.copy()
    df_copy['Date'] = df_copy['Periood'].dt.date
    daily_sum = df_copy.groupby('Date')['consumption'].sum()

    daily_sum.index = pd.to_datetime(daily_sum.index)

    calplot.calplot(daily_sum, cmap='YlOrRd', figsize=(25, 11))
    return plt.gcf()


def plot_hourly_by_month(df: pd.DataFrame):
    df = df.copy()
    df['Month'] = df['Periood'].dt.to_period('M')
    df['Hour'] = df['Periood'].dt.hour

    hourly_monthly = df.groupby(['Month', 'Hour'])['consumption'].mean().unstack()

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
    st.title("Interesting Visualizations of Energy Usage")

    # Section 1: Circle plot
    st.subheader("Keskmine iga tunni jooksul")
    st.pyplot(circle_plt)
    st.markdown("_Keskmine tarbimine iga tunni jooksul päevas._")

    st.markdown("---")

    st.subheader("Päeva kogutarbimise heatmap")
    with st.container():
        st.pyplot(calendar)
        st.markdown("_Soojuskaart, mis näitab päevaseid tarbimisharjumusi kuude lõikes._")

    st.markdown("---")

    st.subheader("Kuu keskmine päevane tarbimine igal tunnil")
    with st.container():
        st.pyplot(hourly_by_month)
        st.markdown("_Keskmine energiatarbimine tunniti kuu lõikes._")

    st.markdown("<br>", unsafe_allow_html=True)

def main():
    filename = "raw-loaded-data.csv"
    contents = load_data_from_file(filename)
    df = clean_data(contents)

    circle = create_hourly_average_circle(df)
    fig3 = plot_calendar_heatmap(df)
    fig4 = plot_hourly_by_month(df)

    design(circle, fig3, fig4)

if __name__ == "__main__":
    main()

