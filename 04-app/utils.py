from typing import Literal, List, Dict


import logging
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import uuid

logging.basicConfig(level=logging.INFO)


def query_anomalies(start_ts: str, end_ts: str) -> List[dict]:
    """Queries the database for anomalies within a specified time range.

    Args:
        start_ts (str): Start timestamp in 'YYYY-MM-DD HH:MM:SS' (24H - Military Time).
        end_ts (str): End timestamp in 'YYYY-MM-DD HH:MM:SS' (24H - Military Time).

        start_ts and end_ts must be between 2022-01-01 and 2022-06-02.

    Returns:
        List[dict]: A list of dictionaries, each representing an anomaly with keys:
            event_id (int): Unique identifier for the anomalous event.
            anomaly_start_ts (str): Start timestamp of the anomaly.
            anomaly_end_ts (str): End timestamp of the anomaly.
            event_duration_in_secs (int): Duration of the anomaly in seconds.
    """
    logging.info('Querying database for anomalies..')
    with sqlite3.connect(r'../02-data/data.db') as con:
        df = pd.read_sql(f"""
            select 
                event_id
                ,start_ts as anomaly_start_ts
                ,end_ts as anomaly_end_ts
                ,event_duration_in_secs
            from anomalies_consolidated
            where 
                event_duration_in_secs > 60*5
                and (
                    ('{start_ts}' <= start_ts and start_ts <= '{end_ts}')
                    or ('{start_ts}' <= end_ts and end_ts <= '{end_ts}')
                )
            order by anomaly_start_ts asc
        """, con=con)
    
    return df.to_dict(orient='records')


def query_analog_sensor_importances(start_ts: str, end_ts: str) -> Dict[str, int]:
    """The anomaly detection model works by analyzing sensor data from different sensors
    and checking if sensors are out of their expected range for >5 minutes. If >5 sensors (out of 8)
    are out of their expected range, then the timestamp is flagged as an anomaly.

    This function queries the database and returns the number of seconds each sensor was out of its 
    expected range within the specified time range. This serves as a proxy for sensor importance.

    Args:
        start_ts (str): Start timestamp in 'YYYY-MM-DD HH:MM:SS' (24H - Military Time).
        end_ts (str): End timestamp in 'YYYY-MM-DD HH:MM:SS' (24H - Military Time).

        start_ts and end_ts must be between 2022-01-01 and 2022-06-02.

    Returns:
        dict[str, int]: Dictionary with:
            key -> sensor_name
            value -> number of seconds the sensor was out of its expected range
    """
    logging.info('Querying sensor importances..')
    with sqlite3.connect(r'../02-data/data.db') as con:
        df = pd.read_sql(f"""
            select
                sum(tp2_pred) as tp2_pred 
                ,sum(tp3_pred) as tp3_pred
                ,sum(h1_pred) as h1_pred
                ,sum(dv_pressure_pred) as dv_pressure_pred
                ,sum(reservoirs_pred) as reservoirs_pred
                ,sum(oil_temperature_pred) as oil_temperature_pred
                ,sum(flowmeter_pred) as flowmeter_pred
                ,sum(motor_current_pred) as motor_current_pred
            from results_agg
            where '{start_ts}' <= ts and ts <= '{end_ts}'
        """, con=con)
    return df.to_dict(orient='records')[0]


def query_digital_sensor_activations(start_ts: str, end_ts: str) -> Dict[str, int]:
    """Queries the database and returns the number of seconds each digital sensor was activated.

    Digital sensors are binary (0 or 1). 
    1 indicates the sensor is activated, while 0 indicates it is not.

    Args:
        start_ts (str): Start timestamp in 'YYYY-MM-DD HH:MM:SS' (24H - Military Time).
        end_ts (str): End timestamp in 'YYYY-MM-DD HH:MM:SS' (24H - Military Time).

        start_ts and end_ts must be between 2022-01-01 and 2022-06-02.

    Returns:
        dict[str, int]: Dictionary with:
            key -> sensor_name
            value -> number of seconds the sensor was activated
    """
    logging.info('Querying digital sensor activations..')
    with sqlite3.connect(r'../02-data/data.db') as con:
        df = pd.read_sql(f"""
            select
                sum(comp) as comp 
                ,sum(dv_electric) as dv_electric
                ,sum(towers) as towers
                ,sum(mpg) as mpg
                ,sum(lps) as lps
                ,sum(pressure_switch) as pressure_switch
                ,sum(oil_level) as oil_level
                ,sum(caudal_impulses) as caudal_impulses
            from train_data
            where '{start_ts}' <= ts and ts <= '{end_ts}'
        """, con=con)
    return df.to_dict(orient='records')[0]


def update_analog_sensor_plot_for_user(
        sensor_name: Literal[
            'tp2',
            'tp3',
            'h1',
            'dv_pressure',
            'reservoirs',
            'oil_temperature',
            'flowmeter',
            'motor_current'
        ],
        start_ts: str,
        end_ts: str
    ) -> None:
    """Plots analog sensor data for the user.

    Args:
        sensor_name (str): Name of the sensor to plot. Must be an analog sensor, names provided in function signature.
        start_ts (str): Start timestamp in 'YYYY-MM-DD HH:MM:SS' (24H - Military Time).
        end_ts (str): End timestamp in 'YYYY-MM-DD HH:MM:SS' (24H - Military Time).

        start_ts and end_ts must be between 2022-01-01 and 2022-06-02.

    Returns: None. Displays a plot to the user.
    """
    logging.info('Plotting analog sensor data..')
    with sqlite3.connect(r'../02-data/data.db') as con:
        df = pd.read_sql(f"""
            select 
                t.ts
                ,t.{sensor_name}
                ,t.pred_filtered
                ,r.yhat_lower_with_buffer
                ,r.yhat_upper_with_buffer
            from train_data_processed as t
                left join results as r
                    on  r.ts = t.ts
                    and r.sensor = '{sensor_name}'
            where 
                '{start_ts}' <= t.ts and t.ts <= '{end_ts}'
        """, con=con, parse_dates=['ts'])

    # Plot
    ax = df.plot(
        x='ts', 
        y=[sensor_name, 'yhat_lower_with_buffer', 'yhat_upper_with_buffer'],
        figsize=(15, 5),
        color=['#003057', '#B3A369', '#B3A369'],
        legend=False
    );
    ax.set_title(f'Sensor Data: {sensor_name}');
    ax.set_xlabel(f'Time: {start_ts} to {end_ts}');

    # Modify Transparency
    for line, alpha in zip(ax.lines, [1., 0.3, 0.3]):
        line.set_alpha(alpha)

    # Hardcode Plot Limits -- to prevent fill_between from updating them
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax);
    ax.set_xlim(start_ts, end_ts);

    # Flag Anomalies
    ax.fill_between(
        df['ts'].to_list(),
        df[sensor_name].min() - 100,
        df[sensor_name].max() + 100,
        where=(df['pred_filtered'] == 1).to_list(),
        color='blue',
        alpha=0.4,
        label='Anomaly'
    );

    # Plot Legend
    ax.legend();
    handles, labels = ax.get_legend_handles_labels()
    handles = [h for h, l in zip(handles, labels) if l != 'yhat_lower_with_buffer']
    labels = ['Sensor Data', 'Anomaly Threshold', 'Anomaly']
    ax.legend(handles, labels, fancybox=True, shadow=True);

    # Save to file
    img_path = f'./imgs/{uuid.uuid4()}.png'
    fig = ax.get_figure()
    fig.savefig(img_path, dpi=300, bbox_inches='tight');
    plt.close(fig);

    return img_path


def update_digital_sensor_plot_for_user(
        sensor_name: Literal[
            'comp',
            'dv_electric',
            'towers',
            'mpg',
            'lps',
            'pressure_switch',
            'oil_level',
            'caudal_impulses'
        ],
        start_ts: str,
        end_ts: str
    ) -> None:
    """Plots analog sensor data for the user.

    Args:
        sensor_name (str): Name of the sensor to plot. Must be an analog sensor, names provided in function signature.
        start_ts (str): Start timestamp in 'YYYY-MM-DD HH:MM:SS' (24H - Military Time).
        end_ts (str): End timestamp in 'YYYY-MM-DD HH:MM:SS' (24H - Military Time).

        start_ts and end_ts must be between 2022-01-01 and 2022-03-01.

    Returns: None. Displays a plot to the user.
    """
    logging.info('Plotting digital sensor data..')
    with sqlite3.connect(r'../02-data/data.db') as con:
        df = pd.read_sql(f"""
            select 
                t.ts
                ,t.{sensor_name}
                ,t.pred_filtered
            from train_data_processed as t
            where 
                '{start_ts}' <= t.ts and t.ts <= '{end_ts}'
        """, con=con, parse_dates=['ts'])

    # Plot
    ax = df.plot(
        x='ts', 
        y=sensor_name,
        figsize=(15, 5),
        color=['#003057'],
        legend=False,
        alpha=1.,
        label='Sensor Data'
    );
    ax.set_title(f'Sensor Data: {sensor_name}');
    ax.set_xlabel(f'Time: {start_ts} to {end_ts}');

    # Hardcode Plot Limits -- to prevent fill_between from updating them
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax);
    ax.set_xlim(start_ts, end_ts);

    # Flag Anomalies
    ax.fill_between(
        df['ts'].to_list(),
        df[sensor_name].min() - 100,
        df[sensor_name].max() + 100,
        where=(df['pred_filtered'] == 1).to_list(),
        color='blue',
        alpha=0.4,
        label='Anomaly'
    );

    # Plot Legend
    ax.legend(loc='lower right', fancybox=True, shadow=True);
    plt.tight_layout()

    # Save to file
    img_path = f'./imgs/{uuid.uuid4()}.png'
    fig = ax.get_figure()
    fig.savefig(img_path, dpi=300, bbox_inches='tight');
    plt.close(fig);
    
    return img_path