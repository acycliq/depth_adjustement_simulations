import numpy as np
import pandas as pd
import plotly.express as px


def plot_confusion_matrix(cm, keys, opts_3D):
    # Convert confusion matrix to DataFrame for Plotly
    cm_df = pd.DataFrame(cm, index=sorted(keys), columns=sorted(keys))

    # Reset index for long-form DataFrame
    cm_long = cm_df.reset_index().melt(id_vars='index')
    cm_long.columns = ['Actual', 'Predicted', 'Probability']

    # Format the settings dictionary for display as subtitle
    settings_text = "<br>".join([f"{key}: {value}" for key, value in opts_3D.items()])
    print(f"Settings text length: {len(settings_text)}")
    print("Full settings text:")
    print(settings_text.replace('<br>', '\n'))

    # Create interactive heatmap with tooltip
    fig = px.imshow(
        cm,
        x=sorted(keys),
        y=sorted(keys),
        labels=dict(x="Predicted Class", y="Actual Class", color="Probability"),
        color_continuous_scale="Viridis",
    )
    fig.update_traces(
        xgap=0.25,  # Gap between cells along the x-axis for grid lines
        ygap=0.25,  # Gap between cells along the y-axis for grid lines
        hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Prob: %{z:.3f}<extra></extra>",
        colorbar=dict(len=0.5, y=0.5)  # Adjust the color bar height and position
    )
    # Calculate the sum of the diagonal (correct classifications)
    diagonal_sum = np.trace(cm)
    diagonal_avg = diagonal_sum / cm.shape[0]

    fig.update_layout(
        title=f'Confusion Matrix (Diagonal Sum: {diagonal_sum:.3f}, Avg: {diagonal_avg:.3f})',
        xaxis_tickangle=-90,
        xaxis=dict(showgrid=True, gridwidth=5, gridcolor="lightgray", dtick=1, tickfont=dict(size=11)),
        yaxis=dict(showgrid=True, gridwidth=5, gridcolor="lightgray", dtick=1, tickfont=dict(size=11)),
        autosize=False,
        width=1000,
        height=1400,
        margin=dict(t=80, b=600, l=100, r=50),
        annotations=[
            dict(
                text=f"<b>Settings:</b><br>{settings_text}",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.5, y=-0.15,
                xanchor='center', yanchor='top',
                font=dict(size=11)
            )
        ]
    )
    return fig


def confusion_matrix(keys, out):
    class_to_idx = {cls: i for i, cls in enumerate(sorted(keys))}
    confusion_prob_matrix = np.zeros((len(keys), len(keys)))

    # Parse ClassName and Prob columns from string to list
    for _, row in out.iterrows():
        actual_class = row['Actual_class']
        actual_idx = class_to_idx.get(actual_class, -1)

        if actual_idx == -1:
            continue

        try:
            predicted_classes = row['Estimated_class']
            predicted_probs = row['Prob']
        except:
            continue

        for pred_class, prob in zip(predicted_classes, predicted_probs):
            pred_idx = class_to_idx.get(pred_class, -1)
            if pred_idx != -1:
                confusion_prob_matrix[actual_idx, pred_idx] += float(prob)

    return confusion_prob_matrix