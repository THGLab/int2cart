import matplotlib.pyplot as plt

def plot_scatter(x_values, y_values, x_label, y_label, title):
    non_zero_filter = y_values != 0
    fig = plt.figure(figsize=(12, 6))
    plt.scatter(x_values[non_zero_filter], y_values[non_zero_filter], s=1)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    return fig