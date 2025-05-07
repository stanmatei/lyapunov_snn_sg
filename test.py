import pickle
from plotting import plot_gradients_dht_dh0, plot_lyapunov_and_loss, plot_mambrane_voltages, plot_mean_firing_rates, plot_membrane_voltage_dist, plot_rasters, plot_rec_eigenvalues

with open("/Users/user/Desktop/lyapunov_snn_sg/results/2025-05-06/22:21:00.525131/results.pkl", 'rb') as f:
    results = pickle.load(f)

print(results["training_results"]["accuracies"])