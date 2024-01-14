import numpy as np

def bin_spectra_with_id(data, bin_width):
    """
    Bin spectra by averaging over bin_width number of points.
    
    Parameters:
        data (ndarray): 2D array where the first column is the sample ID and subsequent columns are spectral features.
        bin_width (int): Width of the bin over which to average.
        
    Returns:
        binned_data (ndarray): Binned spectra with sample IDs.
    """
    
    # Separate sample IDs and spectral data
    sample_ids = data[:, 0]
    spectra = data[:, 1:]
    
    n_samples, n_features = spectra.shape
    n_bins = n_features // bin_width
    binned_spectra = np.zeros((n_samples, n_bins), dtype=np.float32)
    
    for i in range(n_bins):
        start = i * bin_width
        end = start + bin_width
        binned_spectra[:, i] = np.mean(spectra[:, start:end], axis=1)
    
    # Reattach sample IDs
    binned_data = np.hstack((sample_ids[:, np.newaxis], binned_spectra))
    
    return binned_data
