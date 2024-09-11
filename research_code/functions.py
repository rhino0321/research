import numpy as np
import pywt
def semi_soft_thresholding(temp_coeffs, ph_coeffs, ec_coeffs):
    for i in range(1, len(temp_coeffs)):
        temp_detail_coeff = temp_coeffs[i]
        ph_detail_coeff = ph_coeffs[i]
        ec_detail_coeff = ec_coeffs[i]
        temp_detail_len = len(temp_coeffs[i])
        ph_detail_len = len(ph_coeffs[i])
        ec_detail_len = len(ec_coeffs[i])
        temp_median_abs = np.median(np.abs(temp_coeffs[i]))
        ph_median_abs = np.median(np.abs(ph_coeffs[i]))
        ec_median_abs = np.median(np.abs(ec_coeffs[i]))
        #calculate lamda
        temp_lamda = (temp_median_abs / 0.6745) * (np.sqrt(2 * np.log(temp_detail_len)))
        ph_lamda = (ph_median_abs / 0.6745) * np.sqrt(2 * np.log(ph_detail_len))
        ec_lamda = (ec_median_abs / 0.6745) * np.sqrt(2 * np.log(ec_detail_len))
        #threshold
        temp_detail_coeff = np.where(np.abs(temp_detail_coeff) >= temp_lamda,np.sign(temp_detail_coeff) * (np.abs(temp_detail_coeff) - 0.1 * temp_lamda), 0)
        ph_detail_coeff = np.where(np.abs(ph_detail_coeff) >= ph_lamda,np.sign(ph_detail_coeff) * (np.abs(ph_detail_coeff) - 0.9 * ph_lamda), 0)
        ec_detail_coeff = np.where(np.abs(ec_detail_coeff) >= ec_lamda,np.sign(ec_detail_coeff) * (np.abs(ec_detail_coeff) - 0.9 * ec_lamda), 0)
        temp_coeffs[i] = temp_detail_coeff
        ph_coeffs[i] = ph_detail_coeff
        ec_coeffs[i] = ec_detail_coeff
    return temp_coeffs, ph_coeffs, ec_coeffs

def wavelet_decomposition(zscore_temp, zscore_ph, zscore_ec, decomposition_level):
    return pywt.wavedec(zscore_temp, wavelet = 'db1', level= decomposition_level, mode = 'periodic'), pywt.wavedec(zscore_ph, wavelet = 'db1', level= decomposition_level, mode = 'periodic'), pywt.wavedec(zscore_ec, wavelet = 'db1', level= decomposition_level, mode = 'periodic')

def wavelet_reconstruction(temp_coeffs, temp, ph_coeffs, ph, ec_coeffs, ec):
    temp_rec = pywt.waverec(temp_coeffs, wavelet = 'db1', mode = 'periodic')
    temp_rec = temp_rec * np.std(temp) + np.mean(temp)
    ph_rec = pywt.waverec(ph_coeffs, wavelet = 'db1', mode = 'periodic')
    ph_rec = ph_rec * np.std(ph) + np.mean(ph)
    ec_rec = pywt.waverec(ec_coeffs, wavelet = 'db1', mode = 'periodic')
    ec_rec = ec_rec * np.std(ec) + np.mean(ec)
    return temp_rec, ph_rec, ec_rec

def approx_entropy(coeffs):
    min_value = np.min(coeffs[0])
    value_counts = np.bincount((coeffs[0] - min_value).astype(int))
    total_coefficients = len(coeffs[0])
    probabilities = value_counts / total_coefficients
    probabilities = probabilities[probabilities > 0]
    entropies = -np.sum((probabilities**2) * np.log2(probabilities**2 + 1e-10))
    return entropies