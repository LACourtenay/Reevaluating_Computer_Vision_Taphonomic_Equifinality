import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import os
import scipy
from scipy.stats import shapiro, median_abs_deviation
import skimage as ski
from skimage.exposure import is_low_contrast
import csv

print("\nLibraries\n-----------------------------")
print(f"OpenCV: {cv2.__version__}")
print(f"Numpy: {np.__version__}")
print(f"Matplotlib: {matplotlib.__version__}")
print(f"SciPy: {scipy.__version__}")
print(f"SciKit Image: {ski.__version__}")

def true_value_exists(matrix):
    for row in matrix:
        if True in row:
            return True
    return False

def quantile_CI(x, q=[0.05, 0.95]):

    if not isinstance(x, (list, np.ndarray)) or len(x) == 0:

        raise ValueError("Quantile values can only be calculated on non-empty lists or arrays")

    if not isinstance(q, (list, np.ndarray)):

        raise ValueError("Quantiles must be provided as a list or array")

    if len(q) == 2:

        if not all(isinstance(quant, (int, float)) for quant in q):

            raise ValueError("Quantiles must be numeric")

        if any(quant < 0 or quant > 1 for quant in q):

            raise ValueError("Quantile values must be between 0 and 1")

        x_sorted = np.sort(x)
        lower_CI = x_sorted[int(q[0] * len(x_sorted))]
        upper_CI = x_sorted[int(q[1] * len(x_sorted))]

        return [lower_CI, upper_CI]

    elif len(q) == 1:

        if not isinstance(q[0], (int, float)):

            raise ValueError("Quantile must be numeric")

        x_sorted = np.sort(x)
        x_prima = x_sorted[int(q[0]) * len(x_sorted)]

        return x_prima

    else:

        raise ValueError("Invalid number of quantiles provided")


def descriptive_statistics(data, percentage = False):

    print(f"Shapiro results w = {shapiro(data)[0]:.2f}, p = {shapiro(data)[1]}")
    
    quantile_ranges = quantile_CI(data, [0.025, 0.975])

    if percentage:
        
        print(f"\nMin: {np.min(data):.2f} %")
        
        if shapiro(data)[1] < 0.003:
            
            print(f"Median: {np.median(data):.3f} %")
            print(f"NMAD: {(median_abs_deviation(data) * 1.4826):.3f}")
            
        else:
            
            print(f"Mean: {np.mean(data):.3f}%")
            print(f"Standard Deviation: {np.std(data):.3f}")

        print(
            f"95% Quantile Intevals: [{quantile_ranges[0]:.2f},",
            f"{quantile_ranges[1]:.2f}] %"
        )
        
        print(f"Max: {np.max(data):.2f} %")

    else:

        print(f"\nMin: {np.min(data):.2f}")

        if shapiro(data)[1] < 0.003:
            
            print(f"Median: {np.median(data):.3f}")
            print(f"NMAD: {(median_abs_deviation(data) * 1.4826):.3f}")
            
        else:
            
            print(f"Mean: {np.mean(data):.3f}")
            print(f"Standard Deviation: {np.std(data):.3f}")

        print(
            f"95% Quantile Intevals: [{quantile_ranges[0]:.2f},",
            f"{quantile_ranges[1]:.2f}] %"
        )
        print(f"Max: {np.max(data):.2f}")

class loadImage():

    def __init__(self, image):

        self.image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        low_contrast_bool = is_low_contrast(self.image, fraction_threshold = 0.35)
        if low_contrast_bool:
            self.adequate_contrast = False
        else:
            self.adequate_contrast = True

class LaplacianAnalysis(loadImage):

    def __init__(self, image):

        super().__init__(image)

        gaussian_blur = cv2.GaussianBlur(self.image, (5, 5), 0)
        self.laplacian_map = cv2.Laplacian(gaussian_blur, cv2.CV_64F, ksize = 3)
        self.laplacian_vector = self.laplacian_map.ravel()

        self.threshold = quantile_CI(np.abs(self.laplacian_vector), [0, 0.66]) [1]
    
    def calculate_variance(self):

        return(self.laplacian_vector.var())
    
    def calculate_quantile_range(self):

        interquantile_range = quantile_CI(
            self.laplacian_vector, [0.025, 0.975]
        )[1] - quantile_CI(
            self.laplacian_vector, [0.025, 0.975]
        )[0]

        return(interquantile_range)
    
    def binary_map(self):
        
        binary_matrix = (np.abs(self.laplacian_map) > self.threshold).astype(int)
        
        return(binary_matrix)
 

class CannyEdgeAnalysis(loadImage):

    def __init__(self, image):

        super().__init__(image)

        gaussian_blur = cv2.GaussianBlur(self.image, (5, 5), 0)
        self.edges = cv2.Canny(gaussian_blur, 10, 20)
    
    def calculate_features(self):

        dilate_kernel = np.ones((3, 3), np.uint8)
        dilated_edges = cv2.dilate(self.edges, dilate_kernel, iterations = 1)
        perc_detectable_features = (
            np.sum(dilated_edges.ravel() > 1) / dilated_edges.ravel().shape[0]
        ) * 100

        return(perc_detectable_features)

class FastFourierTransform(loadImage):

    def calculate_FFT(self, size = 60, robust = False):
        
        h, w = self.image.shape
        cX, cY = (int(w / 2), int(h / 2))
        
        fft = np.fft.fft2(self.image)
        fftShift = np.fft.fftshift(fft)
        
        fftShift[cY - size:cY + size, cX - size:cX + size] = 0
        fftShift = np.fft.ifftshift(fftShift)
        
        recon = np.fft.ifft2(fftShift)
        
        magnitude = 20 * np.log(np.abs(recon))
        magnitude = magnitude.ravel()

        if robust:
            
            shapiro_p_mag = shapiro(magnitude)[1]

            if shapiro_p_mag < 0.003:

                return(np.median(magnitude))
            
            else:

                return(np.mean(magnitude))

        else:
            
            return(np.mean(magnitude))

class SobelGradientMaps(loadImage):

    def __init__(self, image):

        super().__init__(image)

        self.sobel_x = cv2.Sobel(self.image, cv2.CV_64F, 1, 0)
        self.sobel_y = cv2.Sobel(self.image, cv2.CV_64F, 0, 1)

    def gradient_map(self):

        return(cv2.addWeighted(self.sobel_x, 0.5, self.sobel_y, 0.5, 0))
    
    def scaled_gradient_map(self):

        gX = cv2.convertScaleAbs(self.sobel_x)
        gY = cv2.convertScaleAbs(self.sobel_y)

        return(cv2.addWeighted(gX, 0.5, gY, 0.5, 0))
    
class SpectralReflection():

    def __init__(self, image, erode_iterations = 2):

        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        B = np.mean(image[:,:,0].ravel())
        self.bin_image = hsv_image[:,:,2] > (B * 2)
        self.bin_image = self.bin_image.astype(np.uint8)
        
        (h, w) = image.shape[0], image.shape[1]
        
        kernel = np.ones((3, 3), np.uint8)
        self.clean_bin_image = cv2.erode(
            self.bin_image, kernel, iterations = erode_iterations
        )
        
        self.num_pixels = sum(self.bin_image.ravel())
        self.num_pixels_clean = sum(self.clean_bin_image.ravel())
        self.percentage = (self.num_pixels / (h * w)) * 100
        self.percentage_clean = (self.num_pixels_clean / (h * w)) * 100
        
        if self.num_pixels > 0:
            self.detected = True
        else:
            self.detected = False
            
        if self.num_pixels_clean > 0:
            self.detected_clean = True
        else:
            self.detected_clean = False

results_folder = "./CV Results"
results_ds1_subfolder = "./CV Results/DS1"
os.makedirs(results_folder, exist_ok = True)
os.makedirs(results_ds1_subfolder, exist_ok = True)
canny_folder = "./CV Results/DS1/Canny Edge Detection"
gradient_maps_folder = "./CV Results/DS1/Sobel Gradient Maps"
specular_reflection_folder = "./CV Results/DS1/Specularities"
os.makedirs(canny_folder, exist_ok = True)
os.makedirs(gradient_maps_folder, exist_ok = True)
os.makedirs(specular_reflection_folder, exist_ok = True)
output_csv_path = "./CV Results/results_DS1.csv"

main_directory = "./DS1"

number_of_images_good_quality_images = 0
number_of_images = 0
detected_spectral_reflection = 0
clean_detected_spectral_reflection = 0
LoG_Variance = []
LoG_Range = []
percentages = []
fft_magnitudes = []
percentage_pixels = []
percentage_pixels_clean = []

with open(output_csv_path, mode = "w", newline = "") as csv_file:

    csv_writer = csv.writer(csv_file, delimiter = ",", quotechar = '"', quoting=csv.QUOTE_MINIMAL)

    csv_writer.writerow([
        'Image Name', 'LoG Variance', 'LoG Range', 'Canny Percentage Features', 'FFT Magnitude',
        'Specularities', '%', 'Clean Specularities', '%', 'Adequate Contrast'
    ])

    for subfolder in os.listdir(main_directory):
        
        subfolder_path = os.path.join(main_directory, subfolder)
        
        if os.path.isdir(subfolder_path):
            
            for file_name in os.listdir(subfolder_path):
                
                if file_name.lower().endswith((".bmp")):
                    
                    image_path = os.path.join(subfolder_path, file_name)
                    image = cv2.imread(image_path)

                    if image is not None:

                        number_of_images += 1
                        
                        laplace = LaplacianAnalysis(image)
                        
                        LoGVar = laplace.calculate_variance()
                        LoGQR = laplace.calculate_quantile_range()
                        
                        LoG_Variance.append(LoGVar)
                        LoG_Range.append(LoGQR)

                        canny = CannyEdgeAnalysis(image)
                        percentage_features = canny.calculate_features()
                        
                        percentages.append(percentage_features)

                        canny_edge_path = os.path.join(canny_folder, f"{file_name}.png")
                        cv2.imwrite(canny_edge_path, canny.edges)

                        sobel = SobelGradientMaps(image)

                        sobel_map_path = os.path.join(gradient_maps_folder, f"{file_name}.png")

                        plt.imshow(sobel.gradient_map(), cmap = "gray")
                        plt.axis("off")
                        plt.savefig(sobel_map_path, bbox_inches = "tight", pad_inches = 0.1)
                        plt.close()

                        FFT = FastFourierTransform(image)
                        magnitude = FFT.calculate_FFT()
                        
                        fft_magnitudes.append(magnitude)

                        image_contrast = loadImage(image).adequate_contrast

                        image_spec = SpectralReflection(image)

                        if (image_spec.detected):
                            detected_spectral_reflection += 1
                            percentage_pixels.append(image_spec.percentage)
                            specularities = "True"
                        else:
                            specularities = "False"
                        
                        if (image_spec.detected_clean):
                            clean_detected_spectral_reflection += 1
                            percentage_pixels_clean.append(image_spec.percentage_clean)
                            clean_specularities = "True"
                        else:
                            clean_specularities = "False"
                        
                        specularity_path = os.path.join(specular_reflection_folder, f"{file_name}.png")
                        plt.imshow(image_spec.bin_image, cmap = "gray")
                        plt.axis("off")
                        plt.savefig(specularity_path, bbox_inches = "tight", pad_inches = 0.1)
                        plt.close()
                    
                        if image_contrast:
                            number_of_images_good_quality_images +=1
                            contrast_quality = "True"
                        else:
                            contrast_quality = "False"
                        
                        csv_writer.writerow([
                            file_name, LoGVar, LoGQR, percentage_features, magnitude,
                            specularities, image_spec.percentage,
                            clean_specularities, image_spec.percentage_clean,
                            contrast_quality
                        ])

                        
print("\nDS1")

print("\nStatistics on LoG Variance ----------------------------------------------------\n")
descriptive_statistics(LoG_Variance)

print("\nStatistics on LoG Range  ----------------------------------------------------\n")
descriptive_statistics(LoG_Range)

print("\nStatistics on Percentage of Detectable Features  ----------------------------------------------------\n")
descriptive_statistics(percentages)

print("\nStatistics on Mean Magnitude of Representations by FFT  ----------------------------------------------------\n")
descriptive_statistics(fft_magnitudes)

print("\nStatistics on the Number of Pixels presenting Specular Reflections  ----------------------------------------------------\n")
descriptive_statistics(percentage_pixels, percentage = True)

print("\nStatistics on the Number of Pixels presenting Specular Reflections after erosion  ----------------------------------------------------\n")
descriptive_statistics(percentage_pixels_clean, percentage = True)

specular_reflection_detection = detected_spectral_reflection / number_of_images
clean_specular_reflection_detection = clean_detected_spectral_reflection / number_of_images
print(
    f"\n\n{specular_reflection_detection * 100:.2f} %",
    "of images present piques in specular properties"
)
print(
    f"{clean_specular_reflection_detection * 100:.2f} % of images present",
    "piques in specular properties even after 2 erosion operations"
)

perc_good_images = number_of_images_good_quality_images / number_of_images
print(f"\n\n{perc_good_images * 100:.2f} % of images have sufficient contrast")


output_csv_path = "./CV Results/results_DS2.csv"

main_directory = "./DS2"

number_of_images_good_quality_images = 0
number_of_images = 0
detected_spectral_reflection = 0
clean_detected_spectral_reflection = 0
LoG_Variance = []
LoG_Range = []
percentages = []
fft_magnitudes = []
percentage_pixels = []
percentage_pixels_clean = []

with open(output_csv_path, mode = "w", newline = "") as csv_file:

    csv_writer = csv.writer(csv_file, delimiter = ",", quotechar = '"', quoting=csv.QUOTE_MINIMAL)

    csv_writer.writerow([
        'Image Name', 'LoG Variance', 'LoG Range', 'Canny Percentage Features', 'FFT Magnitude',
        'Specularities', '%', 'Clean Specularities', '%', 'Adequate Contrast'
    ])

    for subfolder in os.listdir(main_directory):
        
        subfolder_path = os.path.join(main_directory, subfolder)
        
        if os.path.isdir(subfolder_path):
            
            for file_name in os.listdir(subfolder_path):
                
                if file_name.lower().endswith((".bmp")):
                    
                    image_path = os.path.join(subfolder_path, file_name)
                    image = cv2.imread(image_path)

                    if image is not None:

                        number_of_images += 1
                        
                        laplace = LaplacianAnalysis(image)
                        
                        LoGVar = laplace.calculate_variance()
                        LoGQR = laplace.calculate_quantile_range()
                        
                        LoG_Variance.append(LoGVar)
                        LoG_Range.append(LoGQR)

                        canny = CannyEdgeAnalysis(image)
                        percentage_features = canny.calculate_features()
                        
                        percentages.append(percentage_features)

                        sobel = SobelGradientMaps(image)

                        FFT = FastFourierTransform(image)
                        magnitude = FFT.calculate_FFT()
                        
                        fft_magnitudes.append(magnitude)

                        image_contrast = loadImage(image).adequate_contrast

                        image_spec = SpectralReflection(image)

                        if (image_spec.detected):
                            detected_spectral_reflection += 1
                            percentage_pixels.append(image_spec.percentage)
                            specularities = "True"
                        else:
                            specularities = "False"
                        
                        if (image_spec.detected_clean):
                            clean_detected_spectral_reflection += 1
                            percentage_pixels_clean.append(image_spec.percentage_clean)
                            clean_specularities = "True"
                        else:
                            clean_specularities = "False"
                    
                        if image_contrast:
                            number_of_images_good_quality_images +=1
                            contrast_quality = "True"
                        else:
                            contrast_quality = "False"
                        
                        csv_writer.writerow([
                            file_name, LoGVar, LoGQR, percentage_features, magnitude,
                            specularities, image_spec.percentage,
                            clean_specularities, image_spec.percentage_clean,
                            contrast_quality
                        ])

                        
print("\nDS2")

print("\nStatistics on LoG Variance ----------------------------------------------------\n")
descriptive_statistics(LoG_Variance)

print("\nStatistics on LoG Range  ----------------------------------------------------\n")
descriptive_statistics(LoG_Range)

print("\nStatistics on Percentage of Detectable Features  ----------------------------------------------------\n")
descriptive_statistics(percentages)

print("\nStatistics on Mean Magnitude of Representations by FFT  ----------------------------------------------------\n")
descriptive_statistics(fft_magnitudes)

print("\nStatistics on the Number of Pixels presenting Specular Reflections  ----------------------------------------------------\n")
descriptive_statistics(percentage_pixels, percentage = True)

print("\nStatistics on the Number of Pixels presenting Specular Reflections after erosion  ----------------------------------------------------\n")
descriptive_statistics(percentage_pixels_clean, percentage = True)

specular_reflection_detection = detected_spectral_reflection / number_of_images
clean_specular_reflection_detection = clean_detected_spectral_reflection / number_of_images
print(
    f"\n\n{specular_reflection_detection * 100:.2f} %",
    "of images present piques in specular properties"
)
print(
    f"{clean_specular_reflection_detection * 100:.2f} % of images present",
    "piques in specular properties even after 2 erosion operations"
)

perc_good_images = number_of_images_good_quality_images / number_of_images
print(f"\n\n{perc_good_images * 100:.2f} % of images have sufficient contrast")

results_ds3_subfolder = "./CV Results/DS3"
os.makedirs(results_ds3_subfolder, exist_ok = True)
canny_folder = "./CV Results/DS3/Canny Edge Detection"
gradient_maps_folder = "./CV Results/DS3/Sobel Gradient Maps"
specular_reflection_folder = "./CV Results/DS3/Specularities"
os.makedirs(canny_folder, exist_ok = True)
os.makedirs(gradient_maps_folder, exist_ok = True)
os.makedirs(specular_reflection_folder, exist_ok = True)
output_csv_path = "./CV Results/results_DS3.csv"

main_directory = "./DS3"

number_of_images_good_quality_images = 0
number_of_images = 0
detected_spectral_reflection = 0
clean_detected_spectral_reflection = 0
LoG_Variance = []
LoG_Range = []
percentages = []
fft_magnitudes = []
percentage_pixels = []
percentage_pixels_clean = []

with open(output_csv_path, mode = "w", newline = "") as csv_file:

    csv_writer = csv.writer(csv_file, delimiter = ",", quotechar = '"', quoting=csv.QUOTE_MINIMAL)

    csv_writer.writerow([
        'Image Name', 'LoG Variance', 'LoG Range', 'Canny Percentage Features', 'FFT Magnitude',
        'Specularities', '%', 'Clean Specularities', '%', 'Adequate Contrast'
    ])

    for subfolder in os.listdir(main_directory):
        
        subfolder_path = os.path.join(main_directory, subfolder)
        
        if os.path.isdir(subfolder_path):
            
            for file_name in os.listdir(subfolder_path):
                
                if file_name.lower().endswith((".bmp")):
                    
                    image_path = os.path.join(subfolder_path, file_name)
                    image = cv2.imread(image_path)

                    if image is not None:

                        number_of_images += 1
                        
                        laplace = LaplacianAnalysis(image)
                        
                        LoGVar = laplace.calculate_variance()
                        LoGQR = laplace.calculate_quantile_range()
                        
                        LoG_Variance.append(LoGVar)
                        LoG_Range.append(LoGQR)

                        canny = CannyEdgeAnalysis(image)
                        percentage_features = canny.calculate_features()
                        
                        percentages.append(percentage_features)

                        canny_edge_path = os.path.join(canny_folder, f"{file_name}.png")
                        cv2.imwrite(canny_edge_path, canny.edges)

                        sobel = SobelGradientMaps(image)

                        sobel_map_path = os.path.join(gradient_maps_folder, f"{file_name}.png")

                        plt.imshow(sobel.gradient_map(), cmap = "gray")
                        plt.axis("off")
                        plt.savefig(sobel_map_path, bbox_inches = "tight", pad_inches = 0.1)
                        plt.close()

                        FFT = FastFourierTransform(image)
                        magnitude = FFT.calculate_FFT()
                        
                        fft_magnitudes.append(magnitude)

                        image_contrast = loadImage(image).adequate_contrast

                        image_spec = SpectralReflection(image)

                        if (image_spec.detected):
                            detected_spectral_reflection += 1
                            percentage_pixels.append(image_spec.percentage)
                            specularities = "True"
                        else:
                            specularities = "False"
                        
                        if (image_spec.detected_clean):
                            clean_detected_spectral_reflection += 1
                            percentage_pixels_clean.append(image_spec.percentage_clean)
                            clean_specularities = "True"
                        else:
                            clean_specularities = "False"
                        
                        specularity_path = os.path.join(specular_reflection_folder, f"{file_name}.png")
                        plt.imshow(image_spec.bin_image, cmap = "gray")
                        plt.axis("off")
                        plt.savefig(specularity_path, bbox_inches = "tight", pad_inches = 0.1)
                        plt.close()
                    
                        if image_contrast:
                            number_of_images_good_quality_images +=1
                            contrast_quality = "True"
                        else:
                            contrast_quality = "False"
                        
                        csv_writer.writerow([
                            file_name, LoGVar, LoGQR, percentage_features, magnitude,
                            specularities, image_spec.percentage,
                            clean_specularities, image_spec.percentage_clean,
                            contrast_quality
                        ])

                        
print("\nDS3")

print("\nStatistics on LoG Variance ----------------------------------------------------\n")
descriptive_statistics(LoG_Variance)

print("\nStatistics on LoG Range  ----------------------------------------------------\n")
descriptive_statistics(LoG_Range)

print("\nStatistics on Percentage of Detectable Features  ----------------------------------------------------\n")
descriptive_statistics(percentages)

print("\nStatistics on Mean Magnitude of Representations by FFT  ----------------------------------------------------\n")
descriptive_statistics(fft_magnitudes)

print("\nStatistics on the Number of Pixels presenting Specular Reflections  ----------------------------------------------------\n")
descriptive_statistics(percentage_pixels, percentage = True)

print("\nStatistics on the Number of Pixels presenting Specular Reflections after erosion  ----------------------------------------------------\n")
descriptive_statistics(percentage_pixels_clean, percentage = True)

specular_reflection_detection = detected_spectral_reflection / number_of_images
clean_specular_reflection_detection = clean_detected_spectral_reflection / number_of_images
print(
    f"\n\n{specular_reflection_detection * 100:.2f} %",
    "of images present piques in specular properties"
)
print(
    f"{clean_specular_reflection_detection * 100:.2f} % of images present",
    "piques in specular properties even after 2 erosion operations"
)

perc_good_images = number_of_images_good_quality_images / number_of_images
print(f"\n\n{perc_good_images * 100:.2f} % of images have sufficient contrast")