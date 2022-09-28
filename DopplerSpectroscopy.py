# -*- coding: utf-8 -*-
"""

PHYS20161 Final Assignment: Doppler spectroscopy

The program calculates the mass of an extrasolar planet by examining a varying
Doppler shift in the emitted light of a star, given by two data files. It also
calculates the magnitude and angular frequency of the star's velocity, as well
as distance between the planet and the star. Results are illustrated by chi
squared contour and star's velocity sinusoidal fit plots.

Erik Germanovic 13/12/2020

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin
from scipy.optimize import curve_fit

# SI units

EMITTED_WAVELENGTH = 656.281e-9
ANGULAR_FREQUENCY_START = 3e-8
SPEED_OF_LIGHT = 3e8
MASS_OF_STAR = 5.52942e30
GRAVITATIONAL_CONSTANT = 6.674e-11
INITIAL_PHASE = np.pi
JOVIAN_MASS = 1.898e27
LINE_OF_SIGHT_ANGLE = np.pi / 2
ASTRONOMICAL_UNIT = 149597870700

FILE_NAME_1 = 'doppler_data_1.csv'
FILE_NAME_2 = 'doppler_data_2.csv'

def file_check():
    """
    Checks whether file names and locations are correct

    Outputs True or False based on whether files are found or not

    Erik Germanovic 13/12/2020
    """
    try:
        file_1 = open(FILE_NAME_1, 'r')
        file_1.close()
        file_2 = open(FILE_NAME_2, 'r')
        file_2.close()
        return True
    except FileNotFoundError:
        return False

def data_reading_function():
    """
    Reads and combines data from two csv files

    Outputs 2D float array (first column contains time data, second column
    contains wavelength data, third column contains wavelength uncertainty
    data)

    Erik Germanovic 13/12/2020
    """
    data_1 = np.genfromtxt(FILE_NAME_1, delimiter=',', comments='%')
    data_2 = np.genfromtxt(FILE_NAME_2, delimiter=',', comments='%')
    return np.concatenate((data_1, data_2))

def data_validation_function(all_data):
    """
    Validates combined data from two csv files

    all_data (2D float numpy array)

    Outputs 2D float array (first column contains time data, second column
    contains wavelength data, third column contains wavelength uncertainty
    data)

    Erik Germanovic 13/12/2020
    """
    index_array_1 = np.where(np.isnan(all_data))
    index_array_2 = np.where(all_data[:, 2] == 0)
    index_array_3 = np.where(np.isinf(all_data))
    index_array_total = np.concatenate((index_array_1[0], index_array_2[0],
                                        index_array_3[0]))
    all_data = np.delete(all_data, index_array_total, 0)

    index_array_4 = np.where(all_data < 0)
    all_data = np.delete(all_data, index_array_4[0], 0)
    return all_data

def unit_conversion(all_data):
    """
    Converts units to meters and seconds of combined data

    all_data (2D float numpy array)

    Outputs 2D float array (first column contains time data, second column
    contains wavelength data, third column contains wavelength uncertainty
    data)

    Erik Germanovic 13/12/2020
    """
    all_data[:, 0] = all_data[:, 0] * 365 * 24 * 60 * 60
    all_data[:, 1] = all_data[:, 1] * 10 ** -9
    all_data[:, 2] = all_data[:, 2] * 10 ** -9
    return all_data

def extreme_outlier(all_data):
    """
    Finds and deletes outliers which are more than 5 standard deviations away
    from average wavelength value until no outliers are found in that limit

    all_data (2D float numpy array)

    Outputs 2D float array (first column contains time data, second column
    contains wavelength data, third column contains wavelength uncertainty
    data)

    Erik Germanovic 13/12/2020
    """
    standard_deviation_limit = 5
    average_wavelength = np.average(all_data[:, 1])
    standard_deviation = np.std(all_data[:, 1])
    index_array = np.where(abs(all_data[:, 1]) > abs(
        average_wavelength + standard_deviation_limit * standard_deviation))
    check_count = 1
    for index in index_array[0]:
        print('After {0:d} validation data corresponding to wavelength of '
              '{1:6.3f} nm was omitted due to being more than {2:g} standard '
              'deviations away from average observed wavelength value\n'
              .format(check_count, all_data[:, 1][index],
                      standard_deviation_limit))
    all_data = np.delete(all_data, index_array, 0)
    data_count = len(all_data)
    data_count_test = data_count - 1
    while data_count_test < data_count:
        data_count = len(all_data)
        check_count = check_count + 1
        average_wavelength = np.average(all_data[:, 1])
        standard_deviation = np.std(all_data[:, 1])
        index_array = np.where(abs(all_data[:, 1]) > abs(
            average_wavelength + standard_deviation_limit *
            standard_deviation))
        for index in index_array[0]:
            print('After {0:d} validations data corresponding to wavelength of'
                  ' {1:6.3f} nm was omitted due to being more than {2:g} '
                  'standard deviations away from average observed wavelength '
                  'value\n'.format(check_count, all_data[:, 1][index],
                                   standard_deviation_limit))
        all_data = np.delete(all_data, index_array, 0)
        data_count_test = len(all_data)
    return all_data

def star_velocity_function(wavelength, wavelength_uncertainty):
    """
    Calculates velocity and velocity uncertainty of the star

    wavelength (float numpy array)
    wavelength_uncertainty (float numpy array)

    Outputs velocity and velocity uncertainty of the star (velocity_of_star:
    float numpy array, velocity_uncertainty_of_star: float numpy array)

    Erik Germanovic 13/12/2020
    """
    velocity_of_star = (((wavelength * SPEED_OF_LIGHT) / EMITTED_WAVELENGTH)
                        - SPEED_OF_LIGHT) * (1 / np.sin(LINE_OF_SIGHT_ANGLE))
    velocity_uncertainty_of_star = wavelength_uncertainty * SPEED_OF_LIGHT / (
        EMITTED_WAVELENGTH * np.sin(LINE_OF_SIGHT_ANGLE))
    return velocity_of_star, velocity_uncertainty_of_star

def fit_function(time_of_measurement, velocity_magnitude, angular_frequency):
    """
    Calculates star's velocity at different times using sinusoidal function

    time_of_measurement (float numpy array)
    velocity_magnitude (float)
    angular_frequency (float)

    Outputs float

    Erik Germanovic 13/12/2020
    """
    return velocity_magnitude * np.sin(angular_frequency * time_of_measurement
                                       + INITIAL_PHASE)

def chi_squared_function(velocity_and_frequency, velocity_of_star,
                         time_of_measurement, velocity_uncertainty_of_star):
    """
    Calculates chi-squared value for a given magnitude of velocity and angular
    frequency of a star

    velocity_and_frequency (tuple)
    velocity_of_star (float numpy array)
    time_of_measurement (float numpy array)
    velocity_uncertainty_of_star (float numpy array)

    Outputs float

    Erik Germanovic 13/12/2020
    """
    velocity = velocity_and_frequency[0]
    angular_frequency = velocity_and_frequency[1]
    return np.sum(((velocity_of_star -
                    fit_function(time_of_measurement, velocity,
                                 angular_frequency)) /
                   velocity_uncertainty_of_star) ** 2)

def fit_parameter_function(velocity_of_star, time_of_measurement,
                           velocity_uncertainty_of_star, velocity_start):
    """
    Finds best fit parameters of velocity magnitude and angular frequency by
    minimizing chi squared values of sinusoidal fit

    velocity_of_star (float numpy array)
    time_of_measurement (float numpy array)
    velocity_uncertainty_of_star (float numpy array)
    velocity_start (float)

    Outputs velocity magnitude and angular frequency of star, minimum
    chi-squared of the fit (fit_results[0][0]: float, fit_results[0][1]: float,
    fit_results[1] float)

    Erik Germanovic 13/12/2020
    """
    fit_results = fmin(
        chi_squared_function, (velocity_start, ANGULAR_FREQUENCY_START),
        args=(velocity_of_star, time_of_measurement,
              velocity_uncertainty_of_star), full_output=True, disp=False)
    return fit_results[0][0], fit_results[0][1], fit_results[1]

def fit_parameter_uncertainty(time_of_measurement, velocity_of_star,
                              velocity_uncertainty_of_star, velocity_start,
                              angular_frequency_start):
    """
    Calculates uncertainties of best fit parameters of sinusoidal fit using
    curve_fit

    time_of_measurement (float numpy array)
    velocity_of_star (float numpy array)
    velocity_uncertainty_of_star (float numpy array)
    velocity_start (float)

    Outputs array of parameter uncertainties (uncertainty_array: float numpy
    array)

    Erik Germanovic 13/12/2020
    """
    uncertainty_covariance_matrix = curve_fit(
        fit_function, time_of_measurement, velocity_of_star,
        sigma=velocity_uncertainty_of_star, p0=[velocity_start,
                                                angular_frequency_start])[1]
    uncertainty_array = np.sqrt(np.diag(uncertainty_covariance_matrix))
    return uncertainty_array

def non_extreme_outlier(velocity_magnitude, angular_frequency, velocity_of_star
                        , velocity_uncertainty_of_star, all_data):
    """
    Finds and deletes outliers which are 5 standard deviations away from star
    velocity fit line

    velocity_magnitude (float)
    angular_frequency (float)
    velocity_of_star (float numpy array)
    velocity_uncertainty_of_star (float numpy array)
    all_data (2D float numpy array)

    Outputs validated data and deleted point information
    (all_data: 2D float numpy array, velocity_of_star: float numpy array,
    velocity_uncertainty_of_star: float numpy array, outlier_array: 2D float
    numpy array)

    Erik Germanovic 13/12/2020
    """
    standard_deviation_limit = 5
    outlier_time = np.array([])
    outlier_velocity = np.array([])
    outlier_uncertainty = np.array([])
    fit_velocity_array = fit_function(all_data[:, 0], velocity_magnitude,
                                      angular_frequency)
    index_array = np.where(abs(velocity_of_star - fit_velocity_array) >
                           standard_deviation_limit *
                           velocity_uncertainty_of_star)
    for index in index_array[0]:
        outlier_time = np.append(outlier_time, all_data[:, 0][index])
        outlier_velocity = np.append(outlier_velocity, velocity_of_star[index])
        outlier_uncertainty = np.append(outlier_uncertainty,
                                        velocity_uncertainty_of_star[index])
        print('Data corresponding to star velocity of {0:4.2f} m/s was omitted'
              ' due to being more than {1:g} standard deviations away from '
              'sinusoidal fit\n'.format(velocity_of_star[index],
                                        standard_deviation_limit))
    outlier_array = np.vstack((outlier_time, outlier_velocity,
                               outlier_uncertainty))
    velocity_of_star = np.delete(velocity_of_star, index_array)
    velocity_uncertainty_of_star = np.delete(velocity_uncertainty_of_star,
                                             index_array)
    all_data = np.delete(all_data, index_array[0], 0)
    return (all_data, velocity_of_star, velocity_uncertainty_of_star,
            outlier_array)

def reduced_chi_squared(minimum, velocity_of_star):
    """
    Calculates reduced chi squared value of star velocity sinusoidal fit

    minimum (float)
    velocity_of_star (float numpy array)

    Outputs float

    Erik Germanovic 13/12/2020
    """
    return minimum / (len(velocity_of_star) - 2)

def velocity_fit_plot(velocity_of_star, velocity_uncertainty_of_star,
                      time_of_measurement, outlier_array,
                      velocity_magnitude_and_frequency):
    """
    Plots star's velocity sinusoidal fit line with star velocity data points
    and non extreme emitted outliers

    velocity_of_star (float numpy array)
    velocity_uncertainty_of_star (float numpy array)
    time_of_measurement (float numpy array)
    outlier_array (2D numpy float array)
    velocity_magnitude_and_frequency (tuple)

    Outputs None

    Erik Germanovic 12/13/2020
    """
    velocity_magnitude = velocity_magnitude_and_frequency[0]
    angular_frequency = velocity_magnitude_and_frequency[1]
    time_array = np.linspace(0, np.max(time_of_measurement), 100)
    sinusoidal_fit_figure = plt.figure()
    axes = sinusoidal_fit_figure.add_subplot(111)
    axes.plot(time_array / (365 * 24 * 60 * 60),
              fit_function(time_array, velocity_magnitude, angular_frequency),
              c='k', label=r'$v_0$$sin$($\omega$$t$ + $\pi$)')
    axes.errorbar(time_of_measurement / (365 * 24 * 60 * 60), velocity_of_star,
                  velocity_uncertainty_of_star, fmt='D', c='y', alpha=0.6,
                  label='Calculated_velocities')
    axes.errorbar(outlier_array[0] / (365 * 24 * 60 * 60), outlier_array[1],
                  outlier_array[2], fmt='D', c='r', label='Omitted outliers')

    axes.set_title("Star's velocity sinusoidal fit", fontsize=15,
                   fontname='Arial', color='k')
    axes.legend(loc='lower right', fontsize=8)
    axes.set_xlabel('Time of measurement ($years$)', fontsize=15,
                    fontname='Arial')
    axes.set_ylabel('Velocity ($m/s$)', fontsize=15, fontname='Arial')
    plt.savefig('star_velocity_fit_figure.png', dpi=300)
    plt.show()

def mesh_arrays(angular_frequency, velocity_magnitude, velocity_of_star,
                velocity_uncertainty_of_star, time_of_measurement):
    """
    Creates mesh arrays for magnitude of velocity, frequency of star and
    chi-squared values of star velocity sinusoidal fit

    angular_frequency (float)
    velocity_magnitude (float)
    velocity_of_star (float numpy array)
    velocity_uncertainty_of_star (float numpy array)
    time_of_measurement (float numpy array)

    Outputs meshed data (frequency_mesh: 2D float numpy array, velocity_mesh:
    2D float numpy array, chi_squared_mesh: 2D float numpy array)

    Erik Germanovic 13/12/2020
    """
    velocity_array = np.linspace(velocity_magnitude - velocity_magnitude * 0.09
                                 , velocity_magnitude + velocity_magnitude *
                                 0.09, 100)
    frequency_array = np.linspace(angular_frequency - angular_frequency * 0.09,
                                  angular_frequency + angular_frequency * 0.09,
                                  100)
    frequency_mesh, velocity_mesh = np.meshgrid(frequency_array,
                                                velocity_array)
    chi_squared_mesh = np.zeros(velocity_mesh.shape)
    for row, frequency in enumerate(frequency_mesh):
        for column in range(len(frequency_mesh[0])):
            chi_squared_mesh[row][column] = (
                np.sum(((velocity_of_star -
                         fit_function(time_of_measurement,
                                      velocity_mesh[row][column],
                                      frequency[column])) /
                        velocity_uncertainty_of_star) ** 2))
    return frequency_mesh, velocity_mesh, chi_squared_mesh

def chi_squared_plot(plot_data, velocity_magnitude, angular_frequency,
                     minimum):
    """
    Plots chi-squared contours against magnitude of velocity and angular
    frequency of the star

    plot_data (3D float numpy array)
    velocity_magnitude (float)
    angular_frequency (float)
    minimum (float)

    Outputs None

    Erik Germanovic 13/12/2020
    """
    chi_squared_figure = plt.figure()
    axes = chi_squared_figure.add_subplot(111)
    axes.set_title(r'$\chi^2$ contours against parameters', fontsize=15,
                   fontname='Arial', color='k')
    axes.set_xlabel('Angular frequency $(rad/s)$', fontsize=15,
                    fontname='Arial')
    axes.set_ylabel('Velocity $(m/s)$', fontsize=15, fontname='Arial')
    axes.set_xlim((np.min(plot_data[0]), np.max(plot_data[0])))
    axes.set_ylim((np.min(plot_data[1]), np.max(plot_data[1])))
    axes.scatter(angular_frequency, velocity_magnitude, c='r')
    axes.contour(plot_data[0], plot_data[1], plot_data[2],
                 levels=[minimum + 1.00], linestyles='dashed', colors='k')
    chi_squared_levels = (minimum + 2.30, minimum + 5.99, minimum + 9.21)
    contour_plot = axes.contour(plot_data[0], plot_data[1],
                                plot_data[2], levels=chi_squared_levels,
                                colors=['dodgerblue', 'limegreen', 'indigo'])
    labels = ['Minimum', r'$\chi^2_{{\mathrm{{min.}}}}+1.00$',
              r'$\chi^2_{{\mathrm{{min.}}}}+2.30$',
              r'$\chi^2_{{\mathrm{{min.}}}}+5.99$',
              r'$\chi^2_{{\mathrm{{min.}}}}+9.21$']
    axes.clabel(contour_plot)
    box = axes.get_position()
    axes.set_position([box.x0, box.y0, box.width * 0.75, box.height])
    for index, label in enumerate(labels):
        axes.collections[index].set_label(label)
        axes.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)
    plt.savefig('chi_squared_contour_plot.png', dpi=300)
    plt.show()

def distance_function(angular_frequency, angular_frequency_uncertainty):
    """
    Calculates distance between planet and star and its associated uncertainty

    fit_frequency (float)
    fit_frequency_uncertainty (float)

    Outputs distance: float, distance_uncertainty: float

    Erik Germanovic 13/12/2020
    """
    distance = (GRAVITATIONAL_CONSTANT * MASS_OF_STAR /
                angular_frequency ** 2) ** (1 / 3)
    distance_uncertainty = (2 / 3 * (GRAVITATIONAL_CONSTANT * MASS_OF_STAR) **
                            (1 / 3) / (angular_frequency) ** (5 / 3) *
                            angular_frequency_uncertainty)
    return distance, distance_uncertainty

def planet_velocity_function(distance, distance_uncertainty):
    """
    Calculates velocity of the planet and its associated uncertainty

    distance (float)
    distance_uncertainty (float)

    Outputs velocity: float, velocity_uncertainty: float

    Erik Germanovic 13/12/2020
    """
    velocity = (GRAVITATIONAL_CONSTANT * MASS_OF_STAR / distance) ** 0.5
    velocity_uncertainty = (0.5 * (GRAVITATIONAL_CONSTANT * MASS_OF_STAR) **
                            0.5 / (distance) ** 1.5 * distance_uncertainty)
    return velocity, velocity_uncertainty

def planet_mass_function(velocity_magnitude, velocity_magnitude_uncertainty,
                         velocity_of_planet, velocity_of_planet_uncertainty):
    """
    Calculates mass of the planet and its associated uncertainty

    velocity_magnitude (float)
    velocity_magnitude_uncertainty (float)
    velocity_of_planet (float)
    velocity_of_planet_uncertainty (float)

    Outputs mass: float, mass_uncertainty: float

    Erik Germanovic 13/12/2020
    """
    mass = MASS_OF_STAR * velocity_magnitude / (velocity_of_planet *
                                                JOVIAN_MASS)
    mass_uncertainty = mass * ((velocity_magnitude_uncertainty /
                                velocity_magnitude) ** 2 +
                               (velocity_of_planet_uncertainty/
                                velocity_of_planet) ** 2) ** 0.5
    return mass, mass_uncertainty

def angular_frequency_printing(angular_frequency,
                               angular_frequency_uncertainty):
    """
    Finds the correct format of angular frequency uncertainty by adjusting to
    the format of angular frequency and prints the result

    angular_frequency (float)
    angular_frequency_uncnertainty (float)

    Outputs None

    Erik Germanovic 13/12/2020
    """
    power_of_frequency = int(np.floor(np.log10(angular_frequency)))
    coefficient_frequency = (angular_frequency / float(
        10 ** power_of_frequency))
    coefficient_frequency_uncertainty = (angular_frequency_uncertainty /
                                         float(10 ** power_of_frequency))
    print("Angular frequency of star was found to be "
          "{0:.3f}e{1:d} ± {2:.3f}e{1:d} rad/s\n".format(
              coefficient_frequency, power_of_frequency,
              coefficient_frequency_uncertainty))

def main():
    """
    Main code for programme. Calls functions and prints results.

    Outputs None

    Erik Germanovic 13/12/2020
    """
    if np.sin(LINE_OF_SIGHT_ANGLE) == 0:
        print('It is not possible to observe Doppler shift with such line of '
              'sight angle')
    elif file_check():
        combined_data = data_reading_function()
        combined_data = data_validation_function(combined_data)
        combined_data = extreme_outlier(combined_data)
        combined_data = unit_conversion(combined_data)

        star_velocity, star_velocity_uncertainty = star_velocity_function(
            combined_data[:, 1], combined_data[:, 2])

        (star_velocity_magnitude, star_angular_frequency,
         minimum_chi_squared) = (
             fit_parameter_function(star_velocity, combined_data[:, 0],
                                    star_velocity_uncertainty,
                                    np.max(abs(star_velocity))))

        combined_data, star_velocity, star_velocity_uncertainty, outliers = (
            non_extreme_outlier(star_velocity_magnitude, star_angular_frequency
                                , star_velocity, star_velocity_uncertainty,
                                combined_data))

        (star_velocity_magnitude, star_angular_frequency,
         minimum_chi_squared) = (
             fit_parameter_function(star_velocity, combined_data[:, 0],
                                    star_velocity_uncertainty,
                                    np.max(abs(star_velocity))))

        (star_velocity_magnitude_uncertainty,
         star_angular_frequency_uncertainty) = (fit_parameter_uncertainty(
             combined_data[:, 0], star_velocity, star_velocity_uncertainty,
             star_velocity_magnitude, star_angular_frequency))

        velocity_fit_plot(star_velocity, star_velocity_uncertainty,
                          combined_data[:, 0], outliers,
                          (star_velocity_magnitude, star_angular_frequency))
        mesh_data = mesh_arrays(star_angular_frequency, star_velocity_magnitude
                                , star_velocity, star_velocity_uncertainty,
                                combined_data[:, 0])
        chi_squared_plot(mesh_data, star_velocity_magnitude,
                         star_angular_frequency, minimum_chi_squared)

        planet_distance, planet_distance_uncertainty = distance_function(
            star_angular_frequency, star_angular_frequency_uncertainty)
        planet_velocity, planet_velocity_uncertainty = (
            planet_velocity_function(planet_distance,
                                     planet_distance_uncertainty))
        planet_mass, planet_mass_uncertainty = (
            planet_mass_function(star_velocity_magnitude,
                                 star_velocity_magnitude_uncertainty,
                                 planet_velocity, planet_velocity_uncertainty))
        print("Chi-squared reduced of star's velocity sinusoidal fit was"
              " determined to be {0:4.3f}\n".format(reduced_chi_squared(
                  minimum_chi_squared, star_velocity)))
        print("Magnitude of star's velocity was found to be {0:4.2f} ±"
              " {1:4.2f} m/s\n".format(star_velocity_magnitude,
                                       star_velocity_magnitude_uncertainty))
        angular_frequency_printing(star_angular_frequency,
                                   star_angular_frequency_uncertainty)
        print('Mass of the planet was found to be {0:4.3f} ± {1:4.3f}'
              ' Jovian masses\n'.format(planet_mass, planet_mass_uncertainty))
        print('Distance between planet and star was found to be {0:4.3f} ± '
              '{1:4.3f} AU'.format(
                  planet_distance / ASTRONOMICAL_UNIT,
                  planet_distance_uncertainty / ASTRONOMICAL_UNIT))
    else:
        print("Check whether data files' names and locations are correct.")
main()
